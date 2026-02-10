"""Pipeline orchestration for UC-Eye reconstruction."""

import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import SimpleITK as sitk

from .calibration import calibrate_from_mask, microsearch_apex, qc_row_arc_against_mask
from .config import (
    DEFAULT_DEPTH_MM,
    DEFAULT_INTERP_ORDER,
    DEFAULT_SLAB_SIZE,
    DEFAULT_SIGMA_THETA_DEG,
    DEFAULT_THETA_SPAN_DEG,
    DEFAULT_VOXEL_MM,
    DO_QC,
    SOS_SCALE,
)
from .geometry import GeometryConfig
from .io_utils import load_numeric_sorted_images, timer
from .io_utils import generate_iterative_filename
from .qc import compare_edge_vs_threshold_radius, generate_qc, qc_theta_even_odd
from .recon import resample_to_cartesian
from .registration import register_frames_level1
from .seam import (
    adjacent_corr_score,
    apply_circular_shift_best_cut,
    closure_blend,
    enforce_closure_drift,
    enforce_strictly_increasing,
    find_best_cut,
    roi_for_similarity,
)
from .unwrap import unwrap_frames


def run_pipeline(
    img_dir: Path,
    mask_path: Path | None,
    output_path: Path = None,
    depth_mm: float = DEFAULT_DEPTH_MM,
    voxel_mm: float = DEFAULT_VOXEL_MM,
    sigma_theta_deg: float = DEFAULT_SIGMA_THETA_DEG,
    theta_span_deg: float = DEFAULT_THETA_SPAN_DEG,
    interp_order: int = DEFAULT_INTERP_ORDER,
    slab_size: int = DEFAULT_SLAB_SIZE,
    do_registration: bool = True,
    drop_duplicate: bool = True,
    verbose: bool = True,
    out_dir_for_iter: Path | None = None,
):
    print("=" * 60)
    print("UC-Eye Radial Ultrasound Pipeline (CALIBRATED)")
    print("=" * 60)

    img_dir = Path(img_dir)
    mask_path = Path(mask_path) if mask_path is not None else None
    if output_path is not None:
        output_path = Path(output_path)

    if not img_dir.exists() or not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found or not a directory: {img_dir}")
    if mask_path is not None and not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    with timer("Load frames", verbose):
        files = load_numeric_sorted_images(img_dir)
        if not files:
            raise ValueError(f"No images found in {img_dir}")
        imgs = [iio.imread(p) for p in files]
        if verbose:
            print(f"  Loaded {len(imgs)} frames")

    mask_cal_path = img_dir.parent / "mask_detection.json"
    if mask_path is None:
        h_img, w_img = imgs[0].shape[:2]
        mask_params = {
            "apex_xy": (0.0, float(h_img) / 2.0),
            "r_max_pix": float(max(w_img - 1, 1)),
            "image_height": int(h_img),
            "image_width": int(w_img),
            # Fallback uses device fan defaults in GeometryConfig.
            "phi0_mask": None,
            "phi1_mask": None,
            # Effective rows fallback to full image rows.
            "n_rows_mask": int(h_img),
        }
        if verbose:
            print("[MASK] No mask provided; using device-default fan geometry fallback.")
    elif mask_cal_path.exists():
        with open(mask_cal_path, "r") as f:
            mask_params = json.load(f)
        if verbose:
            print(f"[MASK] Loaded mask detection from {mask_cal_path}")
    else:
        with timer("Detect mask parameters", verbose):
            mask_params = calibrate_from_mask(mask_path, verbose=verbose)
        with open(mask_cal_path, "w") as f:
            json.dump(mask_params, f, indent=2)
        if verbose:
            print(f"[MASK] Saved mask detection to {mask_cal_path}")

    geom = GeometryConfig(
        image_height=mask_params["image_height"],
        r_max_pix=mask_params["r_max_pix"],
        apex_xy=tuple(mask_params["apex_xy"]),
        depth_mm=depth_mm,
        n_rows_mask=mask_params.get("n_rows_mask"),
        phi0_mask=mask_params.get("phi0_mask"),
        phi1_mask=mask_params.get("phi1_mask"),
    )

    if verbose:
        geom.print_summary()

    if DO_QC and verbose and mask_path is not None:
        with timer("QC: Row-arc vs mask", verbose):
            mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            qc_row_arc_against_mask(imgs[len(imgs) // 2], geom, mask_gray, depths_mm=(10, 20, 30, 40), verbose=verbose)

    with timer("Apex micro-search", verbose):
        geom = microsearch_apex(
            unwrap_frames,
            imgs,
            geom,
            mask_path,
            dxdy_range=(-2, 2),
            step=1,
            depths_mm=(12, 24, 36),
            verbose=verbose,
        )

    with timer("Unwrap frames", verbose):
        unwrapped, mask_unwrapped = unwrap_frames(imgs, geom, mask_path, drop_duplicate=drop_duplicate, verbose=verbose)
        if verbose:
            print(f"  Unwrapped shape: {unwrapped.shape} (N, r, phi)")

    ntheta = unwrapped.shape[0]
    theta_span_rad = np.deg2rad(theta_span_deg)
    theta_k = np.linspace(0.0, theta_span_rad, ntheta, endpoint=False, dtype=np.float32)

    if do_registration:
        with timer("Register φ across frames", verbose):
            unwrapped, drifts_r, drifts_phi = register_frames_level1(unwrapped, verbose=verbose)
            phi_shifts = np.round(drifts_phi).astype(np.int32)
    else:
        drifts_r = np.zeros(unwrapped.shape[0], dtype=np.float32)
        drifts_phi = np.zeros(unwrapped.shape[0], dtype=np.float32)
        phi_shifts = np.zeros(unwrapped.shape[0], dtype=np.int32)

    enable_best_cut_shift = True
    enable_closure_blend = True
    blend_k = 5

    r_sl = roi_for_similarity(unwrapped, geom, depth_mm_lo=15.0, depth_mm_hi=35.0)
    closure_before = adjacent_corr_score(unwrapped[-1, r_sl, ::4], unwrapped[0, r_sl, ::4])
    if verbose:
        print(f"[seam] closure corr BEFORE shift/blend: {closure_before:.4f}")

    if enable_best_cut_shift:
        k_cut, k_score = find_best_cut(unwrapped, geom, depth_mm_lo=15.0, depth_mm_hi=35.0, phi_stride=4, ignore_margin=2)
        if verbose:
            print(f"[seam] best cut k={k_cut} with adjacent corr={k_score:.4f}")

        unwrapped, theta_k, phi_shifts, drifts_r, drifts_phi = apply_circular_shift_best_cut(
            unwrapped,
            theta_k,
            phi_shifts,
            drifts_r,
            drifts_phi,
            k_cut,
            theta_span_rad,
            verbose=verbose,
        )

    if do_registration:
        unwrapped, drifts_r, drifts_phi = enforce_closure_drift(unwrapped, drifts_r, drifts_phi, verbose=verbose)

    ntheta = unwrapped.shape[0]
    theta_base = np.linspace(0.0, theta_span_rad, ntheta, endpoint=False, dtype=np.float32)

    if do_registration:
        phi_shifts = np.round(drifts_phi).astype(np.int32)
    else:
        phi_shifts = np.zeros(ntheta, dtype=np.int32)

    dphi = geom.delta_phi_per_px
    delta_theta = -phi_shifts.astype(np.float32) * dphi
    theta_k = theta_base + delta_theta

    theta_k -= theta_k.min()
    theta_k *= theta_span_rad / max(theta_k.max(), 1e-6)
    theta_k = enforce_strictly_increasing(theta_k)

    if enable_closure_blend:
        unwrapped = closure_blend(unwrapped, k=blend_k, verbose=verbose)

    closure_after = adjacent_corr_score(unwrapped[-1, r_sl, ::4], unwrapped[0, r_sl, ::4])
    if verbose:
        print(f"[seam] closure corr AFTER shift/blend: {closure_after:.4f}")

    if DO_QC and verbose:
        with timer("Segmentation bias check (2D)", verbose):
            _ = compare_edge_vs_threshold_radius(unwrapped, geom, depth_mm=24.0, verbose=verbose)
        with timer("QC: rotation even/odd", verbose):
            _ = qc_theta_even_odd(imgs, geom, mask_path, unwrap_frames, resample_to_cartesian, voxel_mm=0.30, verbose=verbose)

    print("[theta_k] min/max:", float(theta_k.min()), float(theta_k.max()))
    print("[theta_k] any non-increasing:", bool(np.any(np.diff(theta_k) <= 0)))
    print("[theta_k] max span (rad):", float(theta_k.max() - theta_k.min()))

    with timer("Resample to 3D Cartesian", verbose):
        vol = resample_to_cartesian(
            unwrapped,
            geom,
            voxel_mm,
            sigma_theta_deg,
            theta_span_deg,
            interp_order,
            slab_size,
            theta_k=theta_k,
            verbose=verbose,
        )

    if output_path is None:
        output_path = generate_iterative_filename(img_dir, out_dir_for_iter, verbose)

    with timer("Save", verbose):
        vol_zyx = np.transpose(vol, (2, 1, 0)).astype(np.float32)
        img = sitk.GetImageFromArray(vol_zyx)
        img.SetSpacing((voxel_mm, voxel_mm, voxel_mm))

        apex_physical = (vol.shape[0] // 2, vol.shape[1] // 2, 0)
        origin_mm = tuple(-apex_physical[i] * voxel_mm for i in range(3))
        img.SetOrigin(origin_mm)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(output_path))
        if verbose:
            print(f"  Saved: {output_path}")
            print(f"  Origin set to: {origin_mm} (apex near Z=0)")
            print(
                f"  [CAL NOTE] SOS_SCALE={SOS_SCALE:.5f}, "
                f"effective_depth_mm={geom.depth_mm_sos_corrected:.3f}mm, "
                f"voxel_mm={voxel_mm:.4f}mm"
            )

    qc_dir = output_path.parent / "QC"
    with timer("QC", verbose):
        generate_qc(vol, qc_dir, unwrapped, mask_unwrapped)

    print("=" * 60)
    print("Pipeline complete")
    print("=" * 60)
    return output_path
