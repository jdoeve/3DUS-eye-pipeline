"""QC checks and QC image generation."""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DEFAULT_THETA_SPAN_DEG


def pca_axis_ratio_of_volume(vol: np.ndarray, thresh_frac=0.6) -> float:
    t = float(np.percentile(vol, 100 * thresh_frac))
    coords = np.argwhere(vol > t)
    if coords.shape[0] < 500:
        return 1.0
    cov = np.cov(coords.T)
    w, _ = np.linalg.eigh(cov)
    w = np.sort(w)
    return float(np.sqrt(w[-1] / w[0])) if w[0] > 0 else 1.0


def compare_edge_vs_threshold_radius(unwrapped, geom, depth_mm=24.0, verbose=True):
    k = unwrapped.shape[0] // 2
    r_idx = int(round(depth_mm / geom.dr_mm))
    if r_idx >= unwrapped.shape[1]:
        return None

    row = unwrapped[k, r_idx, :].astype(np.float32)
    if row.max() <= 0:
        return None

    t = 0.5 * (row.max() + row.min())
    cols = np.where(row > t)[0]
    if cols.size < 2:
        return None
    thr_span = cols.max() - cols.min()

    r = (row - row.min()) / (np.ptp(row) + 1e-6)
    r8 = (255 * r).astype(np.uint8).reshape(1, -1)
    edges = cv2.Canny(r8, 50, 150).flatten() > 0
    ecols = np.where(edges)[0]
    edge_span = thr_span if ecols.size < 2 else (ecols.max() - ecols.min())

    mm_per_px = depth_mm * geom.delta_phi_per_px
    thr_mm = thr_span * mm_per_px
    edge_mm = edge_span * mm_per_px

    if verbose:
        print(
            f"[SEG-QC] depth={depth_mm:.1f}mm  thr_diam={thr_mm:.2f}mm  "
            f"edge_diam={edge_mm:.2f}mm  diff={thr_mm-edge_mm:+.2f}mm"
        )
    return thr_mm, edge_mm


def qc_theta_even_odd(imgs, geom, mask_path, unwrap_fn, resample_fn, voxel_mm=0.30, verbose=True):
    uw_all, _ = unwrap_fn(imgs, geom, mask_path, drop_duplicate=True, verbose=False)
    n = uw_all.shape[0]
    idx_even = np.arange(0, n, 2)
    idx_odd = np.arange(1, n, 2)

    theta_span = DEFAULT_THETA_SPAN_DEG
    vol_all = resample_fn(uw_all, geom, voxel_mm, sigma_theta_deg=0.0, theta_span_deg=theta_span, interp_order=1, slab_size=48, verbose=False)
    vol_even = resample_fn(uw_all[idx_even], geom, voxel_mm, sigma_theta_deg=0.0, theta_span_deg=theta_span, interp_order=1, slab_size=48, verbose=False)
    vol_odd = resample_fn(uw_all[idx_odd], geom, voxel_mm, sigma_theta_deg=0.0, theta_span_deg=theta_span, interp_order=1, slab_size=48, verbose=False)

    ar_all = pca_axis_ratio_of_volume(vol_all)
    ar_even = pca_axis_ratio_of_volume(vol_even)
    ar_odd = pca_axis_ratio_of_volume(vol_odd)

    if verbose:
        print(f"[QC-θ] axis_ratio all={ar_all:.3f} even={ar_even:.3f} odd={ar_odd:.3f}")
        drift = max(abs(ar_even - ar_all), abs(ar_odd - ar_all))
        print("[QC-θ] >2% drift detected — consider θ-from-image regrid." if drift > 0.02 else "[QC-θ] OK (≤2%).")

    return ar_all, ar_even, ar_odd


def validate_calibration_qc(cal: dict, dr_mm: float, verbose: bool = True):
    if not verbose:
        return

    delta_phi = cal.get("delta_phi_per_px")
    phi0 = cal.get("phi0_rad")
    phi1 = cal.get("phi1_rad")

    if delta_phi is None or phi0 is None or phi1 is None:
        print("[QC] Skipping calibration QC - missing geometry parameters")
        return

    print("\n" + "=" * 60)
    print("CALIBRATION QC CHECKS")
    print("=" * 60)

    print("\n[QC Test 1] Circle test - arc length at different radii:")
    for r_mm in [10.0, 20.0, 30.0, 40.0]:
        arc_length_per_px = r_mm * delta_phi
        total_arc = r_mm * (phi1 - phi0)
        print(
            f"  r={r_mm:5.1f}mm: arc_per_px={arc_length_per_px:.4f}mm, "
            f"total_arc={total_arc:.2f}mm, fan_angle={np.degrees(phi1-phi0):.1f}°"
        )

    print("\n[QC Test 2] Expected sphere measurements:")
    print("  A 7.2mm diameter sphere should measure:")
    print("    - Radial (depth): ~7.2mm (with SOS correction)")
    print("    - Lateral (arc): depends on depth")

    r_mid = 24.0
    sphere_diameter = 7.2
    angular_span_rad = sphere_diameter / r_mid
    angular_span_deg = np.degrees(angular_span_rad)
    lateral_pixels = angular_span_rad / delta_phi

    print(f"  At r={r_mid}mm depth:")
    print(f"    - Angular span: {angular_span_deg:.2f}° ({angular_span_rad:.4f} rad)")
    print(f"    - Lateral pixels: {lateral_pixels:.1f} px")
    print(f"    - Arc length: {sphere_diameter:.2f}mm")

    print("\n[QC Test 3] Resolution comparison:")
    print(f"  Radial pixel spacing: {dr_mm:.4f} mm/px (constant)")
    print(f"  Lateral spacing at r=10mm: {10.0 * delta_phi:.4f} mm/px")
    print(f"  Lateral spacing at r=20mm: {20.0 * delta_phi:.4f} mm/px")
    print(f"  Lateral spacing at r=30mm: {30.0 * delta_phi:.4f} mm/px")
    print(f"  Lateral spacing at r=40mm: {40.0 * delta_phi:.4f} mm/px")
    print("  NOTE: Lateral resolution degrades (spacing increases) with depth")
    print("=" * 60 + "\n")


def generate_qc(vol, qc_dir: Path, unwrapped=None, mask_unwrapped=None):
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    vmin = float(np.percentile(vol, 60.0))
    vmax = float(np.percentile(vol, 99.7))

    mip_xy = vol.max(axis=2).T
    mip_xz = vol.max(axis=1).T
    mip_yz = vol.max(axis=0).T

    x_signal = vol.sum(axis=(1, 2))
    cx = int(np.argmax(x_signal))
    cy = vol.shape[1] // 2
    cz = vol.shape[2] // 2

    slc_x = vol[cx, :, :].T
    slc_y = vol[:, cy, :].T
    slc_z = vol[:, :, cz].T

    plt.imsave(qc_dir / "mip_xy.png", mip_xy, vmin=vmin, vmax=vmax, origin="lower", cmap="gray")
    plt.imsave(qc_dir / "mip_xz.png", mip_xz, vmin=vmin, vmax=vmax, origin="lower", cmap="gray")
    plt.imsave(qc_dir / "mip_yz.png", mip_yz, vmin=vmin, vmax=vmax, origin="lower", cmap="gray")

    plt.imsave(qc_dir / f"slice_x_{cx}.png", slc_x, vmin=vmin, vmax=vmax, origin="lower", cmap="gray")
    plt.imsave(qc_dir / f"slice_y_{cy}.png", slc_y, vmin=vmin, vmax=vmax, origin="lower", cmap="gray")
    plt.imsave(qc_dir / f"slice_z_{cz}.png", slc_z, vmin=vmin, vmax=vmax, origin="lower", cmap="gray")

    if unwrapped is not None:
        k = int(np.floor(unwrapped.shape[0] / 2))
        plt.imsave(qc_dir / "unwrapped_mid.png", unwrapped[k], cmap="gray")

    if mask_unwrapped is not None:
        plt.imsave(qc_dir / "mask_unwrapped.png", mask_unwrapped, cmap="gray")

    print(f"QC images saved to: {qc_dir}")
