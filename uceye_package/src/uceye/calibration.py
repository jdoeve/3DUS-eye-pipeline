"""Cone-mask calibration and apex search utilities."""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .geometry import GeometryConfig
from .io_utils import ang_norm, find_main_contour, intersect_lines, largest_component, line_fit_cv, to_gray


def qc_row_arc_against_mask(raw_img: np.ndarray, geom, mask_gray: np.ndarray, depths_mm=(10, 20, 30, 40), verbose=True):
    g = to_gray(raw_img).astype(np.uint8)
    _ = g
    mb = (mask_gray > 128).astype(np.uint8)
    apx_x, apx_y = geom.apex_xy
    dphi = geom.delta_phi_per_px
    dr = geom.dr_mm

    h, _w = mb.shape
    errs = []
    for r_mm in depths_mm:
        r_px = r_mm / dr
        phi_mid = 0.5 * (geom.phi0 + geom.phi1)
        y = int(round(apx_y + r_px * np.sin(phi_mid)))
        y = int(np.clip(y, 0, h - 1))
        row = mb[y, :]
        if row.sum() < 2:
            continue
        xs = np.where(row > 0)[0]
        n_px = xs.max() - xs.min() + 1
        l_exp = r_mm * (n_px * dphi)
        l_meas = n_px * (r_mm * dphi)
        err = 0.0 if l_exp == 0 else (l_meas - l_exp) / l_exp
        errs.append((r_mm, err))

    if verbose and errs:
        print("[QC-ROW-ARC] depth_mm  rel_error")
        for r_mm, err in errs:
            print(f"               {r_mm:7.1f}  {100.0*err:+6.2f}%")
    return errs


def microsearch_apex(
    unwrapped_builder,
    imgs,
    geom_seed,
    mask_path: Path,
    dxdy_range=(-3, 3),
    step=1,
    depths_mm=(12, 24, 36),
    verbose=True,
):
    mid = [imgs[len(imgs) // 2]]
    best = {"var": np.inf, "apex": geom_seed.apex_xy, "geom": geom_seed}
    apx_x0, apx_y0 = geom_seed.apex_xy

    for dx in range(dxdy_range[0], dxdy_range[1] + 1, step):
        for dy in range(dxdy_range[0], dxdy_range[1] + 1, step):
            apx = (apx_x0 + dx, apx_y0 + dy)
            geom = GeometryConfig(
                image_height=geom_seed.image_height,
                r_max_pix=geom_seed.r_max_pix,
                apex_xy=apx,
                depth_mm=geom_seed.depth_mm_displayed,
                n_rows_mask=geom_seed.image_height_eff,
                phi0_mask=geom_seed.phi0,
                phi1_mask=geom_seed.phi1,
            )
            uw, _ = unwrapped_builder(mid, geom, mask_path, drop_duplicate=False, verbose=False)

            radii = []
            for r_mm in depths_mm:
                r_idx = int(round((r_mm / geom.dr_mm)))
                if r_idx >= uw.shape[1]:
                    continue
                row = uw[0, r_idx, :]
                grad = np.abs(np.gradient(row.astype(float)))
                thr = 0.5 * np.max(grad) if grad.max() > 0 else 0
                edge_cols = np.where(grad > thr)[0]
                if edge_cols.size < 6:
                    continue
                radii.append(float(r_mm))

            if len(radii) >= 2:
                v = float(np.var(radii))
                if v < best["var"]:
                    best = {"var": v, "apex": apx, "geom": geom}

    if best["var"] == np.inf:
        if verbose:
            print(
                "[APEX-QA] WARNING: microsearch found no valid candidates (best var=inf). "
                "Falling back to seed apex."
            )
        return geom_seed

    if verbose:
        print(f"[APEX-QA] chosen apex {best['apex']} with var={best['var']:.6f}")
    return best["geom"]


def calibrate_from_mask(mask_path: Path, verbose: bool = True):
    mask_path = Path(mask_path)
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask_gray is not None, f"Mask not found: {mask_path}"
    mask_bin = largest_component((mask_gray > 128).astype(np.uint8))
    hm, wm = mask_bin.shape
    ys_all, xs_all = np.where(mask_bin > 0)

    xmin = int(xs_all.min())
    centers = []
    for x in range(xmin, min(xmin + 5, wm)):
        ys = np.where(mask_bin[:, x] > 0)[0]
        if ys.size:
            centers.append(0.5 * (ys.min() + ys.max()))
    assert centers, "Cannot estimate apex from mask."
    apx0 = (float(xmin), float(np.mean(centers)))

    edge_pts = find_main_contour(mask_bin)
    dx = edge_pts[:, 1] - apx0[0]
    dy = edge_pts[:, 0] - apx0[1]
    phi_all = np.arctan2(dy, dx)
    phi_lo, phi_hi = np.percentile(phi_all, [2.0, 98.0])

    tol = np.deg2rad(2.5)
    edge_a = edge_pts[np.abs(phi_all - phi_lo) < tol]
    edge_b = edge_pts[np.abs(phi_all - phi_hi) < tol]
    if len(edge_a) < 30 or len(edge_b) < 30:
        tol = np.deg2rad(4.0)
        edge_a = edge_pts[np.abs(phi_all - phi_lo) < tol]
        edge_b = edge_pts[np.abs(phi_all - phi_hi) < tol]

    l1 = line_fit_cv(edge_a)
    l2 = line_fit_cv(edge_b)
    apx = intersect_lines(l1, l2)
    assert apx, "Edge lines nearly parallel."
    apx_x, apx_y = apx

    ang1 = ang_norm(np.arctan2(l1[3], l1[2]))
    ang2 = ang_norm(np.arctan2(l2[3], l2[2]))
    phi0, phi1 = sorted([ang1, ang2])
    if (phi1 - phi0) > np.deg2rad(180):
        phi0, phi1 = phi1, phi0 + 2 * np.pi

    r = np.hypot(xs_all - apx_x, ys_all - apx_y)
    r_max_pix = float(np.percentile(r, 99.5))

    row_has_data = np.sum(mask_bin > 0, axis=1) > 0
    n_rows_mask = int(np.sum(row_has_data))

    phi0_mask = float(phi0)
    phi1_mask = float(phi1)
    fan_angle_rad_from_mask = phi1_mask - phi0_mask

    if verbose:
        print("\n[MASK DETECTION]")
        print(f"  Apex (x,y):     ({apx_x:.2f}, {apx_y:.2f}) px")
        print(f"  r_max:          {r_max_pix:.1f} px")
        print(f"  Fan angle:      {np.degrees(fan_angle_rad_from_mask):.1f}° (mask-detected)")
        print(f"  Image size:     {hm} × {wm} px")
        print(f"  Usable rows:    {n_rows_mask} px (rows with mask data)")

    try:
        with open(mask_path, "rb") as f:
            mask_hash = hashlib.sha1(f.read()).hexdigest()[:12]
    except Exception:
        mask_hash = ""

    return {
        "apex_xy": (float(apx_x), float(apx_y)),
        "r_max_pix": float(r_max_pix),
        "image_height": int(hm),
        "image_width": int(wm),
        "phi0_mask": phi0_mask,
        "phi1_mask": phi1_mask,
        "n_rows_mask": n_rows_mask,
        "mask_fan_angle_deg": float(np.degrees(fan_angle_rad_from_mask)),
        "mask_file": str(mask_path.name),
        "mask_sha1_12": mask_hash,
    }
