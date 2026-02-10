"""Cartesian reconstruction from unwrapped radial frames."""

import time

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from .config import DEBUG_PHI_PHANTOM, DEBUG_RADIAL_PHANTOM, DEBUG_THETA_PHANTOM
from .io_utils import fmt_t


def resample_to_cartesian(
    unwrapped,
    geom,
    voxel_mm: float,
    sigma_theta_deg: float,
    theta_span_deg: float,
    interp_order: int,
    slab_size: int,
    theta_k=None,
    theta_pad: int | None = None,
    verbose: bool = True,
):
    n, _r, _w = unwrapped.shape
    dr_mm = geom.dr_mm
    dphi = geom.delta_phi_per_px
    phi0 = geom.phi0
    phi1 = geom.phi1
    depth_mm = geom.depth_mm_sos_corrected

    v = np.transpose(unwrapped, (1, 2, 0)).astype(np.float32)
    rn, wang_v, ntheta = v.shape
    v = np.ascontiguousarray(v, dtype=np.float32)

    if theta_k is not None:
        theta_k = np.asarray(theta_k, dtype=np.float32)
        if theta_k.shape[0] != ntheta:
            raise ValueError(f"theta_k length {theta_k.shape[0]} does not match Ntheta {ntheta}")
        theta_k = _make_strictly_increasing(theta_k)
        theta_min = float(theta_k[0])
        theta_max = float(theta_k[-1])
    else:
        theta_min = 0.0
        theta_max = np.deg2rad(theta_span_deg)
        theta_k = np.linspace(theta_min, theta_max, ntheta, endpoint=False, dtype=np.float32)

    frame_idx_array = np.arange(ntheta, dtype=np.float32)
    theta_span_rad_eff = max(theta_max - theta_min, 1e-6)

    if DEBUG_THETA_PHANTOM:
        theta_vals = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False, dtype=np.float32)
        cos_vals = 0.5 * (np.cos(theta_vals) + 1.0)
        for k in range(ntheta):
            v[:, :, k] = cos_vals[k]
    elif DEBUG_PHI_PHANTOM:
        for j in range(wang_v):
            v[:, j, :] = float(j)
    elif DEBUG_RADIAL_PHANTOM:
        for i in range(rn):
            v[i, :, :] = float(i)

    if theta_pad is None:
        pad = 2 if interp_order >= 3 else 1
    else:
        pad = int(theta_pad)

    v = np.concatenate([v[:, :, -pad:], v, v[:, :, :pad]], axis=2).astype(np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    theta_k = theta_k.astype(np.float32, copy=False)
    theta_k_pad = np.concatenate(
        [theta_k[-pad:] - theta_span_rad_eff, theta_k, theta_k[:pad] + theta_span_rad_eff]
    ).astype(np.float32)

    rn, wang_v, ntheta = v.shape
    frame_idx_array = np.arange(ntheta, dtype=np.float32)
    theta_k = theta_k_pad
    theta_k_min = float(theta_k[0])
    theta_k_max = float(theta_k[-1])

    print(f"[theta_k padded] min={theta_k_min:.6f} max={theta_k_max:.6f} Ntheta={ntheta} PAD={pad}")

    did_smooth = False
    sigma_theta = 0.0
    if sigma_theta_deg > 0:
        did_smooth = True
        sigma_theta = sigma_theta_deg * ((ntheta - 1) / max(theta_span_deg, 1e-3))
        v = gaussian_filter1d(v, sigma=sigma_theta, axis=2, mode="wrap")

    _ = did_smooth
    _ = sigma_theta

    half_extent = depth_mm
    n_axis = int(np.ceil((2 * half_extent) / voxel_mm))
    xs = np.linspace(-half_extent, half_extent, n_axis, dtype=np.float32)
    ys = np.linspace(-half_extent, half_extent, n_axis, dtype=np.float32)
    zs = np.linspace(-half_extent, half_extent, n_axis, dtype=np.float32)

    nx = ny = nz = n_axis
    prefilter = interp_order >= 3

    if verbose:
        print(f"Output volume: ({nx}, {ny}, {nz}) at {voxel_mm:.3f} mm/voxel")
        print(f"[GEOMETRY] Fan angle: {geom.fan_angle_deg:.1f}° (device spec)")
        print(f"[GEOMETRY] Δφ per pixel: {dphi:.6f} rad/px")
        print(f"[GEOMETRY] Radial dr: {dr_mm:.4f} mm/px")
        print(f"[GEOMETRY] Depth (SoS corrected): {depth_mm:.3f} mm")

    y2, z2 = np.meshgrid(ys, zs, indexing="ij")
    rho2d = np.sqrt(y2**2 + z2**2).astype(np.float32)
    psi2d = np.mod(np.arctan2(z2, y2).astype(np.float32), 2 * np.pi).astype(np.float32)

    vol = np.zeros((nx, ny, nz), dtype=np.float32)
    n_slabs = int(np.ceil(nx / slab_size))
    coords_buf = None
    t0 = time.time()

    first_valid_s = None
    last_valid_s = None

    for s in range(n_slabs):
        x0 = s * slab_size
        x1 = min((s + 1) * slab_size, nx)
        xs_slab = xs[x0:x1][:, None, None]

        rho = rho2d[None, :, :]
        r_cart = np.sqrt(xs_slab**2 + rho**2)
        phi_cart = np.arctan2(rho, xs_slab)

        valid = (r_cart <= depth_mm) & (phi_cart >= phi0) & (phi_cart <= phi1)
        if int(np.count_nonzero(valid)) > 0:
            if first_valid_s is None:
                first_valid_s = s
            last_valid_s = s

    if first_valid_s is None:
        return vol

    processed = 0
    total_work_slabs = last_valid_s - first_valid_s + 1

    for s in range(first_valid_s, last_valid_s + 1):
        processed += 1
        x0 = s * slab_size
        x1 = min((s + 1) * slab_size, nx)
        xs_slab = xs[x0:x1][:, None, None]

        rho = rho2d[None, :, :]
        r_cart = np.sqrt(xs_slab**2 + rho**2)
        phi_cart = np.arctan2(rho, xs_slab)

        valid = (r_cart <= depth_mm) & (phi_cart >= phi0) & (phi_cart <= phi1)
        r_idx = r_cart / dr_mm
        phi_idx = (phi_cart - phi0) / dphi
        idx_valid = np.where(valid)

        if verbose:
            print(f"[Slab {s+1}/{n_slabs}] n_valid={idx_valid[0].size}")

        if idx_valid[0].size == 0:
            continue

        r_idx_v = np.clip(r_idx[idx_valid], 0, rn - 1 - 1e-3).astype(np.float32)
        phi_idx_v = np.clip(phi_idx[idx_valid], 0, wang_v - 1 - 1e-3).astype(np.float32)

        psi_v = psi2d[idx_valid[1], idx_valid[2]]
        theta_virtual_v = (psi_v / (2 * np.pi)) * theta_span_rad_eff + theta_min
        theta_virtual_v = np.clip(theta_virtual_v.astype(np.float32, copy=False), theta_k_min, theta_k_max)
        theta_idx_v = np.interp(theta_virtual_v, theta_k, frame_idx_array)
        theta_idx_v = np.clip(theta_idx_v, 0.0, (ntheta - 1.0) - 1e-6)

        npts = r_idx_v.size
        if coords_buf is None or coords_buf.shape[1] < npts:
            coords_buf = np.empty((3, npts), dtype=np.float32)

        coords_buf[0, :npts] = r_idx_v
        coords_buf[1, :npts] = phi_idx_v
        coords_buf[2, :npts] = theta_idx_v

        sampled = map_coordinates(v, coords_buf[:, :npts], order=interp_order, mode="nearest", prefilter=prefilter).astype(np.float32)
        vol[x0 + idx_valid[0], idx_valid[1], idx_valid[2]] = sampled

        elapsed = time.time() - t0
        if verbose:
            remaining = total_work_slabs - processed
            eta = (elapsed / max(processed, 1)) * remaining
            print(f"  Slab {processed:2d}/{total_work_slabs} | Elapsed {fmt_t(elapsed)} | ETA {fmt_t(eta)}")

    return vol


def _make_strictly_increasing(tk: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = tk.astype(np.float32, copy=True)
    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out
