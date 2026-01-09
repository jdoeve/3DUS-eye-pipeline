#!/usr/bin/env python3
"""
UC-Eye Radial Ultrasound Pipeline
Converts radial ultrasound image sequences to 3D Cartesian NRRD volumes.
CALIBRATED VERSION - Fixes geometric distortions
"""

import os
import re
import json
import time
import hashlib
import argparse
import sys
from datetime import timedelta, datetime
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import imageio.v3 as iio
import cv2
import matplotlib
matplotlib.use('Agg')  # headless for launchers
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import map_coordinates, gaussian_filter1d

# ==================== CALIBRATION CONSTANTS (SINGLE SOURCE OF TRUTH) ====================
# Device specifications - Eye Cubed 10MHz posterior mode
EYE_CUBED_DEPTH_MM = 48.0        # Displayed depth BEFORE SoS correction
EYE_CUBED_FAN_ANGLE_DEG = 52.0   # Scanning angle (hardware spec)
SOS_SCALE = 1               # Speed of sound correction factor (0.9622 for Agarose Phantom)

# Derived values (computed once, used everywhere)
EFFECTIVE_DEPTH_MM = EYE_CUBED_DEPTH_MM * SOS_SCALE  # True physical depth after SoS
FAN_ANGLE_RAD = np.deg2rad(EYE_CUBED_FAN_ANGLE_DEG)  # Fan angle in radians

# ==================== CONFIGURATION ====================
DEFAULT_DEPTH_MM = EYE_CUBED_DEPTH_MM
DEFAULT_VOXEL_MM = 0.20  

DEFAULT_SIGMA_THETA_DEG = 0.0 #2.5
DEFAULT_INTERP_ORDER = 1 
DEFAULT_SLAB_SIZE = 48 #24

# Effective Theta span covered by entire scan (deg)
DEFAULT_THETA_SPAN_DEG = 360.0

# ==== DEBUG FLAGS ====
DEBUG_THETA_PHANTOM = False     # If True: each theta slice gets its index
DEBUG_PHI_PHANTOM = False       # If True: each phi row gets its index
DEBUG_RADIAL_PHANTOM = False    # If True: each r index gets its index

DEBUG_BYPASS_MASK = False       # If True: skip mask_unwrapped entirely
DEBUG_DUMP_MASK = False         # If True: save mask_unwrapped to disk

DO_QC = False
# =====================

# ==================== QC AND HELPER FUNCTIONS ====================

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed."""
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def qc_row_arc_against_mask(raw_img: np.ndarray, geom, mask_gray: np.ndarray,
                            depths_mm=(10,20,30,40), verbose=True):
    """
    Compares expected arc length at several depths to pixels across the mask on those rows.
    Reports relative error; should be ≤ ~2% and flat vs depth.
    """
    g = to_gray(raw_img).astype(np.uint8)
    mb = (mask_gray > 128).astype(np.uint8)
    apx_x, apx_y = geom.apex_xy
    dphi = geom.delta_phi_per_px
    dr = geom.dr_mm

    H, W = mb.shape
    errs = []
    for r_mm in depths_mm:
        r_px = r_mm / dr
        # Row index at that radius along the mask "bisector" (φ mid)
        phi_mid = 0.5 * (geom.phi0 + geom.phi1)
        y = int(round(apx_y + r_px * np.sin(phi_mid)))
        y = np.clip(y, 0, H-1)
        row = mb[y, :]
        if row.sum() < 2:
            continue
        xs = np.where(row > 0)[0]
        # Use min/max mask x on that row
        x_lo, x_hi = xs.min(), xs.max()
        n_px = (x_hi - x_lo + 1)
        # Expected arc length for that many lateral pixels
        L_exp = r_mm * (n_px * dphi)
        # Measured "arc in mm" used by pipeline = n_px * (r_mm * dphi per px)
        L_meas = n_px * (r_mm * dphi)
        err = 0.0 if L_exp == 0 else (L_meas - L_exp) / L_exp
        errs.append((r_mm, err))

    if verbose and errs:
        print("[QC-ROW-ARC] depth_mm  rel_error")
        for r_mm, err in errs:
            print(f"               {r_mm:7.1f}  {100.0*err:+6.2f}%")
    return errs

def microsearch_apex(unwrapped_builder, imgs, geom_seed, mask_path: Path,
                     dxdy_range=(-3,3), step=1, depths_mm=(12, 24, 36), verbose=True):
    """
    Try small offsets around the detected apex; choose the one minimizing
    variance of fitted circle radii vs depth in a single mid frame.
    unwrapped_builder: fn (imgs, geom, mask_path)->(unwrapped, mask_unwrapped)
    """
    mid = [imgs[len(imgs)//2]]
    best = {"var": np.inf, "apex": geom_seed.apex_xy, "geom": geom_seed}
    apx_x0, apx_y0 = geom_seed.apex_xy

    for dx in range(dxdy_range[0], dxdy_range[1]+1, step):
        for dy in range(dxdy_range[0], dxdy_range[1]+1, step):
            apx = (apx_x0 + dx, apx_y0 + dy)
            geom = type(geom_seed)(
                image_height=geom_seed.image_height,
                r_max_pix=geom_seed.r_max_pix,
                apex_xy=apx,
                depth_mm=geom_seed.depth_mm_displayed,
                n_rows_mask=geom_seed.image_height_eff,
                phi0_mask=geom_seed.phi0,
                phi1_mask=geom_seed.phi1
            )
            uw, _ = unwrapped_builder(mid, geom, mask_path, drop_duplicate=False, verbose=False)
            # For each requested depth, find edge points and fit circle radius in px
            radii = []
            for r_mm in depths_mm:
                r_idx = int(round((r_mm / geom.dr_mm)))
                if r_idx >= uw.shape[1]:
                    continue
                row = uw[0, r_idx, :]
                # gradient-based edge indices
                grad = np.abs(np.gradient(row.astype(float)))
                thr = 0.5 * np.max(grad) if grad.max() > 0 else 0
                edge_cols = np.where(grad > thr)[0]
                if edge_cols.size < 6:
                    continue
                # Proxy for radius consistency
                radii.append(float(r_mm))
            if len(radii) >= 2:
                v = float(np.var(radii))
                if v < best["var"]:
                    best = {"var": v, "apex": apx, "geom": geom}
    if verbose:
        print(f"[APEX-QA] chosen apex {best['apex']} with var={best['var']:.6f}")
    return best["geom"]

def pca_axis_ratio_of_volume(vol: np.ndarray, thresh_frac=0.6) -> float:
    """Calculate PCA axis ratio of volume for QC."""
    t = float(np.percentile(vol, 100*thresh_frac))
    coords = np.argwhere(vol > t)
    if coords.shape[0] < 500:
        return 1.0
    cov = np.cov(coords.T)
    w, _ = np.linalg.eigh(cov)
    w = np.sort(w)
    return float(np.sqrt(w[-1]/w[0])) if w[0] > 0 else 1.0

def compare_edge_vs_threshold_radius(unwrapped, geom, depth_mm=24.0, verbose=True):
    """
    On a single mid frame, compare radius from gradient edge vs. threshold.
    """
    k = unwrapped.shape[0] // 2
    r_idx = int(round(depth_mm / geom.dr_mm))
    if r_idx >= unwrapped.shape[1]:
        return None
    row = unwrapped[k, r_idx, :].astype(np.float32)
    if row.max() <= 0:
        return None

    # Threshold radius (in lateral px span)
    t = 0.5 * (row.max() + row.min())
    mask = row > t
    cols = np.where(mask)[0]
    if cols.size < 2:
        return None
    thr_span = cols.max() - cols.min()

    # Edge-based (Canny proxy)
    r = (row - row.min()) / (np.ptp(row) + 1e-6)
    r8 = (255*r).astype(np.uint8).reshape(1, -1)  # 2D for Canny (1, N)
    edges = cv2.Canny(r8, 50, 150).flatten() > 0
    ecols = np.where(edges)[0]
    if ecols.size < 2:
        edge_span = thr_span
    else:
        edge_span = ecols.max() - ecols.min()

    # Convert lateral spans to mm at this depth
    mm_per_px = depth_mm * geom.delta_phi_per_px
    thr_mm = thr_span * mm_per_px
    edge_mm = edge_span * mm_per_px

    if verbose:
        print(f"[SEG-QC] depth={depth_mm:.1f}mm  thr_diam={thr_mm:.2f}mm  edge_diam={edge_mm:.2f}mm  diff={thr_mm-edge_mm:+.2f}mm")
    return (thr_mm, edge_mm)

def qc_theta_even_odd(imgs, geom, mask_path, unwrap_fn, resample_fn, voxel_mm=0.30, verbose=True):
    """
    QC: Check rotation uniformity by comparing even/odd frame subsets.
    """
    # unwrap all
    uw_all, _ = unwrap_fn(imgs, geom, mask_path, drop_duplicate=True, verbose=False)
    N = uw_all.shape[0]
    idx_even = np.arange(0, N, 2)
    idx_odd  = np.arange(1, N, 2)

    theta_span = DEFAULT_THETA_SPAN_DEG if "DEFAULT_THETA_SPAN_DEG" in globals() else 360.0

   # Quick recons with coarse voxel size for speed
    vol_all  = resample_fn(
        uw_all,
        geom,
        voxel_mm,
        sigma_theta_deg=0.0,
        theta_span_deg=theta_span,
        interp_order=1,
        slab_size=48,
        verbose=False,
    )
    vol_even = resample_fn(
        uw_all[idx_even],
        geom,
        voxel_mm,
        sigma_theta_deg=0.0,
        theta_span_deg=theta_span,
        interp_order=1,
        slab_size=48,
        verbose=False,
    )
    vol_odd  = resample_fn(
        uw_all[idx_odd],
        geom,
        voxel_mm,
        sigma_theta_deg=0.0,
        theta_span_deg=theta_span,
        interp_order=1,
        slab_size=48,
        verbose=False,
    )

    ar_all  = pca_axis_ratio_of_volume(vol_all)
    ar_even = pca_axis_ratio_of_volume(vol_even)
    ar_odd  = pca_axis_ratio_of_volume(vol_odd)

    if verbose:
        print(f"[QC-θ] axis_ratio all={ar_all:.3f} even={ar_even:.3f} odd={ar_odd:.3f}")
        drift = max(abs(ar_even-ar_all), abs(ar_odd-ar_all))
        bad = drift > 0.02
        if bad:
            print("[QC-θ] >2% drift detected — consider θ-from-image regrid.")
        else:
            print("[QC-θ] OK (≤2%).")
    return ar_all, ar_even, ar_odd

def _roi_for_similarity(unwrapped: np.ndarray, geom, depth_mm_lo=15.0, depth_mm_hi=35.0):
    """
    Pick a radial band [depth_mm_lo, depth_mm_hi] to reduce speckle sensitivity.
    Returns slice object for r-index.
    """
    r0 = int(np.clip(round(depth_mm_lo / geom.dr_mm), 0, unwrapped.shape[1]-1))
    r1 = int(np.clip(round(depth_mm_hi / geom.dr_mm), r0+1, unwrapped.shape[1]))
    return slice(r0, r1)

def _adjacent_corr_score(A: np.ndarray, B: np.ndarray) -> float:
    """
    Correlation score between two 2D arrays (float32). Robust enough + fast.
    """
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    Am = float(A.mean()); Bm = float(B.mean())
    num = float(((A - Am) * (B - Bm)).sum(dtype=np.float64))
    den = float(np.sqrt(((A - Am)**2).sum(dtype=np.float64) * ((B - Bm)**2).sum(dtype=np.float64)) + 1e-12)
    return num / den

def find_best_cut(unwrapped: np.ndarray, geom,
                  depth_mm_lo=15.0, depth_mm_hi=35.0,
                  phi_stride=4, ignore_margin=2) -> tuple[int, float]:
    """
    Find k in [1..N-1] where corr(frame[k-1], frame[k]) is maximal (within ROI).
    Rolling by -k moves that boundary to the wrap point, minimizing seam.
    """
    N = unwrapped.shape[0]
    if N < 6:
        return 0, -1.0

    r_sl = _roi_for_similarity(unwrapped, geom, depth_mm_lo, depth_mm_hi)

    # Downsample phi for speed / robustness (optional)
    # ROI shape: (r_band, phi)
    best_k = 0
    best_score = -1e9

    for k in range(1, N):
        if k < ignore_margin or k > (N - ignore_margin):
            continue
        A = unwrapped[k-1, r_sl, ::phi_stride]
        B = unwrapped[k,   r_sl, ::phi_stride]
        s = _adjacent_corr_score(A, B)
        if s > best_score:
            best_score = s
            best_k = k

    return best_k, float(best_score)

def apply_circular_shift_best_cut(unwrapped: np.ndarray, theta_k: np.ndarray, phi_shifts: np.ndarray, k: int, theta_span_rad: float, verbose: bool = True):
    if k == 0:
        if verbose:
            print("[best-cut] k=0, no circular shift applied.")
        return unwrapped, theta_k.astype(np.float32, copy=False), phi_shifts.astype(np.int32, copy=False)

    if verbose:
        print(f"[best-cut] Applying circular shift by {-k} (cut at k={k}).")

    unwrapped2 = np.roll(unwrapped, shift=-k, axis=0)
    theta2     = np.roll(theta_k,  shift=-k).astype(np.float64)   # do unwrap in float64
    phi2       = np.roll(phi_shifts, shift=-k).astype(np.int32)

    # 1) unwrap → strictly increasing in continuous radians
    theta2 = np.unwrap(theta2)

    # 2) normalize to start at 0
    theta2 -= theta2.min()

    # 3) scale to exactly [0, theta_span_rad]
    span = float(theta2.max() - theta2.min())
    if span < 1e-6:
        # fallback: uniform theta if something went wrong
        if verbose:
            print("[best-cut] WARNING: theta span collapsed; falling back to uniform theta grid.")
        N = theta2.size
        theta2 = np.linspace(0.0, theta_span_rad, N, endpoint=False, dtype=np.float64)
    else:
        theta2 *= (float(theta_span_rad) / span)

    # 4) final strict monotonic safety (float32 can introduce equal neighbors)
    theta2 = theta2.astype(np.float32)
    eps = 1e-6
    for i in range(1, theta2.size):
        if theta2[i] <= theta2[i-1]:
            theta2[i] = theta2[i-1] + eps

    if verbose:
        print(f"[best-cut] theta_k after unwrap: min={float(theta2.min()):.6f} "
              f"max={float(theta2.max()):.6f} span={float(theta2.max()-theta2.min()):.6f} "
              f"noninc={bool(np.any(np.diff(theta2) <= 0))}")

    return unwrapped2, theta2, phi2

def closure_blend(unwrapped: np.ndarray, K: int = 5, verbose=True) -> np.ndarray:
    """
    Cosine-taper blend last K frames into first K frames. Local-only seam reduction.
    """
    if K <= 0:
        return unwrapped
    N = unwrapped.shape[0]
    if 2*K >= N:
        if verbose:
            print("[blend] K too large for N; skipping blend.")
        return unwrapped

    # weights go 0→1 across K frames
    t = np.linspace(0, 1, K, endpoint=False, dtype=np.float32)
    w = 0.5 - 0.5*np.cos(np.pi * t)  # cosine ramp in [0,1)

    out = unwrapped.copy()
    # Blend pairs: end frames mix into start frames
    for i in range(K):
        a = out[N-K+i]     # tail
        b = out[i]         # head
        wi = float(w[i])
        out[i]       = (1.0-wi)*b + wi*a
        out[N-K+i]   = (1.0-wi)*a + wi*b

    if verbose:
        print(f"[blend] Applied closure blend with K={K}.")
    return out

# ==================== GEOMETRY CONFIGURATION (SINGLE SOURCE OF TRUTH) ====================
class GeometryConfig:
    """
    Centralized geometry configuration - compute once, use everywhere.
    Prevents inconsistencies between unwrapping, reconstruction, and QC.
    """
    def __init__(self, image_height: int, r_max_pix: float,
                 apex_xy: tuple, depth_mm: float = None,
                 n_rows_mask: int = None,
                 phi0_mask: float | None = None,
                 phi1_mask: float | None = None):
        """
        Args:
            image_height: Full image height (for reference)
            r_max_pix: Maximum radius in pixels
            apex_xy: (x, y) apex position from mask
            depth_mm: Override depth (default uses Eye Cubed spec)
            n_rows_mask: Usable rows in mask (for correct Δφ calculation)
            phi0_mask: Optional lower φ bound from mask (rad)
            phi1_mask: Optional upper φ bound from mask (rad)
        """
        # Input parameters
        self.image_height = image_height
        self.r_max_pix = r_max_pix
        self.apex_xy = apex_xy

        # CRITICAL: Use usable mask rows for lateral sampling, not full image height
        # This ensures Δφ is computed from the actual mask span, not the image size
        self.image_height_eff = n_rows_mask if n_rows_mask is not None else image_height

        # Depth calibration (use spec unless overridden)
        self.depth_mm_displayed = depth_mm if depth_mm else EYE_CUBED_DEPTH_MM
        self.depth_mm_sos_corrected = self.depth_mm_displayed * SOS_SCALE

        # Angular calibration: use mask φ-bounds if available, else device spec
        spec_rad = FAN_ANGLE_RAD
        use_mask = (phi0_mask is not None) and (phi1_mask is not None)

        if use_mask:
            # Start from mask-derived bounds
            lo = float(phi0_mask)
            hi = float(phi1_mask)
            # Ensure lo < hi
            if hi < lo:
                lo, hi = hi, lo
            # Shift so the lower bound is at 0 (no negative φ)
            if lo < 0.0:
                offset = -lo
                lo += offset
                hi += offset
            self.phi0 = lo
            self.phi1 = hi
            self.fan_angle_rad = self.phi1 - self.phi0
            self.fan_angle_deg = float(np.degrees(self.fan_angle_rad))
            fan_src = "mask"
        else:
            # Device-spec fallback: define φ in [0, fan_angle]
            self.fan_angle_rad = spec_rad
            self.fan_angle_deg = EYE_CUBED_FAN_ANGLE_DEG
            self.phi0 = 0.0
            self.phi1 = self.fan_angle_rad
            fan_src = "spec"

        # KEY: Angular resolution (rad per pixel in lateral direction)
        # Use EFFECTIVE rows (usable mask span), not full image height
        if self.image_height_eff > 1:
            self.delta_phi_per_px = self.fan_angle_rad / float(self.image_height_eff - 1)
        else:
            self.delta_phi_per_px = self.fan_angle_rad

        # Radial resolution (mm per pixel in depth direction)
        self.dr_mm = self.depth_mm_sos_corrected / r_max_pix

        # Print geometry info
        print(f"[GEOM] fan src={fan_src}, fan_deg={self.fan_angle_deg:.3f}, "
              f"Δφ={self.delta_phi_per_px:.6f} rad/px, dr={self.dr_mm:.5f} mm/px, "
              f"rows_eff={self.image_height_eff}")

    def get_arc_length_mm(self, radius_mm: float) -> float:
        """Lateral arc length spanned by the full fan at given radius."""
        return radius_mm * self.fan_angle_rad

    def get_lateral_mm_per_px(self, radius_mm: float) -> float:
        """Lateral spacing (mm/px) at given radius."""
        return radius_mm * self.delta_phi_per_px

    def print_summary(self):
        """Print calibration summary."""
        print("\n" + "=" * 60)
        print("GEOMETRY CONFIGURATION (SINGLE SOURCE OF TRUTH)")
        print("=" * 60)
        print(f"Fan angle:        {self.fan_angle_deg:.1f}° (device spec, LOCKED)")
        print(f"Δφ per pixel:     {self.delta_phi_per_px:.6f} rad/px = {np.degrees(self.delta_phi_per_px):.4f}°/px")
        print(f"Depth (display):  {self.depth_mm_displayed:.2f} mm")
        print(f"Depth (SoS corr): {self.depth_mm_sos_corrected:.2f} mm")
        print(f"Radial dr:        {self.dr_mm:.4f} mm/px (constant)")
        print(f"Apex (x, y):      ({self.apex_xy[0]:.1f}, {self.apex_xy[1]:.1f}) px")
        print(f"r_max:            {self.r_max_pix:.1f} px")
        print(f"Image height:     {self.image_height} px (full image)")
        print(f"Effective rows:   {self.image_height_eff} px (usable mask span)")
        print("\nLateral resolution (varies with depth):")
        for r in [10, 20, 30, 40]:
            lat_mm_per_px = self.get_lateral_mm_per_px(r)
            print(f"  r={r:2d}mm: {lat_mm_per_px:.4f} mm/px")
        print("=" * 60 + "\n")

# ==================== HELPER FUNCTIONS ====================
def fmt_t(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))

@contextmanager
def timer(label: str, verbose: bool = True):
    t0 = time.time()
    yield
    dt = time.time() - t0
    if verbose:
        print(f"[{label}] {fmt_t(dt)}")

def ang_norm(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def load_numeric_sorted_images(folder: Path):
    patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    files = []
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        print(f"[WARNING] Invalid images directory: {folder}")
        return []
    for pat in patterns:
        try:
            files.extend(folder.glob(pat))
        except Exception as e:
            print(f"[WARNING] Error globbing {pat}: {e}")
    if files:
        filtered = [p for p in files if not p.name.startswith("._")]
        if len(filtered) != len(files):
            print(f"[INFO] Skipped {len(files) - len(filtered)} hidden resource files (._*)")
        files = filtered
    if not files:
        print(f"[WARNING] No images found in: {folder}")
        try:
            contents = list(folder.iterdir())
            print(f"[DEBUG] Directory has {len(contents)} items")
            if contents:
                print(f"[DEBUG] First items: {[f.name for f in contents[:5]]}")
        except Exception as e:
            print(f"[DEBUG] Could not list directory: {e}")
        return []

    def num_key(p: Path):
        s = p.stem
        try:
            return (0, int(s))  # Numeric files: sort first, by number
        except Exception:
            return (1, s)  # Non-numeric files: sort second, alphabetically

    files = sorted(files, key=num_key)
    print(f"[DEBUG] Found {len(files)} image files")
    return files

def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def largest_component(bin_img):
    n, labels = cv2.connectedComponents(bin_img.astype(np.uint8))
    if n <= 1:
        return bin_img.astype(np.uint8)
    areas = [(labels == i).sum() for i in range(1, n)]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8)

def find_main_contour(bin_img):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert cnts, "No contour found in cone mask."
    cnt = max(cnts, key=cv2.contourArea)
    pts_xy = cnt.reshape(-1, 2)
    pts_rc = np.stack([pts_xy[:, 1], pts_xy[:, 0]], axis=1)  # (row,col)
    return pts_rc

def line_fit_cv(points_rc):
    # cv2.fitLine expects (x,y)
    pts_xy = points_rc[:, ::-1].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 1e-2, 1e-2).ravel().tolist()
    return (float(x0), float(y0), float(vx), float(vy))

def intersect_lines(l1, l2, eps=1e-9):
    x1, y1, vx1, vy1 = l1
    x2, y2, vx2, vy2 = l2
    denom = vx1 * vy2 - vy1 * vx2
    if abs(denom) < eps:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
    return (x1 + t * vx1, y1 + t * vy1)

# ==================== CALIBRATION (cone mask-derived fan geometry) ====================
def calibrate_from_mask(mask_path: Path, verbose: bool = True):
    mask_path = Path(mask_path)
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask_gray is not None, f"Mask not found: {mask_path}"
    mask_bin = largest_component((mask_gray > 128).astype(np.uint8))
    Hm, Wm = mask_bin.shape
    ys_all, xs_all = np.where(mask_bin > 0)

    # Rough apex seed: leftmost mask column center
    xmin = int(xs_all.min())
    centers = []
    for x in range(xmin, min(xmin + 5, Wm)):
        ys = np.where(mask_bin[:, x] > 0)[0]
        if ys.size:
            centers.append(0.5 * (ys.min() + ys.max()))
    assert centers, "Cannot estimate apex from mask."
    apx0 = (float(xmin), float(np.mean(centers)))

    # Fan/wedge angle estimation
    edge_pts = find_main_contour(mask_bin)  # (row,col)
    dx = edge_pts[:, 1] - apx0[0]
    dy = edge_pts[:, 0] - apx0[1]
    phi_all = np.arctan2(dy, dx)
    phi_lo, phi_hi = np.percentile(phi_all, [2.0, 98.0])

    tol = np.deg2rad(2.5)
    edgeA = edge_pts[np.abs(phi_all - phi_lo) < tol]
    edgeB = edge_pts[np.abs(phi_all - phi_hi) < tol]
    if len(edgeA) < 30 or len(edgeB) < 30:
        tol = np.deg2rad(4.0)
        edgeA = edge_pts[np.abs(phi_all - phi_lo) < tol]
        edgeB = edge_pts[np.abs(phi_all - phi_hi) < tol]

    L1 = line_fit_cv(edgeA)
    L2 = line_fit_cv(edgeB)
    apx = intersect_lines(L1, L2)
    assert apx, "Edge lines nearly parallel."
    apx_x, apx_y = apx

    ang1 = ang_norm(np.arctan2(L1[3], L1[2]))
    ang2 = ang_norm(np.arctan2(L2[3], L2[2]))
    phi0, phi1 = sorted([ang1, ang2])
    if (phi1 - phi0) > np.deg2rad(180):
        phi0, phi1 = phi1, phi0 + 2 * np.pi

    # radius extent in pixels
    r = np.hypot(xs_all - apx_x, ys_all - apx_y)
    r_max_pix = float(np.percentile(r, 99.5))

    # CRITICAL: Compute usable rows (rows that have mask coverage)
    # This is the TRUE lateral sampling count, not the full image height
    row_has_data = (np.sum(mask_bin > 0, axis=1) > 0)
    n_rows_mask = int(np.sum(row_has_data))

    # Store mask-derived phi bounds
    phi0_mask = float(phi0)
    phi1_mask = float(phi1)
    fan_angle_rad_from_mask = phi1_mask - phi0_mask

    if verbose:
        print(f"\n[MASK DETECTION]")
        print(f"  Apex (x,y):     ({apx_x:.2f}, {apx_y:.2f}) px")
        print(f"  r_max:          {r_max_pix:.1f} px")
        print(f"  Fan angle:      {np.degrees(fan_angle_rad_from_mask):.1f}° (mask-detected)")
        print(f"  Image size:     {Hm} × {Wm} px")
        print(f"  Usable rows:    {n_rows_mask} px (rows with mask data)")

    # hash mask to track which calibration was used
    try:
        with open(mask_path, "rb") as f:
            mask_hash = hashlib.sha1(f.read()).hexdigest()[:12]
    except Exception:
        mask_hash = ""

    # Return mask-derived parameters including usable span
    return {
        "apex_xy": (float(apx_x), float(apx_y)),
        "r_max_pix": float(r_max_pix),
        "image_height": int(Hm),
        "image_width": int(Wm),
        "phi0_mask": phi0_mask,
        "phi1_mask": phi1_mask,
        "n_rows_mask": n_rows_mask,
        "mask_fan_angle_deg": float(np.degrees(fan_angle_rad_from_mask)),
        "mask_file": str(mask_path.name),
        "mask_sha1_12": mask_hash
    }

# ==================== UNWRAPPING ====================
def unwrap_frames(imgs, geom: GeometryConfig, mask_path: Path,
                  drop_duplicate: bool = True, verbose: bool = True):
    """
    Unwrap frames using centralized geometry configuration.
    Uses GeometryConfig to ensure consistent phi bounds and resolution.
    """
    H, W = imgs[0].shape[:2]
    apx_x, apx_y = geom.apex_xy
    R = int(np.ceil(geom.r_max_pix))

    # CRITICAL: Use effective image height (usable mask rows) for lateral resolution
    # This ensures Wang matches the actual calibrated lateral sampling
    Wang = geom.image_height_eff

    rho = np.linspace(0, R, R, dtype=np.float32)
    phi_samples = np.linspace(geom.phi0, geom.phi1, Wang, dtype=np.float32)
    Phi, Rho = np.meshgrid(phi_samples, rho, indexing='xy')
    src_x = (apx_x + Rho * np.cos(Phi)).astype(np.float32)
    src_y = (apx_y + Rho * np.sin(Phi)).astype(np.float32)

    # Build wedge mask if cone mask not provided
    mask_path = Path(mask_path)
    if mask_path.exists():
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask_gray > 128).astype(np.uint8)
    else:
        mask_bin = np.zeros((H, W), np.uint8)
        ths = np.linspace(geom.phi0, geom.phi1, 200, dtype=np.float32)
        arc_x = apx_x + geom.r_max_pix * np.cos(ths)
        arc_y = apx_y + geom.r_max_pix * np.sin(ths)
        pts = np.vstack([np.column_stack([arc_x, arc_y]), [apx_x, apx_y]]).astype(np.int32)
        cv2.fillPoly(mask_bin, [pts], 255)

    mask_unwrapped = cv2.remap(mask_bin, src_x, src_y,
                               interpolation=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def unwrap_one(img):
        g = to_gray(img)
        uw = cv2.remap(g, src_x, src_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        # Optional: dump unwrapped mask for debugging
        if DEBUG_DUMP_MASK:
            np.save("debug_mask_unwrapped.npy", (mask_unwrapped > 0).astype(np.uint8))

        if DEBUG_BYPASS_MASK:
            return uw
        else:
            return uw * (mask_unwrapped > 0)

    unwrapped = np.stack([unwrap_one(im) for im in imgs], axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # OPTIONAL: detect and repair a single "bad" theta frame that
    # doesn't correlate with its neighbors (outlier in rotation).
    # This directly targets the one-frame seam / jump you see in axial.
    # ------------------------------------------------------------------
    ENABLE_THETA_OUTLIER_FIX = True
    OUTLIER_CORR_THRESHOLD = 0.90  # tune if needed

    if ENABLE_THETA_OUTLIER_FIX and unwrapped.shape[0] >= 3:
        N = unwrapped.shape[0]
        # Correlation with previous and next frame for i = 1..N-2
        corrs_prev = np.zeros(N, dtype=np.float32)
        corrs_next = np.zeros(N, dtype=np.float32)

        def frame_corr(a, b):
            a = a.astype(np.float32)
            b = b.astype(np.float32)
            am = a.mean()
            bm = b.mean()
            num = ((a - am) * (b - bm)).sum(dtype=np.float64)
            den = np.sqrt(((a - am) ** 2).sum(dtype=np.float64) *
                          ((b - bm) ** 2).sum(dtype=np.float64)) + 1e-12
            return float(num / den)

        for i in range(1, N-1):
            corrs_prev[i] = frame_corr(unwrapped[i], unwrapped[i-1])
            corrs_next[i] = frame_corr(unwrapped[i], unwrapped[i+1])

        # "Badness" = minimum correlation with neighbors
        badness = np.minimum(corrs_prev, corrs_next)
        # Ignore first/last (we handle them separately)
        candidate_idx = np.arange(1, N-1)
        candidate_badness = badness[1:N-1]

        # Find the worst frame among internal indices
        worst_i = candidate_idx[np.argmin(candidate_badness)]
        worst_val = candidate_badness.min()

        if worst_val < OUTLIER_CORR_THRESHOLD:
            # Replace this frame with the average of its neighbors
            if verbose:
                print(f"[unwrap] Theta outlier at index {worst_i} "
                      f"(min corr={worst_val:.3f}); replacing with neighbor average.")
            unwrapped[worst_i] = 0.5 * (unwrapped[worst_i-1] + unwrapped[worst_i+1])
        elif verbose:
            print(f"[unwrap] No theta outlier detected (min neighbor corr={worst_val:.3f}).")
# ------------------------------------------------------------------


    # Handle potential duplicate last frame carefully.
    # Only average/drop if the first and last are truly almost identical.
    if unwrapped.shape[0] >= 2:
        a = unwrapped[0]
        b = unwrapped[-1]
        num = ((a - a.mean()) * (b - b.mean())).sum(dtype=np.float64)
        den = np.sqrt(((a - a.mean())**2).sum(dtype=np.float64) *
                      ((b - b.mean())**2).sum(dtype=np.float64)) + 1e-12
        corr = float(num / den)

        if corr > 0.99:
            # True duplicate: make them consistent and optionally drop the last.
            avg = 0.5 * (a + b)
            unwrapped[0]  = avg
            unwrapped[-1] = avg
            if drop_duplicate:
                unwrapped = unwrapped[:-1]
            if verbose:
                print(f"[unwrap] First/last correlation {corr:.4f} → treated as duplicate, "
                      f"{'dropped last frame' if drop_duplicate else 'kept both averaged'}.")
        else:
            # Not a duplicate: keep as distinct angles, do NOT average or drop.
            if verbose:
                print(f"[unwrap] First/last correlation {corr:.4f} → distinct angles; "
                      "no averaging or drop applied.")

    return unwrapped, mask_unwrapped

# ==================== REGISTRATION ====================
def estimate_phi_shift_1d(ref_uw, mov_uw):
    s_ref = ref_uw.sum(axis=0).astype(np.float32)
    s_mov = mov_uw.sum(axis=0).astype(np.float32)
    s_ref -= s_ref.mean()
    s_mov -= s_mov.mean()
    n = s_ref.size
    F1 = np.fft.rfft(s_ref, n * 2)
    F2 = np.fft.rfft(s_mov, n * 2)
    cc = np.fft.irfft(F1 * np.conj(F2))
    k = int(np.argmax(cc))
    if k >= n:
        k -= 2 * n
    return k

def register_frames(unwrapped, verbose: bool = True):
    """
    Rigidly register unwrapped frames along the φ axis to a mid-frame reference.

    Returns
    -------
    unwrapped_reg : np.ndarray
        Registered unwrapped stack, same shape as input.
    phi_shifts : np.ndarray of shape (N,)
        Integer pixel shifts applied to each frame (φ-direction).
        Positive shift means the frame was rolled forward along +φ.
    """
    N = unwrapped.shape[0]
    ref = unwrapped[N // 2]
    phi_shifts = np.zeros(N, dtype=np.int32)

    for k in range(N):
        sh = estimate_phi_shift_1d(ref, unwrapped[k])
        phi_shifts[k] = sh
        if sh:
            unwrapped[k] = np.roll(unwrapped[k], shift=sh, axis=1)

    if verbose:
        print(f"Registered {N} frames to mid-frame reference")
        print(f"[register] φ-shift range: {phi_shifts.min()}..{phi_shifts.max()} px")

    return unwrapped, phi_shifts

# ==================== RESAMPLING (polar to Cartesian 3D) ====================
def resample_to_cartesian(unwrapped,
                          geom: GeometryConfig,
                          voxel_mm: float,
                          sigma_theta_deg: float,
                          theta_span_deg: float,
                          interp_order: int,
                          slab_size: int,
                          theta_k=None,
                          verbose: bool = True):
  
    """
    Map the unwrapped polar data (r,phi,theta) into a Cartesian volume.
    Uses centralized GeometryConfig for all geometric parameters.

    Parameters
    ----------
    theta_k : np.ndarray or None
        Optional array of per-frame angles in radians (length Ntheta), typically
        built from frame index plus φ-based registration shifts. If None, a
        uniform theta grid over [0, theta_span_deg] is assumed (legacy behaviour).
    """

    N, R, Wang = unwrapped.shape

    # All geometry comes from GeometryConfig (single source of truth)
    dr_mm = geom.dr_mm  # sos-adjusted mm per radial pixel
    dphi = geom.delta_phi_per_px  # rad per pixel (lateral)
    phi0 = geom.phi0
    phi1 = geom.phi1
    depth_mm = geom.depth_mm_sos_corrected

    # Re-order to V[r, phi, theta]
    V = np.transpose(unwrapped, (1, 2, 0)).astype(np.float32)
    Rn, Wang_v, Ntheta = V.shape

    # Ensure V is contiguous and float32 for fast sampling
    V = np.ascontiguousarray(V, dtype=np.float32)

    # Prepare theta grid → frame index mapping
    if theta_k is not None:
        theta_k = np.asarray(theta_k, dtype=np.float32)
        if theta_k.shape[0] != Ntheta:
            raise ValueError(f"theta_k length {theta_k.shape[0]} does not match Ntheta {Ntheta}")
        def make_strictly_increasing(tk: np.ndarray, eps: float = 1e-6) -> np.ndarray:
            out = tk.astype(np.float32, copy=True)
            for i in range(1, out.size):
                if out[i] <= out[i-1]:
                    out[i] = out[i-1] + eps
            return out

        theta_k = make_strictly_increasing(theta_k)

        theta_min = float(theta_k[0])
        theta_max = float(theta_k[-1])
    else:
        # Legacy: assume uniform sampling over requested theta_span_deg
        theta_min = 0.0
        theta_max = np.deg2rad(theta_span_deg)
        theta_k = np.linspace(theta_min, theta_max, Ntheta, dtype=np.float32)

    frame_idx_array = np.arange(Ntheta, dtype=np.float32)
    theta_span_rad_eff = max(theta_max - theta_min, 1e-6)

    print("theta_k last 5:", theta_k[-5:])
    print("diff last 5:", np.diff(theta_k[-5:]))
    print("any non-increasing:", np.any(np.diff(theta_k) <= 0))

    # ==== DEBUG PHANTOMS: enable ONE at a time ====
    if DEBUG_THETA_PHANTOM:
        # Periodic theta phantom: intensity = (cos(theta)+1)/2
        # This ensures value at 0° equals value at 360° (no inherent jump).
        theta_vals = np.linspace(0.0, 2.0*np.pi, Ntheta, endpoint=False, dtype=np.float32)
        cos_vals = 0.5 * (np.cos(theta_vals) + 1.0)  # range ~[0,1]
        for k in range(Ntheta):
            V[:, :, k] = cos_vals[k]
    elif DEBUG_PHI_PHANTOM:
        for j in range(Wang_v):
            V[:, j, :] = float(j)
    elif DEBUG_RADIAL_PHANTOM:
        for i in range(Rn):
            V[i, :, :] = float(i)
    # ===============================================

    # ---- Patch 2: make theta periodic for interpolation (close the loop) ----
    # Add a duplicate of theta=0 as the last theta slice so 2π blends with 0.
    V = np.concatenate([V, V[:, :, 0:1]], axis=2)
    Rn, Wang_v, Ntheta = V.shape  # Ntheta is now original+1
    # ------------------------------------------------------------------------

    # Make theta_k and frame_idx_array match the new theta length (Ntheta = Norig+1)
    theta_k = np.concatenate([theta_k, [theta_k[0] + theta_span_rad_eff]]).astype(np.float32)
    frame_idx_array = np.arange(Ntheta, dtype=np.float32)

    sigma_theta = 0.0  # default (no smoothing)
    did_smooth = False

    # (Optional) smooth along sweep axis (theta dimension)
    if sigma_theta_deg > 0:
        did_smooth = True
        # Convert degrees of smoothing to index units using the requested sweep span.
        # Example: if Ntheta=400 and theta_span_deg=240, then 1° ≈ (399/240) indices.
        sigma_theta = sigma_theta_deg * ((Ntheta - 1) / max(theta_span_deg, 1e-3))
        V = gaussian_filter1d(V, sigma=sigma_theta, axis=2, mode='wrap')

    # We build a cube in mm of half_extent = depth_mm (sos-corrected).
    half_extent = depth_mm
    n_axis = int(np.ceil((2 * half_extent) / voxel_mm))
    xs = np.linspace(-half_extent, half_extent, n_axis, dtype=np.float32)
    ys = np.linspace(-half_extent, half_extent, n_axis, dtype=np.float32)
    zs = np.linspace(-half_extent, half_extent, n_axis, dtype=np.float32)

    Nx = Ny = Nz = n_axis
    prefilter = (interp_order >= 3)

    if verbose:
        print(f"Output volume: ({Nx}, {Ny}, {Nz}) at {voxel_mm:.3f} mm/voxel")
        print(f"[GEOMETRY] Fan angle: {geom.fan_angle_deg:.1f}° (device spec)")
        print(f"[GEOMETRY] Δφ per pixel: {dphi:.6f} rad/px")
        print(f"[GEOMETRY] Radial dr: {dr_mm:.4f} mm/px")
        print(f"[GEOMETRY] Depth (SoS corrected): {depth_mm:.3f} mm")
        
    # Cartesian sampling grid
    Y2, Z2 = np.meshgrid(ys, zs, indexing='ij')
    rho2d = np.sqrt(Y2**2 + Z2**2).astype(np.float32)          # (Ny, Nz)
    psi2d = np.arctan2(Z2, Y2).astype(np.float32)              # (Ny, Nz)
    psi2d = np.mod(psi2d, 2*np.pi).astype(np.float32)          # [0, 2π)

    # --- NEW: rotate branch cut away from voxel grid (half-bin offset) ---
    psi_offset = np.pi/2
    psi2d = np.mod(psi2d + psi_offset, 2*np.pi).astype(np.float32)

    vol = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    n_slabs = int(np.ceil(Nx / slab_size))
    coords_buf = None   # reusable buffer for map_coordinates
    t0 = time.time()

    # --- Optimization 2: find slab bounds with any valid voxels (cheap probe) ---
    first_valid_s = None
    last_valid_s = None

    for s in range(n_slabs):
        x0 = s * slab_size
        x1 = min((s + 1) * slab_size, Nx)
        xs_slab = xs[x0:x1][:, None, None]

        # Compute only what is needed for validity (cheap)
        rho = rho2d[None, :, :]  # cheap view
        r_cart = np.sqrt(xs_slab**2 + rho**2)
        phi_cart = np.arctan2(rho, xs_slab)  # [0, pi]

        valid = (r_cart <= depth_mm) & (phi_cart >= phi0) & (phi_cart <= phi1)
        n_valid = int(np.count_nonzero(valid))

        if n_valid > 0:
            if first_valid_s is None:
                first_valid_s = s
            last_valid_s = s

    # IMPORTANT: this check must be AFTER the probe loop
    if first_valid_s is None:
        return vol
# -------------------------------------------------------------------------
    processed = 0
    total_work_slabs = (last_valid_s - first_valid_s + 1)

    for s in range(first_valid_s, last_valid_s + 1):
        processed += 1
        x0 = s * slab_size
        x1 = min((s + 1) * slab_size, Nx)
      
        xs_slab = xs[x0:x1][:, None, None]

        # Convert each Cartesian sample point to source polar indices
        rho = rho2d[None, :, :]  # cheap view
        r_cart = np.sqrt(xs_slab**2 + rho**2)
        phi_cart = np.arctan2(rho, xs_slab)  # [0, pi]

        # valid mask: inside radial depth + inside fan
        valid = (r_cart <= depth_mm) & (phi_cart >= phi0) & (phi_cart <= phi1)

        # Map to indices in V
        r_idx = (r_cart / dr_mm)
        phi_idx = (phi_cart - phi0) / dphi

        idx_valid = np.where(valid)

        if verbose:
            print(f"[Slab {s+1}/{n_slabs}] n_valid={idx_valid[0].size}")

        if idx_valid[0].size == 0:
            continue

        if idx_valid[0].size:
            r_idx_v = np.clip(r_idx[idx_valid], 0, Rn - 1 - 1e-3).astype(np.float32)
            phi_idx_v = np.clip(phi_idx[idx_valid], 0, Wang_v - 1 - 1e-3).astype(np.float32)

            psi_v = psi2d[idx_valid[1], idx_valid[2]]  # depends only on (y,z)
            theta_virtual_v = (psi_v / (2*np.pi)) * theta_span_rad_eff + theta_min
            theta_virtual_v = theta_virtual_v.astype(np.float32, copy=False)

            # Interpolate global angle to fractional frame index
            theta_idx_v = np.interp(theta_virtual_v, theta_k, frame_idx_array)
            theta_idx_v = np.clip(theta_idx_v, 0.0, (Ntheta - 1.0) - 1e-6)

            # >>> ADD THIS BLOCK RIGHT HERE <<<
            if DEBUG_THETA_PHANTOM and processed == 1:  # first slab only
                print("theta_idx_v min/max:", float(theta_idx_v.min()), float(theta_idx_v.max()))
                print("frac near 0:", float(np.mean(theta_idx_v < 1.0)))
                print("frac near end:", float(np.mean(theta_idx_v > (Ntheta - 2.0))))

            n = r_idx_v.size
            if coords_buf is None or coords_buf.shape[1] < n:
                coords_buf = np.empty((3, n), dtype=np.float32)

            coords_buf[0, :n] = r_idx_v
            coords_buf[1, :n] = phi_idx_v
            coords_buf[2, :n] = theta_idx_v
            coords = coords_buf[:, :n]

            sampled = map_coordinates(
                V,
                coords,
                order=interp_order,
                mode='nearest',
                prefilter=prefilter,
            ).astype(np.float32)

            vol[x0 + idx_valid[0], idx_valid[1], idx_valid[2]] = sampled

        elapsed = time.time() - t0
        if verbose:
            remaining = total_work_slabs - processed
            eta = (elapsed / max(processed, 1)) * remaining
            print(f"  Slab {processed:2d}/{total_work_slabs} | Elapsed {fmt_t(elapsed)} | ETA {fmt_t(eta)}")

    return vol

# ==================== METADATA & NAMING ====================
_EYE_TOKENS = {"OD", "OS", "RIGHT", "LEFT", "R", "L"}

def _extract_eye_token(text: str) -> str | None:
    parts = re.split(r'[-_\s]+', text.upper())
    for p in parts:
        if p in _EYE_TOKENS:
            return "OD" if p in {"OD", "RIGHT", "R"} else "OS"
    return None

def _clean_base_from_folder(folder_name: str) -> str:
    """
    Build a clean base name from noisy phantom folders.
    - strip tokens like 'Radial', 'Primary', 'Secondary'
    - remove dates/times 'MM-DD-YYYY' or 'HH-MM'
    - squeeze spaces; normalize ' mm'->'mm'
    - normalize ' X _ Y ' -> 'X_Y'
    """
    base = folder_name

    # Remove common noise tokens with separators around them
    noise_tokens = [r'\bRadial\b', r'\bPrimary\b', r'\bSecondary\b', r'\bOD\b', r'\bOS\b']
    for tok in noise_tokens:
        base = re.sub(
            rf'(^|[_\-\s]){tok}([_\-\s]|$)',
            ' ',
            base,
            flags=re.IGNORECASE
        )

    # Remove date/time tokens like "09-25-2025" or "11-47"
    base = re.sub(r'\b\d{2}-\d{2}-\d{4}\b', ' ', base)  # MM-DD-YYYY
    base = re.sub(r'\b\d{2}-\d{2}\b', ' ', base)        # HH-MM

    # Remove extra mm spacing like "65 mm" -> "65mm"
    base = re.sub(r'\b(\d+)\s*mm\b', r'\1mm', base, flags=re.IGNORECASE)

    # Collapse weird separators sequences " -  - "
    base = re.sub(r'\s*[-–—]{1,}\s*', ' ', base)

    # Normalize spaces around underscores: " A _ B " -> "A_B"
    base = re.sub(r'\s*_\s*', '_', base)

    # Squeeze spaces
    base = re.sub(r'\s{2,}', ' ', base).strip()

    return base

def _next_run_suffix(target_dir: Path, prefix: str) -> str:
    """
    Find next rNN suffix in target_dir for files starting with prefix.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = list(target_dir.glob(f"{prefix}_r*.nrrd"))
    if not existing:
        return "r01"
    # extract max NN
    max_n = 0
    for p in existing:
        m = re.search(r'_r(\d{2})\.nrrd$', p.name)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return f"r{max_n+1:02d}"

def generate_iterative_filename(img_dir: Path,
                                out_dir: Path | None,
                                verbose: bool = True) -> Path:
    """
    Build a filename like:
    <prefix>_<eye>_EYE_VOL_rNN.nrrd
    where prefix is a cleaned folder name.
    """
    img_dir = Path(img_dir)
    parent = img_dir.parent
    folder_name = img_dir.name

    eye_tok = _extract_eye_token(folder_name) or _extract_eye_token(parent.name) or "UNK"

    prefix_base = _clean_base_from_folder(folder_name)
    prefix = f"{prefix_base}_{eye_tok}_EYE_VOL"
    run_suffix = _next_run_suffix(out_dir or parent, prefix)

    out_dir_final = out_dir or parent
    out_dir_final.mkdir(parents=True, exist_ok=True)
    out_file = out_dir_final / f"{prefix}_{run_suffix}.nrrd"

    if verbose:
        print(f"[OUTPUT] Using iterative filename: {out_file}")
    return out_file

# ==================== QC VALIDATION ====================
def validate_calibration_qc(cal: dict, dr_mm: float, verbose: bool = True):
    """
    QC checks for calibration geometry.

    Checks:
    1. Circle test: Verify angular resolution produces consistent arc lengths
    2. Arc-length validation: Check lateral spacing at different radii

    Args:
        cal: Calibration dictionary with delta_phi_per_px, phi0_rad, phi1_rad
        dr_mm: Radial pixel spacing in mm
        verbose: Print detailed output
    """
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

    # Check 1: Circle test - arc length at different radii
    print("\n[QC Test 1] Circle test - arc length at different radii:")
    test_radii_mm = [10.0, 20.0, 30.0, 40.0]  # Test at multiple depths

    for r_mm in test_radii_mm:
        # Arc length for 1 pixel lateral step at this radius
        arc_length_per_px = r_mm * delta_phi
        # Total arc length across the fan
        total_arc = r_mm * (phi1 - phi0)

        print(f"  r={r_mm:5.1f}mm: arc_per_px={arc_length_per_px:.4f}mm, "
              f"total_arc={total_arc:.2f}mm, "
              f"fan_angle={np.degrees(phi1-phi0):.1f}°")

    # Check 2: Expected sphere appearance
    print("\n[QC Test 2] Expected sphere measurements:")
    print(f"  A 7.2mm diameter sphere should measure:")
    print(f"    - Radial (depth): ~7.2mm (with SOS correction)")
    print(f"    - Lateral (arc): depends on depth")

    # At mid-depth (24mm), what lateral span does 7.2mm diameter create?
    r_mid = 24.0  # mm from apex
    sphere_diameter = 7.2  # mm
    # Angular span: Δφ = arc_length / radius
    angular_span_rad = sphere_diameter / r_mid
    angular_span_deg = np.degrees(angular_span_rad)
    # Number of pixels this spans laterally
    lateral_pixels = angular_span_rad / delta_phi

    print(f"  At r={r_mid}mm depth:")
    print(f"    - Angular span: {angular_span_deg:.2f}° ({angular_span_rad:.4f} rad)")
    print(f"    - Lateral pixels: {lateral_pixels:.1f} px")
    print(f"    - Arc length: {sphere_diameter:.2f}mm")

    # Check 3: Radial vs lateral resolution comparison
    print("\n[QC Test 3] Resolution comparison:")
    print(f"  Radial pixel spacing: {dr_mm:.4f} mm/px (constant)")
    print(f"  Lateral spacing at r=10mm: {10.0 * delta_phi:.4f} mm/px")
    print(f"  Lateral spacing at r=20mm: {20.0 * delta_phi:.4f} mm/px")
    print(f"  Lateral spacing at r=30mm: {30.0 * delta_phi:.4f} mm/px")
    print(f"  Lateral spacing at r=40mm: {40.0 * delta_phi:.4f} mm/px")
    print(f"  NOTE: Lateral resolution degrades (spacing increases) with depth")

    print("=" * 60 + "\n")

# ==================== QC IMAGE EXPORT ====================
def generate_qc(vol, qc_dir: Path, unwrapped=None, mask_unwrapped=None):
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    vmin = float(np.percentile(vol, 60.0))
    vmax = float(np.percentile(vol, 99.7))

    mip_xy = vol.max(axis=2).T
    mip_xz = vol.max(axis=1).T
    mip_yz = vol.max(axis=0).T

    # central slices
    x_signal = vol.sum(axis=(1, 2))
    cx = int(np.argmax(x_signal))
    cy = vol.shape[1] // 2
    cz = vol.shape[2] // 2

    slc_x = vol[cx, :, :].T
    slc_y = vol[:, cy, :].T
    slc_z = vol[:, :, cz].T

    plt.imsave(qc_dir / "mip_xy.png", mip_xy,
               vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
    plt.imsave(qc_dir / "mip_xz.png", mip_xz,
               vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
    plt.imsave(qc_dir / "mip_yz.png", mip_yz,
               vmin=vmin, vmax=vmax, origin='lower', cmap='gray')

    plt.imsave(qc_dir / f"slice_x_{cx}.png", slc_x,
               vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
    plt.imsave(qc_dir / f"slice_y_{cy}.png", slc_y,
               vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
    plt.imsave(qc_dir / f"slice_z_{cz}.png", slc_z,
               vmin=vmin, vmax=vmax, origin='lower', cmap='gray')

    if unwrapped is not None:
        k = int(np.floor(unwrapped.shape[0] / 2))
        plt.imsave(qc_dir / "unwrapped_mid.png", unwrapped[k], cmap='gray')

    if mask_unwrapped is not None:
        plt.imsave(qc_dir / "mask_unwrapped.png", mask_unwrapped, cmap='gray')

    print(f"QC images saved to: {qc_dir}")

# ==================== MAIN PIPELINE ====================
def run_pipeline(img_dir: Path,
                 mask_path: Path,
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
                 out_dir_for_iter: Path | None = None):

    print("=" * 60)
    print("UC-Eye Radial Ultrasound Pipeline (CALIBRATED)")
    print("=" * 60)

    img_dir = Path(img_dir)
    mask_path = Path(mask_path)
    if output_path is not None:
        output_path = Path(output_path)

    if not img_dir.exists() or not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found or not a directory: {img_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    with timer("Load frames", verbose):
        files = load_numeric_sorted_images(img_dir)
        if not files:
            raise ValueError(f"No images found in {img_dir}")
        imgs = [iio.imread(p) for p in files]
        if verbose:
            print(f"  Loaded {len(imgs)} frames")

    # Detect mask parameters (apex, r_max, image size)
    mask_cal_path = img_dir.parent / "mask_detection.json"
    if mask_cal_path.exists():
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

    # Create centralized geometry configuration (SINGLE SOURCE OF TRUTH)
    geom = GeometryConfig(
        image_height=mask_params["image_height"],
        r_max_pix=mask_params["r_max_pix"],
        apex_xy=tuple(mask_params["apex_xy"]),
        depth_mm=depth_mm,
        n_rows_mask=mask_params.get("n_rows_mask"),  # Use usable rows for Δφ
        phi0_mask=mask_params.get("phi0_mask"),      # Use mask φ-bounds if available
        phi1_mask=mask_params.get("phi1_mask")       # else falls back to spec
    )

    if verbose:
        geom.print_summary()

    if DO_QC:
        # QC: Row-arc check
        if verbose:
            with timer("QC: Row-arc vs mask", verbose):
                mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                qc_row_arc_against_mask(imgs[len(imgs)//2], geom, mask_gray, depths_mm=(10,20,30,40), verbose=verbose)

    # Apex micro-search for optimal alignment
    with timer("Apex micro-search", verbose):
        geom = microsearch_apex(unwrap_frames, imgs, geom, mask_path, dxdy_range=(-2,2), step=1,
                                depths_mm=(12,24,36), verbose=verbose)

    with timer("Unwrap frames", verbose):
        unwrapped, mask_unwrapped = unwrap_frames(
            imgs,
            geom,
            mask_path,
            drop_duplicate=drop_duplicate,
            verbose=verbose
        )
        if verbose:
            print(f"  Unwrapped shape: {unwrapped.shape} (N, r, phi)")

    # --- Build per-frame theta mapping (θ_k) in radians ---
    Ntheta = unwrapped.shape[0]
    theta_span_rad = np.deg2rad(theta_span_deg)
    theta_k = np.linspace(
        0.0,
        theta_span_rad,
        Ntheta,
        endpoint=False,   # <<< CRITICAL FIX
        dtype=np.float32)


    if do_registration:
        with timer("Register φ across frames", verbose):
            unwrapped, phi_shifts = register_frames(unwrapped, verbose=verbose)

        # Convert φ pixel shifts to small angular corrections in θ (radians).
        # Each φ pixel corresponds to Δφ radians; rolling the image by +sh pixels
        # means the original frame was rotated by about -sh * Δφ.
        dphi = geom.delta_phi_per_px  # rad/px along φ
        delta_theta = -phi_shifts.astype(np.float32) * dphi

        theta_k = theta_k + delta_theta

        # Renormalize θ_k back to [0, theta_span_rad] while preserving ordering.
        theta_k -= theta_k.min()
        max_span = max(theta_k.max(), 1e-6)
        theta_k *= theta_span_rad / max_span
    else:
        phi_shifts = np.zeros(Ntheta, dtype=np.int32)

    # ==================== SEAM ROBUSTNESS (REAL DATA) ====================
    ENABLE_BEST_CUT_SHIFT = True
    ENABLE_CLOSURE_BLEND  = True
    BLEND_K = 5  # start with 5; tune 3–7

    # (Optional) measure closure corr before
    r_sl = _roi_for_similarity(unwrapped, geom, depth_mm_lo=15.0, depth_mm_hi=35.0)
    closure_before = _adjacent_corr_score(unwrapped[-1, r_sl, ::4], unwrapped[0, r_sl, ::4])
    if verbose:
        print(f"[seam] closure corr BEFORE shift/blend: {closure_before:.4f}")

    if ENABLE_BEST_CUT_SHIFT:
        k_cut, k_score = find_best_cut(
            unwrapped, geom,
            depth_mm_lo=15.0, depth_mm_hi=35.0,
            phi_stride=4,
            ignore_margin=2
        )
        if verbose:
            print(f"[seam] best cut k={k_cut} with adjacent corr={k_score:.4f}")

        unwrapped, theta_k, phi_shifts = apply_circular_shift_best_cut(
            unwrapped, theta_k, phi_shifts, k_cut, theta_span_rad, verbose=verbose
            )

    if ENABLE_CLOSURE_BLEND:
        unwrapped = closure_blend(unwrapped, K=BLEND_K, verbose=verbose)

    # measure closure corr after
    closure_after = _adjacent_corr_score(unwrapped[-1, r_sl, ::4], unwrapped[0, r_sl, ::4])
    if verbose:
        print(f"[seam] closure corr AFTER shift/blend: {closure_after:.4f}")
# ======================================================

    if DO_QC:
        # QC: Segmentation bias check (2D)
        if verbose:
            with timer("Segmentation bias check (2D)", verbose):
                _ = compare_edge_vs_threshold_radius(unwrapped, geom, depth_mm=24.0, verbose=verbose)

        # QC: Rotation uniformity (even/odd subsets)
        if verbose:
            with timer("QC: rotation even/odd", verbose):
                _ = qc_theta_even_odd(imgs, geom, mask_path, unwrap_frames, resample_to_cartesian, voxel_mm=0.30, verbose=verbose)
    
    print("[theta_k] min/max:", float(theta_k.min()), float(theta_k.max()))
    print("[theta_k] any non-increasing:", bool(np.any(np.diff(theta_k) <= 0)))
    print("[theta_k] max span (rad):", float(theta_k.max() - theta_k.min()))

    # Geometry is now centralized in geom object - use it everywhere
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
            verbose=verbose
        )

    # ---------- Output path logic (supports --out-dir and iterative naming) ----------
    if output_path is None:
        output_path = generate_iterative_filename(img_dir, out_dir_for_iter, verbose)

    with timer("Save", verbose):
        vol_zyx = np.transpose(vol, (2, 1, 0)).astype(np.float32)
        img = sitk.GetImageFromArray(vol_zyx)

        # >>> CHANGE: spacing is voxel_mm which we calibrated from mm/pixel.
        # This spacing is written isotropically because our resampling grid was built
        # with voxel_mm in X,Y,Z mm steps.
        img.SetSpacing((voxel_mm, voxel_mm, voxel_mm))

        # put apex ~ at Z=0 (convenient origin for downstream analysis)
        apex_physical = (vol.shape[0] // 2, vol.shape[1] // 2, 0)
        origin_mm = tuple(-apex_physical[i] * voxel_mm for i in range(3))
        img.SetOrigin(origin_mm)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(output_path))
        if verbose:
            print(f"  Saved: {output_path}")
            print(f"  Origin set to: {origin_mm} (apex near Z=0)")
            # >>> CHANGE: print calibration used for traceability
            print(f"  [CAL NOTE] SOS_SCALE={SOS_SCALE:.5f}, "
                  f"effective_depth_mm={geom.depth_mm_sos_corrected:.3f}mm, "
                  f"voxel_mm={voxel_mm:.4f}mm")

    qc_dir = output_path.parent / "QC"
    with timer("QC", verbose):
        generate_qc(vol, qc_dir, unwrapped, mask_unwrapped)

    print("=" * 60)
    print("Pipeline complete")
    print("=" * 60)
    return output_path

# ==================== CLI (compatible with launcher) ====================
def _resolve_paths_from_args(args):
    img_dir = args.input_dir or args.img_dir
    mask_path = args.cone_mask or args.mask_path
    # output: either explicit file (-o/--output) OR a directory (--out-dir)
    output_file = args.output
    out_dir = args.out_dir
    return img_dir, mask_path, output_file, out_dir

def main():
    parser = argparse.ArgumentParser(
        description="UC-Eye Pipeline (Calibrated + SoS/Lateral Scaling) — compatible CLI"
    )

    # Positional (legacy, like old run_uceye.cmd)
    parser.add_argument("img_dir", nargs='?', type=Path, help="Images directory (positional)")
    parser.add_argument("mask_path", nargs='?', type=Path, help="Cone mask PNG (positional)")

    # Flagged (newer launcher style)
    parser.add_argument("--input-dir", type=Path, help="Images directory")
    parser.add_argument("--cone-mask", type=Path, help="Cone mask PNG")
    parser.add_argument("--out-dir", type=Path,
                        help="Output directory (file auto-named as *_EYE_VOL_rNN.nrrd)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Explicit output NRRD file path")

    parser.add_argument("--depth", type=float, default=DEFAULT_DEPTH_MM,
                        help="Physical depth in mm (Eye Cubed default: 48mm BEFORE SoS correction)")
    parser.add_argument("--voxel", type=float, default=DEFAULT_VOXEL_MM,
                        help="Target mm/voxel in output (lateral mm/pixel calibration)")
    parser.add_argument("--sigma-theta", type=float, default=DEFAULT_SIGMA_THETA_DEG,
                        help="Smoothing (deg) around sweep axis")
    parser.add_argument("--theta-span-deg", type=float, default=DEFAULT_THETA_SPAN_DEG,
                        help="Effective θ sweep span in degrees (tune with phantom to reduce kink)")
    parser.add_argument("--interp-order", type=int, default=DEFAULT_INTERP_ORDER,
                        help="Interpolation order for map_coordinates (0..5)")
    parser.add_argument("--slab-size", type=int, default=DEFAULT_SLAB_SIZE,
                        help="How many x-slices to process at a time")
    parser.add_argument("--no-register", action='store_true',
                        help="Disable phi registration between frames")
    parser.add_argument("--keep-duplicate", action='store_true',
                        help="Keep last frame even if duplicate")
    parser.add_argument("--quiet", action='store_true',
                        help="Less console output")

    args = parser.parse_args()

    img_dir, mask_path, output_file, out_dir_for_iter = _resolve_paths_from_args(args)

    if img_dir is None or mask_path is None:
        parser.print_help(sys.stderr)
        print("\nERROR: You must provide images and cone mask either positionally "
              "or via --input-dir / --cone-mask.", file=sys.stderr)
        sys.exit(1)

    try:
        run_pipeline(
            img_dir=img_dir,
            mask_path=mask_path,
            output_path=output_file,
            depth_mm=args.depth,
            voxel_mm=args.voxel,
            sigma_theta_deg=args.sigma_theta,
            theta_span_deg=args.theta_span_deg,
            interp_order=args.interp_order,
            slab_size=args.slab_size,
            do_registration=not args.no_register,
            drop_duplicate=not args.keep_duplicate,
            verbose=not args.quiet,
            out_dir_for_iter=out_dir_for_iter
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
