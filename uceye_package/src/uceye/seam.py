"""Seam detection, circular shift, closure, and blending."""

import numpy as np
from scipy.ndimage import shift as ndi_shift


def roi_for_similarity(unwrapped: np.ndarray, geom, depth_mm_lo=15.0, depth_mm_hi=35.0):
    r0 = int(np.clip(round(depth_mm_lo / geom.dr_mm), 0, unwrapped.shape[1] - 1))
    r1 = int(np.clip(round(depth_mm_hi / geom.dr_mm), r0 + 1, unwrapped.shape[1]))
    return slice(r0, r1)


def adjacent_corr_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    am = float(a.mean())
    bm = float(b.mean())
    num = float(((a - am) * (b - bm)).sum(dtype=np.float64))
    den = float(np.sqrt(((a - am) ** 2).sum(dtype=np.float64) * ((b - bm) ** 2).sum(dtype=np.float64)) + 1e-12)
    return num / den


def find_best_cut(unwrapped: np.ndarray, geom, depth_mm_lo=15.0, depth_mm_hi=35.0, phi_stride=4, ignore_margin=2) -> tuple[int, float]:
    n = unwrapped.shape[0]
    if n < 6:
        return 0, -1.0

    r_sl = roi_for_similarity(unwrapped, geom, depth_mm_lo, depth_mm_hi)
    best_k = 0
    best_score = -1e9

    for k in range(1, n):
        if k < ignore_margin or k > (n - ignore_margin):
            continue
        a = unwrapped[k - 1, r_sl, ::phi_stride]
        b = unwrapped[k, r_sl, ::phi_stride]
        s = adjacent_corr_score(a, b)
        if s > best_score:
            best_score = s
            best_k = k

    return best_k, float(best_score)


def enforce_strictly_increasing(tk: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    tk = tk.astype(np.float32, copy=True)
    for i in range(1, tk.size):
        if tk[i] <= tk[i - 1]:
            tk[i] = tk[i - 1] + eps
    return tk


def apply_circular_shift_best_cut(
    unwrapped: np.ndarray,
    theta_k: np.ndarray,
    phi_shifts: np.ndarray,
    drifts_r: np.ndarray,
    drifts_phi: np.ndarray,
    k: int,
    theta_span_rad: float,
    verbose: bool = True,
):
    if k == 0:
        if verbose:
            print("[best-cut] k=0, no circular shift applied.")
        return (
            unwrapped,
            theta_k.astype(np.float32, copy=False),
            phi_shifts.astype(np.int32, copy=False),
            drifts_r.astype(np.float32, copy=False),
            drifts_phi.astype(np.float32, copy=False),
        )

    if verbose:
        print(f"[best-cut] Applying circular shift by {-k} (cut at k={k}).")

    unwrapped2 = np.roll(unwrapped, shift=-k, axis=0)
    theta2 = np.roll(theta_k, shift=-k).astype(np.float64)
    phi2 = np.roll(phi_shifts, shift=-k).astype(np.int32)
    drifts_r2 = np.roll(drifts_r, shift=-k).astype(np.float32)
    drifts_phi2 = np.roll(drifts_phi, shift=-k).astype(np.float32)

    theta2 = np.unwrap(theta2)
    theta2 -= float(theta2.min())

    span = float(theta2.max() - theta2.min())
    if span < 1e-6:
        if verbose:
            print("[best-cut] WARNING: theta span collapsed; falling back to uniform grid.")
        n = theta2.size
        theta2 = np.linspace(0.0, theta_span_rad, n, endpoint=False, dtype=np.float64)
    else:
        theta2 *= float(theta_span_rad) / span

    theta2 = enforce_strictly_increasing(theta2.astype(np.float32))

    if verbose:
        print(
            f"[best-cut] theta_k after fix: min={theta2.min():.6f} "
            f"max={theta2.max():.6f} span={theta2.max() - theta2.min():.6f} "
            f"noninc={np.any(np.diff(theta2) <= 0)}"
        )

    return unwrapped2, theta2, phi2, drifts_r2, drifts_phi2


def enforce_closure_drift(unwrapped, drifts_r, drifts_phi, verbose=True):
    ntheta = unwrapped.shape[0]
    t = np.linspace(0.0, 1.0, ntheta, dtype=np.float32)

    drifts_phi_corr = drifts_phi - (drifts_phi[0] + (drifts_phi[-1] - drifts_phi[0]) * t)
    drifts_r_corr = drifts_r - (drifts_r[0] + (drifts_r[-1] - drifts_r[0]) * t)

    extra_phi = (drifts_phi_corr - drifts_phi).astype(np.float32)
    extra_r = (drifts_r_corr - drifts_r).astype(np.float32)

    for k in range(ntheta):
        phi_int = int(np.round(extra_phi[k]))
        tmp = np.roll(unwrapped[k], shift=phi_int, axis=1)

        phi_frac = float(extra_phi[k] - phi_int)
        if abs(phi_frac) > 1e-3:
            tmp = ndi_shift(tmp, shift=(0.0, phi_frac), order=1, mode="wrap", prefilter=False)

        r_shift = float(extra_r[k])
        if abs(r_shift) > 1e-3:
            tmp = ndi_shift(tmp, shift=(r_shift, 0.0), order=1, mode="nearest", prefilter=False)

        unwrapped[k] = tmp

    if verbose:
        print(
            f"[closure] enforced closure: Δφ(end-start)={float(drifts_phi_corr[-1]-drifts_phi_corr[0]):.3f}px, "
            f"Δr(end-start)={float(drifts_r_corr[-1]-drifts_r_corr[0]):.3f}px"
        )

    return unwrapped, drifts_r_corr, drifts_phi_corr


def closure_blend(unwrapped: np.ndarray, k: int = 5, verbose=True) -> np.ndarray:
    if k <= 0:
        return unwrapped
    n = unwrapped.shape[0]
    if 2 * k >= n:
        if verbose:
            print("[blend] K too large for N; skipping blend.")
        return unwrapped

    t = np.linspace(0, 1, k, endpoint=False, dtype=np.float32)
    w = 0.5 - 0.5 * np.cos(np.pi * t)

    out = unwrapped.copy()
    for i in range(k):
        a = out[n - k + i]
        b = out[i]
        wi = float(w[i])
        out[i] = (1.0 - wi) * b + wi * a
        out[n - k + i] = (1.0 - wi) * a + wi * b

    if verbose:
        print(f"[blend] Applied closure blend with K={k}.")
    return out
