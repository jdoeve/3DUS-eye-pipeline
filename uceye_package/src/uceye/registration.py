"""Frame-to-frame registration in unwrapped (r, phi) space."""

import numpy as np
from scipy.ndimage import shift as ndi_shift


def estimate_phi_shift_1d(ref_uw, mov_uw):
    s_ref = ref_uw.sum(axis=0).astype(np.float32)
    s_mov = mov_uw.sum(axis=0).astype(np.float32)
    s_ref -= s_ref.mean()
    s_mov -= s_mov.mean()
    n = s_ref.size
    f1 = np.fft.rfft(s_ref, n * 2)
    f2 = np.fft.rfft(s_mov, n * 2)
    cc = np.fft.irfft(f1 * np.conj(f2))
    k = int(np.argmax(cc))
    if k >= n:
        k -= 2 * n
    return k


def phase_corr_shift_2d(a: np.ndarray, b: np.ndarray, eps: float = 1e-8):
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)

    fa = np.fft.fft2(a0)
    fb = np.fft.fft2(b0)
    r = fa * np.conj(fb)
    r /= np.abs(r) + eps

    cc = np.abs(np.fft.ifft2(r))
    r0, c0 = np.unravel_index(np.argmax(cc), cc.shape)
    nr, np_ = cc.shape

    if r0 > nr // 2:
        r0 -= nr
    if c0 > np_ // 2:
        c0 -= np_

    def parabolic(fm1, f0, fp1):
        denom = fm1 - 2 * f0 + fp1
        if abs(denom) < 1e-12:
            return 0.0
        return 0.5 * (fm1 - fp1) / denom

    rr = r0 % nr
    cc0 = c0 % np_
    dr_sub = parabolic(cc[(rr - 1) % nr, cc0], cc[rr, cc0], cc[(rr + 1) % nr, cc0])
    dphi_sub = parabolic(cc[rr, (cc0 - 1) % np_], cc[rr, cc0], cc[rr, (cc0 + 1) % np_])

    return float(r0 + dr_sub), float(c0 + dphi_sub)


def register_frames_level1(
    unwrapped: np.ndarray,
    verbose: bool = True,
    use_roi: bool = True,
    roi_r_frac=(0.20, 0.90),
    roi_phi_frac=(0.05, 0.95),
    apply_subpixel: bool = True,
):
    unwrapped = np.asarray(unwrapped, dtype=np.float32)
    n, r, p = unwrapped.shape
    ref_idx = n // 2
    ref = unwrapped[ref_idx]

    if use_roi:
        r0 = int(roi_r_frac[0] * r)
        r1 = int(roi_r_frac[1] * r)
        p0 = int(roi_phi_frac[0] * p)
        p1 = int(roi_phi_frac[1] * p)
        ref_roi = ref[r0:r1, p0:p1]
    else:
        r0, r1, p0, p1 = 0, r, 0, p
        ref_roi = ref

    drifts_r = np.zeros(n, dtype=np.float32)
    drifts_phi = np.zeros(n, dtype=np.float32)
    out = unwrapped.copy()

    for k in range(n):
        mov = unwrapped[k]
        mov_roi = mov[r0:r1, p0:p1]
        sh_r, sh_phi = phase_corr_shift_2d(ref_roi, mov_roi)

        drifts_r[k] = sh_r
        drifts_phi[k] = sh_phi

        phi_int = int(np.round(sh_phi))
        tmp = np.roll(mov, shift=phi_int, axis=1)

        phi_frac = float(sh_phi - phi_int)
        if apply_subpixel and abs(phi_frac) > 1e-3:
            tmp = ndi_shift(tmp, shift=(0.0, phi_frac), order=1, mode="wrap", prefilter=False)

        if apply_subpixel and abs(sh_r) > 1e-3:
            tmp = ndi_shift(tmp, shift=(sh_r, 0.0), order=1, mode="nearest", prefilter=False)
        else:
            r_int = int(np.round(sh_r))
            if r_int != 0:
                tmp = ndi_shift(tmp, shift=(float(r_int), 0.0), order=0, mode="nearest", prefilter=False)

        out[k] = tmp

    if verbose:
        print(f"Registered {n} frames to mid-frame reference (Level-1: Δr + Δφ)")
        print(f"[register] Δφ range: {drifts_phi.min():.2f}..{drifts_phi.max():.2f} px")
        print(f"[register] Δr  range: {drifts_r.min():.2f}..{drifts_r.max():.2f} px")
        print(f"[register] closure Δφ (end-start): {float(drifts_phi[-1]-drifts_phi[0]):.2f} px")
        print(f"[register] closure Δr  (end-start): {float(drifts_r[-1]-drifts_r[0]):.2f} px")

    return out, drifts_r, drifts_phi
