"""Frame unwrapping into (r, phi) coordinates."""

from pathlib import Path

import cv2
import numpy as np

from .config import DEBUG_BYPASS_MASK, DEBUG_DUMP_MASK
from .geometry import GeometryConfig
from .io_utils import to_gray


def unwrap_frames(imgs, geom: GeometryConfig, mask_path: Path | None, drop_duplicate: bool = True, verbose: bool = True):
    h, w = imgs[0].shape[:2]
    apx_x, apx_y = geom.apex_xy
    r_max = int(np.ceil(geom.r_max_pix))
    wang = geom.image_height_eff

    rho = np.linspace(0, r_max, r_max, dtype=np.float32)
    phi_samples = np.linspace(geom.phi0, geom.phi1, wang, dtype=np.float32)
    phi_grid, rho_grid = np.meshgrid(phi_samples, rho, indexing="xy")
    src_x = (apx_x + rho_grid * np.cos(phi_grid)).astype(np.float32)
    src_y = (apx_y + rho_grid * np.sin(phi_grid)).astype(np.float32)

    if mask_path is not None and Path(mask_path).exists():
        mask_path = Path(mask_path)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask_gray > 128).astype(np.uint8)
    else:
        mask_bin = np.zeros((h, w), np.uint8)
        ths = np.linspace(geom.phi0, geom.phi1, 200, dtype=np.float32)
        arc_x = apx_x + geom.r_max_pix * np.cos(ths)
        arc_y = apx_y + geom.r_max_pix * np.sin(ths)
        pts = np.vstack([np.column_stack([arc_x, arc_y]), [apx_x, apx_y]]).astype(np.int32)
        cv2.fillPoly(mask_bin, [pts], 255)

    mask_unwrapped = cv2.remap(mask_bin, src_x, src_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def unwrap_one(img):
        g = to_gray(img)
        uw = cv2.remap(g, src_x, src_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        if DEBUG_DUMP_MASK:
            np.save("debug_mask_unwrapped.npy", (mask_unwrapped > 0).astype(np.uint8))
        if DEBUG_BYPASS_MASK:
            return uw
        return uw * (mask_unwrapped > 0)

    unwrapped = np.stack([unwrap_one(im) for im in imgs], axis=0).astype(np.float32)

    enable_theta_outlier_fix = True
    outlier_corr_threshold = 0.90

    if enable_theta_outlier_fix and unwrapped.shape[0] >= 3:
        n = unwrapped.shape[0]
        corrs_prev = np.zeros(n, dtype=np.float32)
        corrs_next = np.zeros(n, dtype=np.float32)

        def frame_corr(a, b):
            a = a.astype(np.float32)
            b = b.astype(np.float32)
            am = a.mean()
            bm = b.mean()
            num = ((a - am) * (b - bm)).sum(dtype=np.float64)
            den = np.sqrt(((a - am) ** 2).sum(dtype=np.float64) * ((b - bm) ** 2).sum(dtype=np.float64)) + 1e-12
            return float(num / den)

        for i in range(1, n - 1):
            corrs_prev[i] = frame_corr(unwrapped[i], unwrapped[i - 1])
            corrs_next[i] = frame_corr(unwrapped[i], unwrapped[i + 1])

        badness = np.minimum(corrs_prev, corrs_next)
        candidate_idx = np.arange(1, n - 1)
        candidate_badness = badness[1 : n - 1]
        worst_i = candidate_idx[np.argmin(candidate_badness)]
        worst_val = candidate_badness.min()

        if worst_val < outlier_corr_threshold:
            if verbose:
                print(
                    f"[unwrap] Theta outlier at index {worst_i} "
                    f"(min corr={worst_val:.3f}); replacing with neighbor average."
                )
            unwrapped[worst_i] = 0.5 * (unwrapped[worst_i - 1] + unwrapped[worst_i + 1])
        elif verbose:
            print(f"[unwrap] No theta outlier detected (min neighbor corr={worst_val:.3f}).")

    if unwrapped.shape[0] >= 2:
        a = unwrapped[0]
        b = unwrapped[-1]
        num = ((a - a.mean()) * (b - b.mean())).sum(dtype=np.float64)
        den = np.sqrt(((a - a.mean()) ** 2).sum(dtype=np.float64) * ((b - b.mean()) ** 2).sum(dtype=np.float64)) + 1e-12
        corr = float(num / den)

        if corr > 0.99:
            avg = 0.5 * (a + b)
            unwrapped[0] = avg
            unwrapped[-1] = avg
            if drop_duplicate:
                unwrapped = unwrapped[:-1]
            if verbose:
                state = "dropped last frame" if drop_duplicate else "kept both averaged"
                print(f"[unwrap] First/last correlation {corr:.4f} → treated as duplicate, {state}.")
        elif verbose:
            print(f"[unwrap] First/last correlation {corr:.4f} → distinct angles; no averaging or drop applied.")

    return unwrapped, mask_unwrapped
