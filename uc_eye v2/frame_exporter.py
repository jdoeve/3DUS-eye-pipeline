# frame_exporter.py
"""
Frame export utilities for UC-Eye volumes → ImageJ stacks.

Responsibilities
---------------
- Take a 3D Cartesian volume and export slices as individual image files.
- Support 8-bit or 16-bit grayscale outputs.
- Typically used with horizontal_slicer.extract_slices().

Inputs
------
- slices: list of 2D np.ndarrays.
- out_dir: folder to write images.
- basename: base filename for stack (e.g., "eye_raw").
- bit_depth: 8 or 16.
"""

from pathlib import Path
from typing import Iterable, List

import numpy as np
import imageio.v3 as iio


def _quantize_slice(sl: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")

    v = sl.astype(np.float32)
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros_like(v, dtype=np.uint8 if bit_depth == 8 else np.uint16)

    vmin = float(v[finite].min())
    vmax = float(v[finite].max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    v = (v - vmin) / (vmax - vmin)
    v = np.clip(v, 0.0, 1.0)

    if bit_depth == 8:
        return (v * 255.0).astype(np.uint8)
    else:
        return (v * 65535.0).astype(np.uint16)


def export_slices(
    slices: List[np.ndarray],
    out_dir: Path,
    basename: str,
    bit_depth: int = 16,
    ext: str = ".tif",
) -> None:
    """
    Export a list of slices as individual image files.

    Parameters
    ----------
    slices : list of np.ndarray
        2D arrays (H, W).
    out_dir : Path
        Directory for outputs.
    basename : str
        Base filename; images become basename_0000.tif, etc.
    bit_depth : int
        8 or 16.
    ext : str
        File extension, e.g. ".tif" or ".png".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, sl in enumerate(slices):
        q = _quantize_slice(sl, bit_depth=bit_depth)
        fname = out_dir / f"{basename}_{idx:04d}{ext}"
        iio.imwrite(fname, q)
    print(f"[Frames] Exported {len(slices)} slices to {out_dir}")
