# nrrd_exporter.py
"""
NRRD export utilities for UC-Eye volumes.

Responsibilities
---------------
- Convert numpy volumes to SimpleITK images.
- Set voxel spacing explicitly.
- Quantize to requested bit depth (8 or 16 bit) with optional normalization.
- Write NRRD files for Slicer.

Coordinate assumptions
----------------------
- Input volume is (Nx, Ny, Nz) in mm grid from resample_to_cartesian.
- Spacing is given as (sx, sy, sz) in mm; typically isotropic.

Inputs
------
- vol: np.ndarray, typically float32.
- spacing_mm: tuple (sx, sy, sz) in mm.
- bit_depth: 8 or 16.

Outputs
-------
- NRRD file written to disk.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk


def _quantize(
    vol: np.ndarray,
    bit_depth: int,
    auto_window: bool = True,
) -> np.ndarray:
    """
    Quantize floating volume to uint8 or uint16.

    Parameters
    ----------
    vol : np.ndarray
        Input float volume.
    bit_depth : int
        8 or 16.
    auto_window : bool
        If True, rescale intensities based on data min/max.
        If False, assumes vol is already in 0..1 or 0..255/65535.

    Returns
    -------
    np.ndarray
        Quantized volume.
    """
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")

    v = vol.astype(np.float32)

    if auto_window:
        finite = np.isfinite(v)
        if not finite.any():
            raise ValueError("Volume has no finite values to quantize.")
        vmin = float(v[finite].min())
        vmax = float(v[finite].max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        v = (v - vmin) / (vmax - vmin)

    if bit_depth == 8:
        v = np.clip(v, 0.0, 1.0) * 255.0
        return v.astype(np.uint8)
    else:
        v = np.clip(v, 0.0, 1.0) * 65535.0
        return v.astype(np.uint16)


def save_nrrd(
    vol: np.ndarray,
    output_path: Path,
    spacing_mm: Tuple[float, float, float],
    bit_depth: int = 16,
    auto_window: bool = True,
) -> None:
    """
    Save a 3D numpy volume as a NRRD file.

    Parameters
    ----------
    vol : np.ndarray
        3D volume, shape (Nx, Ny, Nz).
    output_path : Path
        Destination file (.nrrd).
    spacing_mm : (float, float, float)
        Voxel spacing in mm along (x, y, z).
    bit_depth : int
        8 or 16. Controls output pixel type.
    auto_window : bool
        If True, linearly stretches intensities to full bit range.

    Notes
    -----
    - SimpleITK expects numpy array ordered as (z, y, x). We treat axis 0 as x,
      so we transpose vol before writing to keep spacing aligned.
    """
    output_path = Path(output_path)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

    # Transpose (x, y, z) -> (z, y, x) for SimpleITK
    vol_sitk = np.transpose(vol, (2, 1, 0))
    vol_q = _quantize(vol_sitk, bit_depth=bit_depth, auto_window=auto_window)

    img = sitk.GetImageFromArray(vol_q)
    img.SetSpacing(tuple(float(s) for s in spacing_mm))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path))
    print(f"[NRRD] Saved {output_path}  (spacing={spacing_mm}, bit={bit_depth})")
