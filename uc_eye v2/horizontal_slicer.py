# horizontal_slicer.py
"""
Horizontal slice extraction for UC-Eye Cartesian volumes.

Responsibilities
---------------
- Extract 2D slices from a 3D volume along a chosen axis.
- Default behavior: axis 0 (x) is treated as anterior-posterior depth,
  so slices are "horizontal" cross-sections through the globe.

Inputs
------
- vol: np.ndarray, shape (Nx, Ny, Nz).
- axis: int, which axis to slice along (default 0).
- indices: optional list/array of slice indices.

Outputs
-------
- slices: list of 2D numpy arrays.
"""

from typing import Iterable, List

import numpy as np


def extract_slices(
    vol: np.ndarray,
    axis: int = 0,
    indices: Iterable[int] | None = None,
) -> List[np.ndarray]:
    """
    Extract 2D slices from a 3D volume.

    Parameters
    ----------
    vol : np.ndarray
        3D volume, e.g. from resample_to_cartesian, shape (Nx, Ny, Nz).
    axis : int
        Axis along which to slice (0, 1, or 2). Default 0 = depth.
    indices : iterable of int, optional
        Specific slice indices. If None, use all indices along that axis.

    Returns
    -------
    List[np.ndarray]
        List of 2D arrays, each of shape (H, W) depending on axis.
    """
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

    axis = int(axis)
    if axis < 0 or axis > 2:
        raise ValueError("axis must be 0, 1, or 2")

    n_slices = vol.shape[axis]
    if indices is None:
        indices = range(n_slices)

    slices: List[np.ndarray] = []
    for idx in indices:
        if idx < 0 or idx >= n_slices:
            raise IndexError(f"Slice index {idx} out of range [0, {n_slices-1}]")
        if axis == 0:
            sl = vol[idx, :, :]
        elif axis == 1:
            sl = vol[:, idx, :]
        else:
            sl = vol[:, :, idx]
        slices.append(sl.astype(np.float32, copy=False))

    return slices
