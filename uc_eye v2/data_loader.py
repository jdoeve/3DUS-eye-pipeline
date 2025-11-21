# data_loader.py
"""
Data loading utilities for UC-Eye radial ultrasound pipeline.

Responsibilities
---------------
- Discover frame files (JPG/PNG) in a folder.
- Load them as a stack of grayscale numpy arrays.
- Validate basic integrity (count, shapes).

Coordinate assumptions
----------------------
- Each 2D frame is a radial B-scan image from the Eye Cubed.
- Pixel coordinates are in image (row, col) = (y, x) order.
- No geometric meaning is attached here; geometry is handled in geometry_calculator/interpolator.

Inputs
------
- input_dir: directory containing numbered JPG/PNG frames.

Outputs
-------
- paths: list[Path] of image files sorted numerically.
- imgs: np.ndarray of shape (N, H, W), dtype float32.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import imageio.v3 as iio

import uc_eye_pipeline as uceye


def load_image_paths(input_dir: Path) -> List[Path]:
    """
    Discover and numerically sort image files in a folder.

    Parameters
    ----------
    input_dir : Path
        Directory containing JPG/PNG ultrasound frames.

    Returns
    -------
    List[Path]
        Sorted image paths. May be empty if nothing is found.
    """
    return uceye.load_numeric_sorted_images(Path(input_dir))


def load_image_stack(paths: List[Path]) -> np.ndarray:
    """
    Load all frames and convert to a grayscale stack.

    Parameters
    ----------
    paths : list of Path
        Paths to 2D frame images.

    Returns
    -------
    imgs : np.ndarray
        Shape (N, H, W), dtype float32, intensities in native scale.

    Raises
    ------
    ValueError
        If fewer than 2 frames or inconsistent shapes are found.
    """
    if not paths:
        raise ValueError("No image files provided to load_image_stack().")

    frames = []
    shape0 = None

    for p in paths:
        arr = iio.imread(p)
        g = uceye.to_gray(arr)
        if shape0 is None:
            shape0 = g.shape
        elif g.shape != shape0:
            raise ValueError(
                f"Inconsistent frame shape: {p.name} has {g.shape}, "
                f"expected {shape0}."
            )
        frames.append(g.astype(np.float32))

    imgs = np.stack(frames, axis=0)  # (N, H, W)

    if imgs.shape[0] < 2:
        raise ValueError(
            f"Need ≥2 frames for a 3D reconstruction; got N={imgs.shape[0]}"
        )

    return imgs


def summarize_stack(imgs: np.ndarray) -> str:
    """
    Return a short human-readable summary of an image stack.

    Parameters
    ----------
    imgs : np.ndarray
        Shape (N, H, W).

    Returns
    -------
    str
        Summary string for logging.
    """
    N, H, W = imgs.shape
    return f"{N} frames of size {H}×{W}, dtype={imgs.dtype}"
