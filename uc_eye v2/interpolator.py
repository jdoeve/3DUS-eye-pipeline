# interpolator.py
"""
Unwrapping and 3D interpolation for UC-Eye radial ultrasound.

Responsibilities
---------------
- Call uc_eye_pipeline.unwrap_frames() to map radial B-scans into a polar stack.
- Optionally register frames around the sweep axis.
- Call uc_eye_pipeline.resample_to_cartesian() to create Cartesian volumes.
- Provide presets for RAW (minimal smoothing) and SMOOTH (default) reconstructions.

Coordinate transformations
--------------------------
1) unwrap_frames:
   (image x,y)  →  (r, phi) polar grid per frame.
   GeometryConfig supplies apex_xy, fan angle, and dr_mm.

2) resample_to_cartesian:
   (r, phi, theta)  →  (x, y, z) Cartesian cube in mm.
   Uses GeometryConfig for all physical scaling and fan limits.

Inputs
------
- imgs: np.ndarray of shape (N, H, W) or list of 2D arrays.
- geom: GeometryConfig instance.
- mask_path: Path to cone mask PNG.

Outputs
-------
- unwrapped: np.ndarray of shape (N, R, Wang) [theta, r, phi].
- vol: np.ndarray of shape (Nx, Ny, Nz) in float32.
"""

from pathlib import Path
from typing import Tuple

import numpy as np

import uc_eye_pipeline as uceye


def unwrap_and_register(
    imgs: np.ndarray,
    geom: uceye.GeometryConfig,
    mask_path: Path,
    do_registration: bool = True,
    drop_duplicate: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Unwrap radial frames into polar space and optionally register around theta.

    Parameters
    ----------
    imgs : np.ndarray
        Shape (N, H, W), grayscale frames.
    geom : GeometryConfig
        Fan geometry derived from cone mask.
    mask_path : Path
        Cone mask PNG used to constrain valid wedge region.
    do_registration : bool
        If True, run phase-only registration around phi (theta alignment).
    drop_duplicate : bool
        If True, drop final frame if it is a near-duplicate of first
        (Eye Cubed often repeats last frame).
    verbose : bool
        Print basic status.

    Returns
    -------
    unwrapped : np.ndarray
        Shape (N, R, Wang) in float32, polar representation.
        N = number of unique theta frames,
        R = radial pixels, Wang = lateral angular samples.
    """
    mask_path = Path(mask_path)

    if verbose:
        print(f"[Unwrap] Using mask: {mask_path}")
        print(f"[Unwrap] Input stack: {imgs.shape}")

    img_list = [imgs[k] for k in range(imgs.shape[0])]
    unwrapped, mask_unwrapped = uceye.unwrap_frames(
        img_list,
        geom,
        mask_path,
        drop_duplicate=drop_duplicate,
        verbose=verbose,
    )

    if do_registration:
        unwrapped = uceye.register_frames(unwrapped, verbose=verbose)

    if verbose:
        print(f"[Unwrap] Output polar stack: {unwrapped.shape}")
        if mask_unwrapped is not None:
            print(f"[Unwrap] Unwrapped mask shape: {mask_unwrapped.shape}")

    return unwrapped


def reconstruct_volume(
    unwrapped: np.ndarray,
    geom: uceye.GeometryConfig,
    voxel_mm: float,
    sigma_theta_deg: float,
    interp_order: int,
    slab_size: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Map polar stack to Cartesian volume using uc_eye_pipeline.resample_to_cartesian.

    Parameters
    ----------
    unwrapped : np.ndarray
        Shape (N, R, Wang) where N=theta, R=r, Wang=phi.
    geom : GeometryConfig
        Geometry for physical scaling.
    voxel_mm : float
        Target isotropic voxel size (mm).
    sigma_theta_deg : float
        Smoothing along sweep axis (theta) in degrees.
        0 → no smoothing (RAW); >0 → Gaussian along theta (SMOOTH).
    interp_order : int
        Interpolation order for scipy.ndimage.map_coordinates.
        0=nearest, 1=linear, 3=cubic.
    slab_size : int
        Number of x-slices per processing slab to limit memory usage.
    verbose : bool
        Print progress and geometry info.

    Returns
    -------
    vol : np.ndarray
        Shape (Nx, Ny, Nz), float32 Cartesian volume in mm grid.
    """
    vol = uceye.resample_to_cartesian(
        unwrapped,
        geom=geom,
        voxel_mm=float(voxel_mm),
        sigma_theta_deg=float(sigma_theta_deg),
        interp_order=int(interp_order),
        slab_size=int(slab_size),
        verbose=verbose,
    )
    return vol.astype(np.float32, copy=False)


# ---------- Preset helpers for RAW vs SMOOTH ---------------------------------

def reconstruct_raw_and_smooth(
    unwrapped: np.ndarray,
    geom: uceye.GeometryConfig,
    voxel_mm: float,
    raw_sigma_theta_deg: float = 0.0,
    raw_interp_order: int = 1,
    smooth_sigma_theta_deg: float = uceye.DEFAULT_SIGMA_THETA_DEG,
    smooth_interp_order: int = uceye.DEFAULT_INTERP_ORDER,
    slab_size: int = uceye.DEFAULT_SLAB_SIZE,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper to reconstruct BOTH RAW and SMOOTH volumes.

    RAW:
        - sigma_theta_deg ~ 0 (no theta smoothing)
        - interp_order ~ 1 (linear) to preserve higher frequency texture.

    SMOOTH:
        - sigma_theta_deg from geometry (default ~1°)
        - interp_order ~ 3 (cubic) for visually pleasant volume.

    Parameters
    ----------
    unwrapped, geom, voxel_mm, slab_size : see reconstruct_volume()
    raw_sigma_theta_deg : float
        Theta smoothing for RAW; default = 0.0 (off).
    raw_interp_order : int
        Interp order for RAW; default = 1 (linear).
    smooth_sigma_theta_deg : float
        Theta smoothing for SMOOTH (default from uc_eye_pipeline).
    smooth_interp_order : int
        Interp order for SMOOTH (default from uc_eye_pipeline).
    slab_size : int
        Slab size for both reconstructions.
    verbose : bool
        Print which recon is being performed.

    Returns
    -------
    vol_raw, vol_smooth : np.ndarray
        Two Cartesian volumes, same grid, different interpolation.
    """
    if verbose:
        print("\n[Recon] RAW volume (minimal smoothing)...")
    vol_raw = reconstruct_volume(
        unwrapped,
        geom=geom,
        voxel_mm=voxel_mm,
        sigma_theta_deg=raw_sigma_theta_deg,
        interp_order=raw_interp_order,
        slab_size=slab_size,
        verbose=verbose,
    )

    if verbose:
        print("\n[Recon] SMOOTH volume (default smoothing)...")
    vol_smooth = reconstruct_volume(
        unwrapped,
        geom=geom,
        voxel_mm=voxel_mm,
        sigma_theta_deg=smooth_sigma_theta_deg,
        interp_order=smooth_interp_order,
        slab_size=slab_size,
        verbose=verbose,
    )

    return vol_raw, vol_smooth
