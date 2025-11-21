# geometry_calculator.py
"""
Geometry configuration for UC-Eye radial ultrasound.

Responsibilities
---------------
- Use the cone mask PNG to derive fan geometry (apex, fan angle, r_max).
- Construct uc_eye_pipeline.GeometryConfig as the single source of truth.
- Optionally override displayed depth (mm).

Coordinate conventions
----------------------
- Pixel coordinates are image (x, y) in uc_eye_pipeline, with apex_xy in pixels.
- Angular fan bounds (phi0, phi1) come from the mask in radians.
- Radial spacing dr_mm is derived from displayed depth * SoS scale.

Inputs
------
- mask_path: path to cone mask PNG.
- depth_mm_override: optional float; if None, device default is used.

Outputs
-------
- geom: uc_eye_pipeline.GeometryConfig instance.
- calib_info: dict with mask metadata (apex, r_max, etc.).
"""

from pathlib import Path
from typing import Tuple, Dict, Any

import uc_eye_pipeline as uceye


def build_geometry(
    mask_path: Path,
    depth_mm_override: float | None = None,
    verbose: bool = True,
) -> Tuple[uceye.GeometryConfig, Dict[str, Any]]:
    """
    Calibrate fan geometry from cone mask and create GeometryConfig.

    Parameters
    ----------
    mask_path : Path
        Path to cone mask PNG (white = valid fan, black = outside).
    depth_mm_override : float, optional
        Physical depth in mm as displayed on the Eye Cubed console.
        If None, uc_eye_pipeline's DEFAULT/EYE_CUBED_DEPTH_MM is used.
    verbose : bool
        If True, prints calibration and geometry summary.

    Returns
    -------
    geom : GeometryConfig
        Central geometry object used by unwrapping and 3D resampling.
    calib_info : dict
        Raw calibration dictionary from calibrate_from_mask().
    """
    mask_path = Path(mask_path)

    calib_info = uceye.calibrate_from_mask(mask_path, verbose=verbose)

    depth_mm = depth_mm_override if depth_mm_override is not None else None

    geom = uceye.GeometryConfig(
        image_height=calib_info["image_height"],
        r_max_pix=calib_info["r_max_pix"],
        apex_xy=calib_info["apex_xy"],
        depth_mm=depth_mm,
        n_rows_mask=calib_info["n_rows_mask"],
        phi0_mask=calib_info["phi0_mask"],
        phi1_mask=calib_info["phi1_mask"],
    )

    if verbose:
        geom.print_summary()

    return geom, calib_info
