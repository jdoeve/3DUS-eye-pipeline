# main_pipeline.py
"""
Modular UC-Eye pipeline wrapper with RAW + SMOOTH outputs.

This script reuses the core geometry, unwrapping, and resampling logic
implemented in uc_eye_pipeline.py, but organizes the workflow into
modular components and adds dual outputs:

- RAW: minimal smoothing, linear interpolation.
- SMOOTH: default theta smoothing + cubic interpolation.

Usage (examples)
----------------
1) Config file (recommended):

    python main_pipeline.py --config eye_config.yaml

2) Direct CLI:

    python main_pipeline.py \\
        --input-dir /path/to/frames \\
        --cone-mask cone_mask_GIMP.png \\
        --out-dir /path/to/output \\
        --depth 48 --voxel 0.20 \\
        --export-frames

Config file
-----------
YAML or JSON with keys:

    input_dir: "/path/to/frames"
    cone_mask: "/path/to/cone_mask_GIMP.png"
    out_dir:   "/path/to/output"

    depth_mm: 48.0
    voxel_mm: 0.20

    do_registration: true
    drop_duplicate: true

    raw:
      enabled: true
      sigma_theta_deg: 0.0
      interp_order: 1
      bit_depth: 16

    smooth:
      enabled: true
      sigma_theta_deg: 1.0
      interp_order: 3
      bit_depth: 8

    frames:
      enabled: true
      axis: 0
      bit_depth: 16
      basename_raw: "eye_raw"
      basename_smooth: "eye_smooth"

All parameters can also be overridden from the CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

import numpy as np

import uc_eye_pipeline as uceye
import data_loader
import geometry_calculator
import interpolator
import horizontal_slicer
import nrrd_exporter
import frame_exporter


# ---------- Config handling ---------------------------------------------------

def _load_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in (".yml", ".yaml"):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is not installed but a YAML config was provided.")
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    else:
        with open(path, "r") as f:
            return json.load(f)


def _merge_cli_into_config(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """CLI flags override config values when provided."""
    if args.input_dir is not None:
        cfg["input_dir"] = str(args.input_dir)
    if args.cone_mask is not None:
        cfg["cone_mask"] = str(args.cone_mask)
    if args.out_dir is not None:
        cfg["out_dir"] = str(args.out_dir)
    if args.depth is not None:
        cfg["depth_mm"] = float(args.depth)
    if args.voxel is not None:
        cfg["voxel_mm"] = float(args.voxel)

    # toggles
    cfg.setdefault("do_registration", not args.no_register)
    cfg.setdefault("drop_duplicate", not args.keep_duplicate)

    cfg.setdefault("raw", {})
    cfg.setdefault("smooth", {})
    cfg.setdefault("frames", {})

    # raw & smooth specific overrides not exposed on CLI for now (keep simple)
    return cfg


# ---------- Core pipeline -----------------------------------------------------

def run_pipeline(cfg: Dict[str, Any]) -> Tuple[Path | None, Path | None]:
    """
    Execute the full UC-Eye pipeline with RAW + SMOOTH outputs.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (see module docstring).

    Returns
    -------
    (raw_path, smooth_path) : tuple of Path or None
        Paths to NRRD outputs if created.
    """
    # Resolve core paths
    input_dir = Path(cfg["input_dir"])
    cone_mask = Path(cfg["cone_mask"])
    out_dir = Path(cfg.get("out_dir", input_dir / "uceye_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basic numerical parameters
    depth_mm = float(cfg.get("depth_mm", uceye.DEFAULT_DEPTH_MM))
    voxel_mm = float(cfg.get("voxel_mm", uceye.DEFAULT_VOXEL_MM))

    do_registration = bool(cfg.get("do_registration", True))
    drop_duplicate = bool(cfg.get("drop_duplicate", True))

    raw_cfg = cfg.get("raw", {}) or {}
    smooth_cfg = cfg.get("smooth", {}) or {}
    frames_cfg = cfg.get("frames", {}) or {}

    # 1) Load frames
    paths = data_loader.load_image_paths(input_dir)
    imgs = data_loader.load_image_stack(paths)
    print(f"[Main] Loaded frames: {data_loader.summarize_stack(imgs)}")

    # 2) Build geometry from cone mask
    with uceye.timer("Geometry & mask calibration", verbose=True):
        geom, calib_info = geometry_calculator.build_geometry(
            cone_mask, depth_mm_override=depth_mm, verbose=True
        )

    # 3) Unwrap + register
    with uceye.timer("Unwrap & registration", verbose=True):
        unwrapped = interpolator.unwrap_and_register(
            imgs,
            geom=geom,
            mask_path=cone_mask,
            do_registration=do_registration,
            drop_duplicate=drop_duplicate,
            verbose=True,
        )

    # 4) Reconstruct RAW + SMOOTH volumes
    slab_size = int(cfg.get("slab_size", uceye.DEFAULT_SLAB_SIZE))

    raw_enabled = bool(raw_cfg.get("enabled", True))
    smooth_enabled = bool(smooth_cfg.get("enabled", True))

    vol_raw = vol_smooth = None

    if raw_enabled or smooth_enabled:
        with uceye.timer("3D reconstruction (RAW + SMOOTH)", verbose=True):
            vol_raw, vol_smooth = interpolator.reconstruct_raw_and_smooth(
                unwrapped,
                geom=geom,
                voxel_mm=voxel_mm,
                raw_sigma_theta_deg=float(raw_cfg.get("sigma_theta_deg", 0.0)),
                raw_interp_order=int(raw_cfg.get("interp_order", 1)),
                smooth_sigma_theta_deg=float(
                    smooth_cfg.get("sigma_theta_deg", uceye.DEFAULT_SIGMA_THETA_DEG)
                ),
                smooth_interp_order=int(
                    smooth_cfg.get("interp_order", uceye.DEFAULT_INTERP_ORDER)
                ),
                slab_size=slab_size,
                verbose=True,
            )

    # 5) Export NRRDs
    spacing = (voxel_mm, voxel_mm, voxel_mm)
    raw_path = smooth_path = None

    if vol_raw is not None and raw_enabled:
        bit_raw = int(raw_cfg.get("bit_depth", 16))
        raw_name = cfg.get("raw_basename", "eye_raw")
        raw_path = out_dir / f"{raw_name}_{voxel_mm:.3f}mm.nrrd"
        nrrd_exporter.save_nrrd(
            vol_raw,
            raw_path,
            spacing_mm=spacing,
            bit_depth=bit_raw,
            auto_window=False,
        )

    if vol_smooth is not None and smooth_enabled:
        bit_smooth = int(smooth_cfg.get("bit_depth", 8))
        smooth_name = cfg.get("smooth_basename", "eye_smooth")
        smooth_path = out_dir / f"{smooth_name}_{voxel_mm:.3f}mm.nrrd"
        nrrd_exporter.save_nrrd(
            vol_smooth,
            smooth_path,
            spacing_mm=spacing,
            bit_depth=bit_smooth,
            auto_window=False,
        )

    # 6) Optional: Export frame stacks for ImageJ
    if frames_cfg.get("enabled", False):
        axis = int(frames_cfg.get("axis", 0))
        bit_frames = int(frames_cfg.get("bit_depth", 16))

        if vol_raw is not None and raw_enabled:
            slices_raw = horizontal_slicer.extract_slices(vol_raw, axis=axis)
            basename_raw = frames_cfg.get("basename_raw", "eye_raw_slice")
            frame_exporter.export_slices(
                slices_raw,
                out_dir / "frames_raw",
                basename=basename_raw,
                bit_depth=bit_frames,
            )

        if vol_smooth is not None and smooth_enabled:
            slices_smooth = horizontal_slicer.extract_slices(vol_smooth, axis=axis)
            basename_smooth = frames_cfg.get("basename_smooth", "eye_smooth_slice")
            frame_exporter.export_slices(
                slices_smooth,
                out_dir / "frames_smooth",
                basename=basename_smooth,
                bit_depth=bit_frames,
            )

    print("\n[Main] Pipeline complete.")
    if raw_path:
        print(f"  RAW NRRD   : {raw_path}")
    if smooth_path:
        print(f"  SMOOTH NRRD: {smooth_path}")
    print(f"  Output dir : {out_dir}")

    return raw_path, smooth_path


# ---------- CLI ---------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UC-Eye modular pipeline (RAW + SMOOTH volumes)"
    )
    p.add_argument("--config", type=Path, help="YAML/JSON config file")

    # Keep CLI roughly compatible with uc_eye_pipeline
    p.add_argument("--input-dir", type=Path, help="Images directory")
    p.add_argument("--cone-mask", type=Path, help="Cone mask PNG")
    p.add_argument("--out-dir", type=Path, help="Output directory")

    p.add_argument("--depth", type=float, help="Physical depth in mm (Eye Cubed display)")
    p.add_argument("--voxel", type=float, help="Target voxel size in mm")

    p.add_argument("--no-register", action="store_true", help="Disable theta registration")
    p.add_argument("--keep-duplicate", action="store_true", help="Keep duplicate last frame")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}

    if args.config:
        cfg = _load_config(args.config)

    cfg = _merge_cli_into_config(cfg, args)

    if "input_dir" not in cfg or "cone_mask" not in cfg:
        parser.print_help()
        raise SystemExit(
            "\nERROR: You must provide --input-dir and --cone-mask "
            "or specify them in the config file."
        )

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
