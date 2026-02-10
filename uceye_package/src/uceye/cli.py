"""Command line interface for UC-Eye pipeline."""

import argparse
import sys
from pathlib import Path

import matplotlib

# Must be set before any pyplot import for headless environments.
matplotlib.use("Agg", force=True)

from .config import (
    DEFAULT_DEPTH_MM,
    DEFAULT_INTERP_ORDER,
    DEFAULT_SLAB_SIZE,
    DEFAULT_SIGMA_THETA_DEG,
    DEFAULT_THETA_SPAN_DEG,
    DEFAULT_VOXEL_MM,
)
from .io_utils import resolve_paths_from_args


def main():
    parser = argparse.ArgumentParser(description="UC-Eye Pipeline (Calibrated + SoS/Lateral Scaling) — compatible CLI")

    parser.add_argument("img_dir", nargs="?", type=Path, help="Images directory (positional)")
    parser.add_argument("mask_path", nargs="?", type=Path, help="Cone mask PNG (positional)")

    parser.add_argument("--input-dir", type=Path, help="Images directory")
    parser.add_argument("--cone-mask", type=Path, help="Cone mask PNG")
    parser.add_argument("--out-dir", type=Path, help="Output directory (file auto-named as *_EYE_VOL_rNN.nrrd)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Explicit output NRRD file path")

    parser.add_argument("--depth", type=float, default=DEFAULT_DEPTH_MM, help="Physical depth in mm (Eye Cubed default: 48mm BEFORE SoS correction)")
    parser.add_argument("--voxel", type=float, default=DEFAULT_VOXEL_MM, help="Target mm/voxel in output (lateral mm/pixel calibration)")
    parser.add_argument("--sigma-theta", type=float, default=DEFAULT_SIGMA_THETA_DEG, help="Smoothing (deg) around sweep axis")
    parser.add_argument("--theta-span-deg", type=float, default=DEFAULT_THETA_SPAN_DEG, help="Effective θ sweep span in degrees")
    parser.add_argument("--interp-order", type=int, default=DEFAULT_INTERP_ORDER, help="Interpolation order for map_coordinates (0..5)")
    parser.add_argument("--slab-size", type=int, default=DEFAULT_SLAB_SIZE, help="How many x-slices to process at a time")
    parser.add_argument("--no-register", action="store_true", help="Disable phi registration between frames")
    parser.add_argument("--keep-duplicate", action="store_true", help="Keep last frame even if duplicate")
    parser.add_argument("--quiet", action="store_true", help="Less console output")

    args = parser.parse_args()
    img_dir, mask_path, output_file, out_dir_for_iter = resolve_paths_from_args(args)

    if img_dir is None:
        parser.print_help(sys.stderr)
        print("\nERROR: You must provide an images directory (positional or --input-dir).", file=sys.stderr)
        sys.exit(1)

    from .pipeline import run_pipeline

    try:
        run_pipeline(
            img_dir=img_dir,
            mask_path=mask_path,
            output_path=output_file,
            depth_mm=args.depth,
            voxel_mm=args.voxel,
            sigma_theta_deg=args.sigma_theta,
            theta_span_deg=args.theta_span_deg,
            interp_order=args.interp_order,
            slab_size=args.slab_size,
            do_registration=not args.no_register,
            drop_duplicate=not args.keep_duplicate,
            verbose=not args.quiet,
            out_dir_for_iter=out_dir_for_iter,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
