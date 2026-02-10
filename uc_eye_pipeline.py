#!/usr/bin/env python3
"""Compatibility shim for the modular UC-Eye package."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PKG_SRC = ROOT / "uceye_package" / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from uceye.cli import main  # noqa: E402
from uceye.pipeline import run_pipeline  # noqa: E402

__all__ = ["run_pipeline", "main"]

if __name__ == "__main__":
    main()
