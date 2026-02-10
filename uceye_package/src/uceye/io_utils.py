"""I/O and utility helpers used by the UC-Eye pipeline."""

import re
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np


def fmt_t(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


@contextmanager
def timer(label: str, verbose: bool = True):
    t0 = time.time()
    yield
    dt = time.time() - t0
    if verbose:
        print(f"[{label}] {fmt_t(dt)}")


def ang_norm(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_numeric_sorted_images(folder: Path):
    patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    files = []
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        print(f"[WARNING] Invalid images directory: {folder}")
        return []

    for pat in patterns:
        try:
            files.extend(folder.glob(pat))
        except Exception as e:
            print(f"[WARNING] Error globbing {pat}: {e}")

    if files:
        filtered = [p for p in files if not p.name.startswith("._")]
        if len(filtered) != len(files):
            print(f"[INFO] Skipped {len(files) - len(filtered)} hidden resource files (._*)")
        files = filtered

    if not files:
        print(f"[WARNING] No images found in: {folder}")
        try:
            contents = list(folder.iterdir())
            print(f"[DEBUG] Directory has {len(contents)} items")
            if contents:
                print(f"[DEBUG] First items: {[f.name for f in contents[:5]]}")
        except Exception as e:
            print(f"[DEBUG] Could not list directory: {e}")
        return []

    def num_key(p: Path):
        s = p.stem
        try:
            return (0, int(s))
        except Exception:
            return (1, s)

    files = sorted(files, key=num_key)
    print(f"[DEBUG] Found {len(files)} image files")
    return files


def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def largest_component(bin_img):
    n, labels = cv2.connectedComponents(bin_img.astype(np.uint8))
    if n <= 1:
        return bin_img.astype(np.uint8)
    areas = [(labels == i).sum() for i in range(1, n)]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8)


def find_main_contour(bin_img):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert cnts, "No contour found in cone mask."
    cnt = max(cnts, key=cv2.contourArea)
    pts_xy = cnt.reshape(-1, 2)
    return np.stack([pts_xy[:, 1], pts_xy[:, 0]], axis=1)


def line_fit_cv(points_rc):
    pts_xy = points_rc[:, ::-1].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 1e-2, 1e-2).ravel().tolist()
    return (float(x0), float(y0), float(vx), float(vy))


def intersect_lines(l1, l2, eps=1e-9):
    x1, y1, vx1, vy1 = l1
    x2, y2, vx2, vy2 = l2
    denom = vx1 * vy2 - vy1 * vx2
    if abs(denom) < eps:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
    return (x1 + t * vx1, y1 + t * vy1)


_EYE_TOKENS = {"OD", "OS", "RIGHT", "LEFT", "R", "L"}


def _extract_eye_token(text: str) -> str | None:
    parts = re.split(r"[-_\s]+", text.upper())
    for p in parts:
        if p in _EYE_TOKENS:
            return "OD" if p in {"OD", "RIGHT", "R"} else "OS"
    return None


def _clean_base_from_folder(folder_name: str) -> str:
    base = folder_name
    noise_tokens = [r"\bRadial\b", r"\bPrimary\b", r"\bSecondary\b", r"\bOD\b", r"\bOS\b"]
    for tok in noise_tokens:
        base = re.sub(rf"(^|[_\-\s]){tok}([_\-\s]|$)", " ", base, flags=re.IGNORECASE)

    base = re.sub(r"\b\d{2}-\d{2}-\d{4}\b", " ", base)
    base = re.sub(r"\b\d{2}-\d{2}\b", " ", base)
    base = re.sub(r"\b(\d+)\s*mm\b", r"\1mm", base, flags=re.IGNORECASE)
    base = re.sub(r"\s*[-–—]{1,}\s*", " ", base)
    base = re.sub(r"\s*_\s*", "_", base)
    return re.sub(r"\s{2,}", " ", base).strip()


def _next_run_suffix(target_dir: Path, prefix: str) -> str:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = list(target_dir.glob(f"{prefix}_r*.nrrd"))
    if not existing:
        return "r01"

    max_n = 0
    for p in existing:
        m = re.search(r"_r(\d{2})\.nrrd$", p.name)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return f"r{max_n+1:02d}"


def generate_iterative_filename(img_dir: Path, out_dir: Path | None, verbose: bool = True) -> Path:
    img_dir = Path(img_dir)
    parent = img_dir.parent
    folder_name = img_dir.name

    eye_tok = _extract_eye_token(folder_name) or _extract_eye_token(parent.name) or "UNK"
    prefix_base = _clean_base_from_folder(folder_name)
    prefix = f"{prefix_base}_{eye_tok}_EYE_VOL"
    run_suffix = _next_run_suffix(out_dir or parent, prefix)

    out_dir_final = out_dir or parent
    out_dir_final.mkdir(parents=True, exist_ok=True)
    out_file = out_dir_final / f"{prefix}_{run_suffix}.nrrd"

    if verbose:
        print(f"[OUTPUT] Using iterative filename: {out_file}")
    return out_file


def resolve_paths_from_args(args):
    img_dir = args.input_dir or args.img_dir
    mask_path = args.cone_mask or args.mask_path
    output_file = args.output
    out_dir = args.out_dir
    return img_dir, mask_path, output_file, out_dir
