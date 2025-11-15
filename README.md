# UC-Eye Radial Ultrasound Pipeline

Converts radial ultrasound image sequences to calibrated 3D Cartesian NRRD volumes with comprehensive quality control.

## Quick Start

### Windows
1. Install Python 3.8+ from python.org
2. Double-click `run_uceye.bat`

### Mac/Linux
1. Right-click 3DUS-Eye folder → Select new terminal at folder
2. Run: `chmod +x run_uceye.command`
3. Double-click `run_uceye.command`

## Usage

1. Run launcher script
2. Drag/drop image folder (populates path automatically)
3. Drag/drop mask file (populates path automatically)
4. Pipeline runs with full QC checks
5. Find NRRD output in parent folder

## Pipeline Features

### Geometric Calibration
- **Mask-based φ calibration**: Uses actual angular bounds from mask instead of device spec (52°)
- **Speed of Sound correction**: 0.9622× scale factor for Eye Cubed 10MHz
- **Apex detection**: Robust multi-strategy approach with micro-search optimization (±3px)
- **Physical depth**: 48mm displayed depth → 46.2mm effective depth after SoS correction

### Quality Control (QC) Checks

Pipeline automatically performs 4 QC checks:

1. **Row-arc verification**: Compares expected vs measured arc lengths at multiple depths (10, 20, 30, 40mm) to validate Δφ calculation
2. **Apex micro-search**: Tests small offsets (±2px) to minimize depth-wise radius drift
3. **Segmentation bias check**: Compares gradient-based edge detection vs threshold-based radius measurement at 24mm depth
4. **Rotation uniformity**: Validates rotation consistency by comparing PCA axis ratios between even/odd frame subsets

### Output

- **NRRD volume**: `{folder}_reconstruction.nrrd` in parent directory
- **QC images**: Diagnostic visualizations in `QC/` subfolder
- **Calibration JSON**: Reusable geometry parameters with mask φ-bounds

### Command Line Options

```bash
python3 uc_eye_pipeline.py \
    --input-dir /path/to/images \
    --cone-mask /path/to/mask.png \
    -o output.nrrd \
    --depth 48.0 \
    --voxel 0.20
```

**Parameters:**
- `--depth`: Physical depth in mm (default: 48.0 for Eye Cubed 10MHz)
- `--voxel`: Output voxel size in mm (default: 0.20)

## Requirements

- Python 3.8+ (tested with NumPy 2.0+)
- NumPy, OpenCV, SimpleITK, SciPy, matplotlib, imageio
- Auto-installs on first run via `requirements.txt`

## Troubleshooting

- **Check QC images**: Review `QC/` folder for diagnostic plots
- **Console output**: All QC metrics printed with `[GEOM]` and `[QC]` tags
- **Mask issues**: Ensure mask is binary with clear cone region
- **Path errors**: Remove trailing spaces when drag/dropping paths

## Segmentation

1. Load NRRD into 3D Slicer
2. Adjust uc_eye_preprocess.py to match input name to NRRD
3. Copy paste into Python interactive tool in 3D Slicer (It will run automatically)
4. Use _masked file for segmentation
5. See Protocol for additional documentation

---
Dr. Andrew Browne Lab, created by Jeffrey Doeve