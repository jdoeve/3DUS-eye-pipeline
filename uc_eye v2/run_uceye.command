#!/bin/bash
clear
echo "========================================"
echo "   UC-Eye Pipeline Launcher v1.1"
echo "========================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Find the pipeline script
PIPELINE_SCRIPT=""
if [ -f "uc_eye_pipeline.py" ]; then
    PIPELINE_SCRIPT="uc_eye_pipeline.py"
elif [ -f "uceye_pipeline.py" ]; then
    PIPELINE_SCRIPT="uceye_pipeline.py"
elif [ -f "pipeline.py" ]; then
    PIPELINE_SCRIPT="pipeline.py"
else
    # Search for any Python file with "pipeline" in the name
    PIPELINE_SCRIPT=$(find . -maxdepth 1 -name "*pipeline*.py" | head -n 1)
fi

if [ -z "$PIPELINE_SCRIPT" ]; then
    echo "[ERROR] Pipeline script not found"
    echo "Expected: uc_eye_pipeline.py, uceye_pipeline.py, or pipeline.py"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[OK] Found pipeline: $PIPELINE_SCRIPT"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not installed"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[OK] Python3 found"
echo ""

python3 -c "import numpy, cv2, SimpleITK" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    python3 -m pip install -r requirements.txt
fi

echo ""
echo "SELECT IMAGE FOLDER"
echo "(You can drag and drop the folder into this window)"
echo "Tip: Remove any trailing spaces after pasting"
read -e -p "Image folder: " IMG_DIR_RAW

# Remove quotes and trim whitespace
IMG_DIR=$(echo "$IMG_DIR_RAW" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")

# Use echo -e to interpret backslash escapes (this actually works in bash)
IMG_DIR=$(echo -e "$IMG_DIR")

# Validate that path exists
if [ ! -d "$IMG_DIR" ]; then
    echo "[ERROR] Directory not found"
    echo ""
    echo "[DEBUG] Cleaned path: $IMG_DIR"
    echo "[DEBUG] Original input: $IMG_DIR_RAW"
    echo ""
    
    # Check parent directory
    PARENT_DIR=$(dirname "$IMG_DIR")
    if [ -d "$PARENT_DIR" ]; then
        echo "[DEBUG] Parent directory exists. Listing contents:"
        echo "----------------------------------------"
        ls -1 "$PARENT_DIR" 2>/dev/null | head -10
        echo "----------------------------------------"
        echo ""
        echo "Hint: Check if folder name matches exactly (case-sensitive)"
    else
        echo "[DEBUG] Checking if parent exists..."
        PARENT_DIR=$(echo -e "$(dirname "$IMG_DIR_RAW")")
        if [ -d "$PARENT_DIR" ]; then
            echo "[DEBUG] Parent found (after unescape). Listing:"
            echo "----------------------------------------"
            ls -1 "$PARENT_DIR" 2>/dev/null | head -10
            echo "----------------------------------------"
        else
            echo "[DEBUG] Parent directory not accessible"
        fi
    fi
    
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[OK] Found directory"
echo ""

echo "SELECT MASK FILE"
echo "(You can drag and drop the mask file into this window)"
read -e -p "Mask file: " MASK_PATH_RAW

# Remove quotes and whitespace, then interpret escapes
MASK_PATH=$(echo "$MASK_PATH_RAW" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
MASK_PATH=$(echo -e "$MASK_PATH")

# Validate that file exists
if [ ! -f "$MASK_PATH" ]; then
    echo "[ERROR] File not found: $MASK_PATH"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[OK] Found mask file: $MASK_PATH"
echo ""

echo "Starting pipeline..."
echo ""

# Generate output filename based on input directory
PARENT_DIR=$(dirname "$IMG_DIR")
FOLDER_NAME=$(basename "$PARENT_DIR")
OUTPUT_NRRD="${PARENT_DIR}/${FOLDER_NAME}_reconstruction.nrrd"

echo "Output will be saved to: $OUTPUT_NRRD"
echo ""

# Pass paths with proper flags for the pipeline
# CALIBRATED VERSION - proper geometric calibration with device-spec locked fan angle
# --depth: Physical depth in mm (Eye Cubed: 48mm for 10MHz BEFORE SoS correction)
# --voxel: Output voxel size (default 0.20mm)
# Fan angle locked to Eye Cubed spec: 52° (prevents mask-based errors)
python3 "$PIPELINE_SCRIPT" \
    --input-dir "$IMG_DIR" \
    --cone-mask "$MASK_PATH" \
    -o "$OUTPUT_NRRD" \
    --depth 48.0 \
    --voxel 0.20

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "   Pipeline completed successfully!"
    echo "========================================"
else
    echo "========================================"
    echo "   Pipeline failed with errors"
    echo "========================================"
fi

read -p "Press Enter to exit..."