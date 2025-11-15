"""
================================================================================
PHANTOM DISTANCE CALIBRATION - 3D Slicer Single-Paste Script
================================================================================
Copy and paste this entire script into 3D Slicer's Python console.

OUTPUTS:
  - Apex to Sphere Distance (mm)
  - Sphere Radius (mm)
  - % within Sphere band
  - PCA Eigenvalues
  - % Error vs Ground Truth (if ground truth values provided)

Author: Auto-generated for 3D Slicer
================================================================================
"""

# ============================================================================
# CONFIGURATION - Edit these variables to match your data
# ============================================================================

# Volume/Segmentation Names
VOLUME_NAME = "Flour _ half sphere_5 balls_85 mm  - Radial - OD - Primary - 09-25-2025 - 11-52_reconstruction_7_masked"
SEGMENTATION_NAME = "Segmentation_14"
SEGMENT_NAMES = ["Flour_8"]  # List of segment names to analyze

# Fiducial Names
APEX_FIDUCIAL_NAME = "Apex"

# Ground Truth Values (set to None to disable error calculation)
GROUND_TRUTH_APEX_TO_SPHERE_MM = 30.0  # Known distance from apex to sphere center (mm)
GROUND_TRUTH_SPHERE_RADIUS_MM = 7.2    # Known sphere radius (mm)

# Sphere Analysis Parameters
SPHERE_TOLERANCE_FRACTION = 0.05  # ±5% tolerance for sphere radius
SUBSAMPLE_EVERY_N = 1  # 1 = use all points, >1 = subsample for speed
MIN_REQUIRED_POINTS = 200  # Minimum surface points for valid analysis

# PCA Parameters
PCA_SEGMENT_NAME = "Flour_8"  # Segment for PCA analysis
PCA_REF_VOLUME_NAME = None  # Reference volume for sweep axis (None = auto-detect)

# Output
CREATE_CSV = True  # Save results to CSV on Desktop
CSV_FILENAME = "phantom_calibration_results.csv"

# ============================================================================
# SCRIPT - Do not edit below unless you know what you're doing
# ============================================================================

import numpy as np
import vtk
import slicer
import os
from itertools import combinations

print("=" * 80)
print(" PHANTOM DISTANCE CALIBRATION - Starting Analysis")
print("=" * 80)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_node_by_name(name, className=None):
    """Get node by exact name with optional class constraint."""
    try:
        node = slicer.util.getNode(name)
        if className and node.GetClassName() != className:
            return None
        return node
    except:
        return None

def find_volume_node(targetName):
    """Find volume node by name or use smart selection."""
    vols = []
    vols.extend(slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode'))
    vols.extend(slicer.util.getNodesByClass('vtkMRMLVectorVolumeNode'))
    if not vols:
        return None

    if targetName:
        try:
            node = slicer.util.getNode(targetName)
            return node
        except:
            pass

    # Fallback to first volume
    return vols[0] if vols else None

def labelmap_center_of_mass_ras(labelmapNode):
    """Compute center of mass in RAS coordinates."""
    imageData = labelmapNode.GetImageData()
    if not imageData:
        return None

    arr = slicer.util.arrayFromVolume(labelmapNode)
    idxs = np.argwhere(arr > 0)
    if idxs.size == 0:
        return None

    mean_kji = idxs.mean(axis=0)
    centerIJK = [float(mean_kji[2]), float(mean_kji[1]), float(mean_kji[0])]

    ijkToRas = vtk.vtkMatrix4x4()
    labelmapNode.GetIJKToRASMatrix(ijkToRas)
    p4 = [centerIJK[0], centerIJK[1], centerIJK[2], 1.0]
    out = [0.0, 0.0, 0.0, 1.0]
    ijkToRas.MultiplyPoint(p4, out)
    return out[:3]

def export_segment_to_labelmap(segmentationNode, segmentId, referenceVolumeNode=None):
    """Export single segment to temporary labelmap."""
    tempSegNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'temp_seg_export')
    tempSegNode.GetSegmentation().CopySegmentFromSegmentation(segmentationNode.GetSegmentation(), segmentId)

    labelmapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'temp_labelmap')
    if referenceVolumeNode:
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            tempSegNode, labelmapNode, referenceVolumeNode
        )
    else:
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            tempSegNode, labelmapNode
        )

    slicer.mrmlScene.RemoveNode(tempSegNode)
    return labelmapNode

def get_segment_polydata(segmentationNode, segmentId):
    """Get closed surface polydata for a segment."""
    # Ensure closed surface exists
    seg = segmentationNode.GetSegmentation()
    repName = slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName()
    if not seg.ContainsRepresentation(repName):
        seg.CreateRepresentation(repName)

    segObj = seg.GetSegment(segmentId)
    polyData = vtk.vtkPolyData.SafeDownCast(segObj.GetRepresentation(repName))

    # Apply transform if needed
    parentTransformNode = segmentationNode.GetParentTransformNode()
    if parentTransformNode and polyData:
        toWorld = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(parentTransformNode, None, toWorld)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(polyData)
        tf.SetTransform(toWorld)
        tf.Update()
        return tf.GetOutput()

    return polyData

def compute_surface_centroid(polyData):
    """Compute centroid from surface vertices using VTK."""
    com = vtk.vtkCenterOfMass()
    com.SetUseScalarsAsWeights(False)
    com.SetInputData(polyData)
    com.Update()
    center_tuple = com.GetCenter()
    return np.array(center_tuple, dtype=float)

def compute_radii_stats(polyData, centroid, subsample=1):
    """Compute radii statistics from surface to centroid."""
    pts = polyData.GetPoints()
    n = pts.GetNumberOfPoints()
    if n == 0:
        return None, 0, 0, 0, 0

    from vtk.util import numpy_support
    vtk_array = pts.GetData()
    xyz = numpy_support.vtk_to_numpy(vtk_array)

    if subsample > 1:
        xyz = xyz[::subsample, :]

    distances = np.linalg.norm(xyz - centroid[None, :], axis=1)
    if distances.size == 0:
        return None, 0, 0, 0, 0

    mean_r = float(distances.mean())
    std_r = float(distances.std(ddof=1)) if distances.size > 1 else 0.0
    min_r = float(distances.min())
    max_r = float(distances.max())

    return distances, mean_r, std_r, min_r, max_r

def compute_sphere_band_percentage(distances, tolerance_fraction):
    """Compute % of points within tolerance band."""
    if distances is None or distances.size == 0:
        return 0.0

    mean_r = distances.mean()
    lo = mean_r * (1 - tolerance_fraction)
    hi = mean_r * (1 + tolerance_fraction)
    mask = (distances >= lo) & (distances <= hi)
    return 100.0 * mask.sum() / distances.size

def compute_pca_eigenvalues(polyData):
    """Compute PCA eigenvalues from surface points."""
    from vtk.util import numpy_support
    pts = polyData.GetPoints()
    n = pts.GetNumberOfPoints()
    if n == 0:
        return None

    vtk_array = pts.GetData()
    xyz = numpy_support.vtk_to_numpy(vtk_array)

    centroid = xyz.mean(axis=0)
    X = xyz - centroid
    C = np.dot(X.T, X) / max(len(X) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]

    return eigenvalues

# ============================================================================
# STEP 1: FIND OR CREATE APEX FIDUCIAL
# ============================================================================

print("\n[Step 1] Finding Apex Fiducial...")

apexNode = get_node_by_name(APEX_FIDUCIAL_NAME, "vtkMRMLMarkupsFiducialNode")

if apexNode and apexNode.GetNumberOfControlPoints() > 0:
    apexRAS = [0.0, 0.0, 0.0]
    apexNode.GetNthControlPointPosition(0, apexRAS)
    print(f"  ✓ Using existing Apex at: [{apexRAS[0]:.2f}, {apexRAS[1]:.2f}, {apexRAS[2]:.2f}]")
else:
    print("  ! Apex fiducial not found, attempting auto-detection...")
    volumeNode = find_volume_node(VOLUME_NAME)

    if not volumeNode:
        raise RuntimeError("No volume found. Please load your volume or create Apex fiducial manually.")

    # Robust apex detection using multiple strategies
    imageData = volumeNode.GetImageData()
    if not imageData:
        raise RuntimeError("Volume has no image data")

    dims = imageData.GetDimensions()
    spacing = imageData.GetSpacing()

    # Strategy 1: Test multiple apex candidate locations
    apex_candidates = [
        [dims[0]//2, dims[1]//2, 0],           # Center of first slice (axial)
        [dims[0]//2, 0, dims[2]//2],           # Center of first row (sagittal)
        [0, dims[1]//2, dims[2]//2],           # Center of first column (coronal)
        [dims[0]//2, dims[1]//2, dims[2]-1],   # Center of last slice
        [0, 0, 0],                              # Origin
    ]

    # Find which candidate has the smallest non-zero region (likely apex)
    best_apex_ijk = None
    min_nonzero_count = float('inf')

    for candidate in apex_candidates:
        count = 0
        region_size = 5  # Check 5x5x5 region

        for dx in range(-region_size, region_size+1):
            for dy in range(-region_size, region_size+1):
                for dz in range(-region_size, region_size+1):
                    x = candidate[0] + dx
                    y = candidate[1] + dy
                    z = candidate[2] + dz

                    if (0 <= x < dims[0] and 0 <= y < dims[1] and 0 <= z < dims[2]):
                        value = imageData.GetScalarComponentAsFloat(x, y, z, 0)
                        if value > 0:
                            count += 1

        if 0 < count < min_nonzero_count:
            min_nonzero_count = count
            best_apex_ijk = candidate

    # Strategy 2: Analyze slice areas to find narrowest point (fallback)
    if best_apex_ijk is None or min_nonzero_count == float('inf'):
        print("    Using slice analysis to find apex...")

        # Check each dimension to find where cone narrows
        for dim in range(3):  # Check all three dimensions
            min_area = float('inf')
            best_idx = 0

            # Sample slices along this dimension
            num_samples = min(20, dims[dim])
            for i in range(num_samples):
                idx = i * dims[dim] // num_samples

                # Count non-zero voxels in this slice
                count = 0
                if dim == 0:  # YZ plane
                    for y in range(dims[1]):
                        for z in range(dims[2]):
                            if imageData.GetScalarComponentAsFloat(idx, y, z, 0) > 0:
                                count += 1
                elif dim == 1:  # XZ plane
                    for x in range(dims[0]):
                        for z in range(dims[2]):
                            if imageData.GetScalarComponentAsFloat(x, idx, z, 0) > 0:
                                count += 1
                else:  # XY plane
                    for x in range(dims[0]):
                        for y in range(dims[1]):
                            if imageData.GetScalarComponentAsFloat(x, y, idx, 0) > 0:
                                count += 1

                if 0 < count < min_area:
                    min_area = count
                    best_idx = idx

            # Set apex based on dimension with clearest narrowing
            if min_area < float('inf'):
                if dim == 0:
                    best_apex_ijk = [best_idx, dims[1]//2, dims[2]//2]
                elif dim == 1:
                    best_apex_ijk = [dims[0]//2, best_idx, dims[2]//2]
                else:
                    best_apex_ijk = [dims[0]//2, dims[1]//2, best_idx]
                break

    # Default to origin if no apex found
    if best_apex_ijk is None:
        print("    Warning: Could not determine apex automatically, using origin")
        best_apex_ijk = [0, 0, 0]

    print(f"    Apex IJK coordinates: {best_apex_ijk}")

    # Transform to RAS coordinates
    ijkToRas = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(ijkToRas)
    apexRAS = [0, 0, 0, 1]
    ijkToRas.MultiplyPoint([best_apex_ijk[0], best_apex_ijk[1], best_apex_ijk[2], 1], apexRAS)
    apexRAS = apexRAS[:3]

    # Create fiducial
    apexNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", APEX_FIDUCIAL_NAME)
    apexNode.AddControlPoint(apexRAS)
    apexNode.SetNthControlPointLabel(0, "Cone Apex")

    displayNode = apexNode.GetDisplayNode()
    if displayNode:
        displayNode.SetSelectedColor(1, 0, 0)
        displayNode.SetTextScale(3)
        displayNode.SetGlyphScale(3)

    print(f"  ✓ Auto-detected Apex at: [{apexRAS[0]:.2f}, {apexRAS[1]:.2f}, {apexRAS[2]:.2f}]")

# ============================================================================
# STEP 2: ANALYZE SEGMENTS
# ============================================================================

print("\n[Step 2] Analyzing Segments...")

segNode = get_node_by_name(SEGMENTATION_NAME, "vtkMRMLSegmentationNode")
if not segNode:
    raise RuntimeError(f"Segmentation '{SEGMENTATION_NAME}' not found.")

seg = segNode.GetSegmentation()
refVol = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')

# Build segment name to ID mapping
nameToId = {}
for i in range(seg.GetNumberOfSegments()):
    name = seg.GetNthSegment(i).GetName()
    sid = seg.GetNthSegmentID(i)
    nameToId[name] = sid

# Verify all requested segments exist
missing = [n for n in SEGMENT_NAMES if n not in nameToId]
if missing:
    raise RuntimeError(f"Missing segments: {missing}. Available: {list(nameToId.keys())}")

# Results storage
results = []

for segName in SEGMENT_NAMES:
    print(f"\n--- Analyzing: {segName} ---")
    segId = nameToId[segName]

    # Get center of mass
    lm = export_segment_to_labelmap(segNode, segId, refVol)
    try:
        centerRAS = labelmap_center_of_mass_ras(lm)
        if centerRAS is None:
            print(f"  ✗ No voxels found for {segName}")
            continue
    finally:
        slicer.mrmlScene.RemoveNode(lm)

    # Get surface polydata
    polyData = get_segment_polydata(segNode, segId)
    if not polyData or polyData.GetNumberOfPoints() < MIN_REQUIRED_POINTS:
        print(f"  ✗ Insufficient surface points ({polyData.GetNumberOfPoints() if polyData else 0})")
        continue

    # Compute apex to center distance
    apex_to_sphere_distance = float(np.linalg.norm(np.array(apexRAS) - np.array(centerRAS)))

    # Compute surface centroid and radius stats
    surfaceCentroid = compute_surface_centroid(polyData)
    distances, mean_radius, std_radius, min_radius, max_radius = compute_radii_stats(
        polyData, surfaceCentroid, SUBSAMPLE_EVERY_N
    )

    # Compute % within sphere band
    pct_in_band = compute_sphere_band_percentage(distances, SPHERE_TOLERANCE_FRACTION)

    # Compute PCA eigenvalues
    eigenvalues = compute_pca_eigenvalues(polyData)

    # Calculate % error vs ground truth (if provided)
    error_apex_to_sphere = None
    error_radius = None

    if GROUND_TRUTH_APEX_TO_SPHERE_MM is not None:
        error_apex_to_sphere = 100.0 * (apex_to_sphere_distance - GROUND_TRUTH_APEX_TO_SPHERE_MM) / GROUND_TRUTH_APEX_TO_SPHERE_MM

    if GROUND_TRUTH_SPHERE_RADIUS_MM is not None:
        error_radius = 100.0 * (mean_radius - GROUND_TRUTH_SPHERE_RADIUS_MM) / GROUND_TRUTH_SPHERE_RADIUS_MM

    # Store results
    result = {
        'segment_name': segName,
        'apex_to_sphere_mm': apex_to_sphere_distance,
        'radius_mean_mm': mean_radius,
        'radius_std_mm': std_radius,
        'radius_min_mm': min_radius,
        'radius_max_mm': max_radius,
        'pct_within_band': pct_in_band,
        'eigenvalue_1': eigenvalues[0] if eigenvalues is not None else 0,
        'eigenvalue_2': eigenvalues[1] if eigenvalues is not None else 0,
        'eigenvalue_3': eigenvalues[2] if eigenvalues is not None else 0,
        'center_x': centerRAS[0],
        'center_y': centerRAS[1],
        'center_z': centerRAS[2],
        'error_apex_to_sphere_pct': error_apex_to_sphere,
        'error_radius_pct': error_radius,
        'ground_truth_apex_mm': GROUND_TRUTH_APEX_TO_SPHERE_MM,
        'ground_truth_radius_mm': GROUND_TRUTH_SPHERE_RADIUS_MM
    }
    results.append(result)

    # Print results
    print(f"  Apex to Sphere Distance: {apex_to_sphere_distance:.2f} mm")
    print(f"  Radius: {mean_radius:.2f} ± {std_radius:.2f} mm (range: [{min_radius:.2f}, {max_radius:.2f}])")
    print(f"  % within ±{int(SPHERE_TOLERANCE_FRACTION*100)}% band: {pct_in_band:.1f}%")
    if eigenvalues is not None:
        print(f"  PCA Eigenvalues: λ1={eigenvalues[0]:.3f}, λ2={eigenvalues[1]:.3f}, λ3={eigenvalues[2]:.3f}")

    # Print error if ground truth provided
    if error_apex_to_sphere is not None:
        print(f"  Error (Apex to Sphere): {error_apex_to_sphere:+.2f}% (measured: {apex_to_sphere_distance:.2f} mm, ground truth: {GROUND_TRUTH_APEX_TO_SPHERE_MM:.2f} mm)")
    if error_radius is not None:
        print(f"  Error (Radius):         {error_radius:+.2f}% (measured: {mean_radius:.2f} mm, ground truth: {GROUND_TRUTH_SPHERE_RADIUS_MM:.2f} mm)")

# ============================================================================
# STEP 3: SUMMARY OUTPUT
# ============================================================================

print("\n" + "=" * 80)
print(" SUMMARY RESULTS")
print("=" * 80)

if not results:
    print("No segments analyzed successfully.")
else:
    for r in results:
        print(f"\n{r['segment_name']}:")
        print(f"  Apex to Sphere Distance:  {r['apex_to_sphere_mm']:.2f} mm")
        print(f"  Radius:                   {r['radius_mean_mm']:.2f} mm")
        print(f"  % within Sphere band:     {r['pct_within_band']:.1f}%")
        print(f"  PCA Eigenvalues:          λ1={r['eigenvalue_1']:.3f}, λ2={r['eigenvalue_2']:.3f}, λ3={r['eigenvalue_3']:.3f}")

        # Print errors if ground truth available
        if r['error_apex_to_sphere_pct'] is not None:
            print(f"  % Error (Apex Distance):  {r['error_apex_to_sphere_pct']:+.2f}%")
        if r['error_radius_pct'] is not None:
            print(f"  % Error (Radius):         {r['error_radius_pct']:+.2f}%")

# ============================================================================
# STEP 4: CREATE SLICER TABLE
# ============================================================================

print("\n[Step 4] Creating Results Table...")

try:
    tableNode = slicer.util.getNode("Phantom_Calibration_Results")
    slicer.mrmlScene.RemoveNode(tableNode)
except:
    pass

tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "Phantom_Calibration_Results")
table = tableNode.GetTable()

# Define columns
columns = [
    ("Segment", "string"),
    ("Apex_to_Sphere_mm", "float"),
    ("Radius_mm", "float"),
    ("Radius_Std_mm", "float"),
    ("Pct_in_Band", "float"),
    ("Eigenvalue_1", "float"),
    ("Eigenvalue_2", "float"),
    ("Eigenvalue_3", "float"),
    ("Error_Apex_pct", "float"),
    ("Error_Radius_pct", "float")
]

for col_name, col_type in columns:
    if col_type == "string":
        col = vtk.vtkStringArray()
    else:
        col = vtk.vtkFloatArray()
    col.SetName(col_name)
    table.AddColumn(col)

# Add rows
for r in results:
    table.InsertNextBlankRow()
    row_idx = table.GetNumberOfRows() - 1

    table.GetColumn(0).SetValue(row_idx, r['segment_name'])
    table.GetColumn(1).SetValue(row_idx, r['apex_to_sphere_mm'])
    table.GetColumn(2).SetValue(row_idx, r['radius_mean_mm'])
    table.GetColumn(3).SetValue(row_idx, r['radius_std_mm'])
    table.GetColumn(4).SetValue(row_idx, r['pct_within_band'])
    table.GetColumn(5).SetValue(row_idx, r['eigenvalue_1'])
    table.GetColumn(6).SetValue(row_idx, r['eigenvalue_2'])
    table.GetColumn(7).SetValue(row_idx, r['eigenvalue_3'])

    # Handle None values for error columns
    if r['error_apex_to_sphere_pct'] is not None:
        table.GetColumn(8).SetValue(row_idx, r['error_apex_to_sphere_pct'])
    else:
        table.GetColumn(8).SetValue(row_idx, 0.0)  # Use 0 for N/A in float column

    if r['error_radius_pct'] is not None:
        table.GetColumn(9).SetValue(row_idx, r['error_radius_pct'])
    else:
        table.GetColumn(9).SetValue(row_idx, 0.0)  # Use 0 for N/A in float column

print(f"  ✓ Table created: {tableNode.GetName()}")

# ============================================================================
# STEP 5: EXPORT TO CSV (OPTIONAL)
# ============================================================================

if CREATE_CSV and results:
    print("\n[Step 5] Exporting to CSV...")

    desktop = os.path.expanduser("~/Desktop")
    csvPath = os.path.join(desktop, CSV_FILENAME)

    try:
        with open(csvPath, 'w') as f:
            # Header
            f.write("Segment,Apex_to_Sphere_mm,Radius_mm,Radius_Std_mm,Pct_in_Band,")
            f.write("Eigenvalue_1,Eigenvalue_2,Eigenvalue_3,")
            f.write("Error_Apex_pct,Error_Radius_pct,")
            f.write("Ground_Truth_Apex_mm,Ground_Truth_Radius_mm,")
            f.write("Center_X,Center_Y,Center_Z\n")

            # Data rows
            for r in results:
                f.write(f"{r['segment_name']},{r['apex_to_sphere_mm']:.2f},")
                f.write(f"{r['radius_mean_mm']:.2f},{r['radius_std_mm']:.2f},")
                f.write(f"{r['pct_within_band']:.1f},")
                f.write(f"{r['eigenvalue_1']:.3f},{r['eigenvalue_2']:.3f},{r['eigenvalue_3']:.3f},")

                # Error columns
                if r['error_apex_to_sphere_pct'] is not None:
                    f.write(f"{r['error_apex_to_sphere_pct']:.2f},")
                else:
                    f.write("N/A,")

                if r['error_radius_pct'] is not None:
                    f.write(f"{r['error_radius_pct']:.2f},")
                else:
                    f.write("N/A,")

                # Ground truth values
                if r['ground_truth_apex_mm'] is not None:
                    f.write(f"{r['ground_truth_apex_mm']:.2f},")
                else:
                    f.write("N/A,")

                if r['ground_truth_radius_mm'] is not None:
                    f.write(f"{r['ground_truth_radius_mm']:.2f},")
                else:
                    f.write("N/A,")

                # Center coordinates
                f.write(f"{r['center_x']:.2f},{r['center_y']:.2f},{r['center_z']:.2f}\n")

        print(f"  ✓ CSV saved: {csvPath}")
    except Exception as e:
        print(f"  ✗ Could not save CSV: {e}")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "=" * 80)
print(" ANALYSIS COMPLETE")
print("=" * 80)
print("\nResults available in:")
print("  - Slicer table: 'Phantom_Calibration_Results'")
if CREATE_CSV:
    print(f"  - CSV file: ~/Desktop/{CSV_FILENAME}")
print()
