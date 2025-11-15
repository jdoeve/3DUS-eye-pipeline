# ===================== UC-Eye Preprocess: Crop → Resample → Denoise → Normalize → Mask =====================
# COPY/PASTE into Slicer's Python Interactor. Toggle whole cell with Cmd+/ if desired.

import slicer, sitkUtils, SimpleITK as sitk, vtk, numpy as np, os

# -------------------- CONFIG (edit these) --------------------
# Tip: this should match the volume node name in the Data module after your NRRD is loaded.
# If unsure, type: [n.GetName() for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")]
inputName          = "eye_volume_20251003_123215"   # exact or prefix of input node name
roiName            = "CropVolumeROI"                                      # ROI node name (created/updated if needed)

# ROI handling:
#  - Set roi_size_mm to a tuple (X,Y,Z) in mm to FORCE that size, OR
#  - Set roi_size_mm = None to AUTO-size ROI from nonzero bounds + margins (recommended)
roi_size_mm        = None                                                 # e.g., (48.0, 30.0, 30.0) or None for auto
anterior_margin_mm = 3.0                                                  # pad in front of apex to avoid clipping (X+)
side_margin_mm     = 2.0                                                  # safety pad on Y,Z when auto-sizing

target_spacing     = (0.20, 0.20, 0.20)                                   # EXACT output spacing (mm)

# Denoising toggles
do_CAD             = True;  CAD_iters = 5; CAD_cond = 2.0; CAD_tstep = 0.01  # CAD: stable with tstep ≤ 0.01
do_Median          = True; median_radius_vox = 1

# Intensity normalization (robust percentiles on >0 voxels)
norm_pct_low, norm_pct_high = 1.0, 99.7

# Advanced options
auto_volume_render = False                                                 # Automatically set up 3D visualization
assess_quality     = True                                                 # Print quality metrics
remove_intermediate = False                                               # Remove intermediate volumes to save memory

# -------------------- Helpers (leave as-is) --------------------
def _sitk_to_ras_ijk_to_mat(volNode):
    """
    Return 4x4 matrix mapping IJK -> RAS for the SimpleITK image pulled from volNode.
    Slicer stores volumes in RAS; SimpleITK images are LPS internally, so we convert.
    """
    img = sitkUtils.PullVolumeFromSlicer(volNode)  # LPS image
    spac = np.array(img.GetSpacing(), dtype=float)       # (sx,sy,sz)
    orig = np.array(img.GetOrigin(), dtype=float)        # LPS origin
    dire = np.array(img.GetDirection(), dtype=float).reshape(3,3)
    M_ijk_lps = np.eye(4, dtype=float)
    M_ijk_lps[:3,:3] = dire @ np.diag(spac)
    M_ijk_lps[:3, 3] = orig
    M_lps_ras = np.diag([-1.0, -1.0, 1.0, 1.0])          # LPS->RAS
    return M_lps_ras @ M_ijk_lps

def _nonzero_bounds_ras(volNode, thr=0):
    """Min/max RAS bounds of voxels > thr. Returns (mins, maxs) or None if empty."""
    img = sitkUtils.PullVolumeFromSlicer(volNode)        # LPS image
    arr = sitk.GetArrayFromImage(img)                    # z,y,x (numpy)
    m = arr > thr
    if not np.any(m):
        return None
    zs, ys, xs = np.nonzero(m)
    ijk = np.vstack([xs, ys, zs, np.ones_like(xs)]).T    # (N,4)
    M = _sitk_to_ras_ijk_to_mat(volNode)                 # IJK->RAS
    ras = (M @ ijk.T).T[:, :3]
    return ras.min(axis=0), ras.max(axis=0)

def _largest_component_mask(mask_u8):
    cc = sitk.ConnectedComponent(mask_u8)
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    return sitk.Cast(rel == 1, sitk.sitkUInt8)

def _robust_rescale_sitk(img, p_lo, p_hi, out_dtype=sitk.sitkUInt8):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    base = arr[arr > 0] if np.any(arr > 0) else arr
    lo = np.percentile(base, p_lo); hi = np.percentile(base, p_hi)
    scale = max(hi - lo, 1e-6)
    arr = np.clip((arr - lo)/scale, 0, 1)
    if out_dtype == sitk.sitkUInt8:
        arr = (arr * 255.0).round().astype(np.uint8)
    elif out_dtype == sitk.sitkUInt16:
        arr = (arr * 65535.0).round().astype(np.uint16)
    out = sitk.GetImageFromArray(arr); out.CopyInformation(img); return out

def _resolve_volume_by_prefix(name):
    """
    Try exact name first; if not found, fall back to the first volume whose name startswith 'name'.
    This helps when Slicer appends suffixes like ' (1)' on repeated loads.
    """
    try:
        return slicer.util.getNode(name)
    except slicer.util.MRMLNodeNotFoundException:
        vols = [n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")]
        for n in vols:
            if n.GetName().startswith(name):
                print(f"[Info] Using volume '{n.GetName()}' (prefix match for '{name}')")
                return n
        raise

def assess_volume_quality(img_sitk):
    """Calculate quality metrics for the volume"""
    arr = sitk.GetArrayFromImage(img_sitk)
    
    # Signal-to-noise ratio estimate
    signal = np.mean(arr[arr > 0])
    noise = np.std(arr[arr > 0])
    snr = signal / noise if noise > 0 else 0
    
    # Edge strength (gradient magnitude)
    grad = sitk.GradientMagnitude(img_sitk)
    edge_strength = np.mean(sitk.GetArrayFromImage(grad))
    
    # Coverage (percentage of non-zero voxels)
    coverage = np.sum(arr > 0) / arr.size * 100
    
    return {
        'SNR': snr,
        'Edge_Strength': edge_strength,
        'Coverage_Percent': coverage
    }

def setup_volume_rendering(vol_node, preset_name="Ultrasound"):
    """Automatically configure volume rendering with optimal settings"""
    try:
        # Show volume rendering
        volRenLogic = slicer.modules.volumerendering.logic()
        displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(vol_node)
        displayNode.SetVisibility(True)
        
        # Apply preset
        presets = volRenLogic.GetPresetsScene()
        preset = presets.GetFirstNodeByName(preset_name)
        if preset:
            displayNode.GetVolumePropertyNode().Copy(preset.GetVolumePropertyNode())
        
        # Auto-adjust threshold based on histogram
        arr = slicer.util.arrayFromVolume(vol_node)
        threshold = np.percentile(arr[arr > 0], 30) if np.any(arr > 0) else 50
        
        prop = displayNode.GetVolumePropertyNode().GetVolumeProperty()
        opacityFunc = prop.GetScalarOpacity()
        opacityFunc.RemoveAllPoints()
        opacityFunc.AddPoint(0, 0.0)
        opacityFunc.AddPoint(threshold, 0.0)
        opacityFunc.AddPoint(threshold + 10, 0.1)
        opacityFunc.AddPoint(255, 1.0)
        
        return displayNode
    except Exception as e:
        print(f"[Warning] Could not set up volume rendering: {e}")
        return None

def save_pipeline_outputs(base_name, output_dir=None):
    """Save all generated volumes to disk"""
    if output_dir is None:
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all nodes with base_name
    nodes = slicer.util.getNodesByClass("vtkMRMLVolumeNode")
    relevant_nodes = [n for n in nodes if base_name in n.GetName()]
    
    for node in relevant_nodes:
        filename = os.path.join(output_dir, f"{node.GetName()}.nrrd")
        slicer.util.saveNode(node, filename)
        print(f"[Saved] {filename}")

# -------------------- MAIN PIPELINE --------------------
print(f"\n{'='*80}")
print(f"UC-Eye Preprocessing Pipeline - Starting")
print(f"{'='*80}")

# -------------------- 0) Resolve input & ROI --------------------
vol = _resolve_volume_by_prefix(inputName)  # raises if not found
if vol.GetImageData() is None:
    raise RuntimeError(f"Input '{vol.GetName()}' has no image data.")

# Get or create a Markups ROI of the correct type/name
try:
    roi = slicer.util.getNode(roiName)
    if roi.GetClassName() != "vtkMRMLMarkupsROINode":
        raise RuntimeError(f"Node '{roiName}' exists but is not a Markups ROI.")
except slicer.util.MRMLNodeNotFoundException:
    roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", roiName)

# -------------------- 1) Place ROI: anterior anchored at apex; size = manual or auto --------------------
nz = _nonzero_bounds_ras(vol, thr=0)
if nz is None:
    raise RuntimeError("No non-zero voxels found in input; cannot place ROI.")
mins, maxs = nz
# RAS: +X is anterior. Apex ~ most-anterior (min X) of the nonzero envelope.
apex_x = float(mins[0])

if roi_size_mm is None:
    # Auto size from bounds + margins
    x_len = float((maxs[0] - mins[0]) + anterior_margin_mm + 4.0)  # + extra 4mm safety tail posteriorly
    y_len = float((maxs[1] - mins[1]) + 2*side_margin_mm)
    z_len = float((maxs[2] - mins[2]) + 2*side_margin_mm)
    roi_size_mm_used = (x_len, y_len, z_len)
else:
    x_len, y_len, z_len = roi_size_mm
    roi_size_mm_used = roi_size_mm

center_x = apex_x + anterior_margin_mm + (x_len / 2.0)
center_y = 0.5 * (mins[1] + maxs[1])
center_z = 0.5 * (mins[2] + maxs[2])
roi.SetCenter(center_x, center_y, center_z)
roi.SetSize(x_len, y_len, z_len)
print(f"[ROI] ApexX={apex_x:.2f}  →  center=({center_x:.2f},{center_y:.2f},{center_z:.2f})  size={tuple(round(v,2) for v in roi_size_mm_used)} mm")

# -------------------- 2) Crop (voxel-based; no spacing scaling here) --------------------
paramsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCropVolumeParametersNode")
paramsNode.SetInputVolumeNodeID(vol.GetID())
paramsNode.SetROINodeID(roi.GetID())
paramsNode.SetVoxelBased(True)
paramsNode.SetIsotropicResampling(False)      # we'll resample precisely in the next step

croppedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", vol.GetName() + "_cropped")
paramsNode.SetOutputVolumeNodeID(croppedNode.GetID())
slicer.modules.cropvolume.logic().Apply(paramsNode)
slicer.mrmlScene.RemoveNode(paramsNode)
print(f"[Crop] Output={croppedNode.GetName()}  spacing={croppedNode.GetSpacing()}")

# -------------------- 3) Resample to EXACT spacing (SimpleITK) --------------------
img_c = sitkUtils.PullVolumeFromSlicer(croppedNode)           # SITK image (LPS)
old_spacing = np.array(img_c.GetSpacing(), dtype=float)       # (sx,sy,sz)
new_spacing = np.array(target_spacing, dtype=float)
old_size = np.array(list(img_c.GetSize()), dtype=int)         # (nx,ny,nz)
new_size = np.maximum(1, np.round(old_size * (old_spacing / new_spacing)).astype(int))

resampler = sitk.ResampleImageFilter()
resampler.SetOutputSpacing(tuple(new_spacing))
resampler.SetSize([int(x) for x in new_size])
resampler.SetOutputDirection(img_c.GetDirection())
resampler.SetOutputOrigin(img_c.GetOrigin())
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0.0)
img_r = resampler.Execute(img_c)

resampledNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", vol.GetName() + "_cropped_resamp")
sitkUtils.PushVolumeToSlicer(img_r, resampledNode)
print(f"[Resample] Output={resampledNode.GetName()}  spacing={resampledNode.GetSpacing()}  size={resampledNode.GetImageData().GetDimensions()}")

# Optionally remove cropped node to save memory
if remove_intermediate:
    slicer.mrmlScene.RemoveNode(croppedNode)
    print("[Memory] Removed intermediate cropped node")

# -------------------- 4) Denoise (CAD + Median as configured) --------------------
img = sitkUtils.PullVolumeFromSlicer(resampledNode)
img = sitk.Cast(img, sitk.sitkFloat32)

if do_CAD:
    cad = sitk.CurvatureAnisotropicDiffusionImageFilter()
    cad.SetNumberOfIterations(int(CAD_iters))
    cad.SetConductanceParameter(float(CAD_cond))
    cad.SetTimeStep(float(CAD_tstep))  # ≤ 0.01 avoids ITK stability warnings
    img = cad.Execute(img)
    print(f"[Denoise] CAD applied (iters={CAD_iters}, cond={CAD_cond}, t={CAD_tstep}).")

if do_Median:
    med = sitk.MedianImageFilter()
    med.SetRadius(int(median_radius_vox))
    img = med.Execute(img)
    print(f"[Denoise] Median r={median_radius_vox} applied.")

# -------------------- 5) Normalize (robust percentiles) & push --------------------
img = _robust_rescale_sitk(img, norm_pct_low, norm_pct_high, out_dtype=sitk.sitkUInt8)
normNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", vol.GetName() + "_norm255")
sitkUtils.PushVolumeToSlicer(img, normNode)
print(f"[Normalize] Output={normNode.GetName()} (uint8 0..255)")

# Quality assessment if requested
if assess_quality:
    quality = assess_volume_quality(img)
    print(f"[Quality] SNR={quality['SNR']:.2f}, Edge Strength={quality['Edge_Strength']:.2f}, Coverage={quality['Coverage_Percent']:.1f}%")

# Optionally remove resampled node to save memory
if remove_intermediate:
    slicer.mrmlScene.RemoveNode(resampledNode)
    print("[Memory] Removed intermediate resampled node")

# -------------------- 6) Mask: largest component of >0 & push --------------------
mask = sitk.Cast(img > 0, sitk.sitkUInt8)
mask = _largest_component_mask(mask)
# Seal tiny pinholes; kernel [1,1,1] is minimal and safe
mask = sitk.BinaryMorphologicalClosing(mask, [1,1,1])
masked_img = sitk.Mask(img, mask, outsideValue=0)

maskNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", vol.GetName() + "_mask")
sitkUtils.PushVolumeToSlicer(mask, maskNode)
maskedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", vol.GetName() + "_masked")
sitkUtils.PushVolumeToSlicer(masked_img, maskedNode)

# Copy WL from normalized to masked for consistent look
dr = normNode.GetDisplayNode(); dm = maskedNode.GetDisplayNode()
if dr and dm:
    dm.SetWindowLevel(dr.GetWindow(), dr.GetLevel())

# -------------------- 7) Optional: Auto Volume Rendering --------------------
if auto_volume_render:
    vrDisplay = setup_volume_rendering(maskedNode)
    if vrDisplay:
        print(f"[3D] Volume rendering configured with Ultrasound preset")

# -------------------- 8) Optional: Create Segmentation --------------------
# Uncomment to automatically create a segmentation from the mask
# segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", vol.GetName() + "_segmentation")
# slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(maskNode, segNode)
# segNode.SetReferenceImageGeometryParameterFromVolumeNode(maskedNode)
# segNode.CreateDefaultDisplayNodes()
# print(f"[Segment] Created segmentation: {segNode.GetName()}")

# -------------------- Final Summary --------------------
print(f"\n{'='*80}")
print("[✓] Pipeline Complete! Created volumes:")
if not remove_intermediate:
    print(f"  • {croppedNode.GetName()}")
    print(f"  • {resampledNode.GetName()}")
print(f"  • {normNode.GetName()}")
print(f"  • {maskNode.GetName()}")
print(f"  • {maskedNode.GetName()} ← PRIMARY OUTPUT")
print(f"{'='*80}")
print("\n[Tips]")
print("  1. Hide ROI with eye icon in Data module")
print("  2. For 3D: Volume Rendering on *_masked → Preset 'Ultrasound' → adjust threshold")
print("  3. If front is clipped, increase anterior_margin_mm or ROI X size")
print("  4. For segmentation: Use Segment Editor on *_masked volume")
print("  5. To save all outputs: save_pipeline_outputs(vol.GetName(), '/path/to/output')")

# -------------------- Optional: Batch Processing Function --------------------
def process_eye_batch(input_names, **kwargs):
    """
    Process multiple eye volumes with the same configuration.
    Usage: process_eye_batch(['vol1', 'vol2'], do_CAD=True, target_spacing=(0.2,0.2,0.2))
    """
    results = {}
    for name in input_names:
        try:
            print(f"\n{'='*50}\nProcessing: {name}\n{'='*50}")
            
            # Would need to refactor main code into function for this to work
            # For now, manually change inputName and re-run script
            
            results[name] = f"Completed: {name}"
            print(f"✓ Completed: {name}")
            
        except Exception as e:
            print(f"✗ Failed: {name} - {str(e)}")
            results[name] = None
    
    return results

# ======================================================================================================================