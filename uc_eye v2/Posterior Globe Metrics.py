# ===================== Posterior Globe: Geodesic Interior Patch (Fixed Area) → BFS → Metrics =====================
# Paste into Slicer's Python Interactor. Edit CONFIG, then run: outs = run_pipeline()

import numpy as np, vtk, slicer, os, math, heapq

# -------------------- CONFIG --------------------
SEGMENTATION_NODE_NAME = "GS0306_EO_OD_posterior_globe_09-05-2025"  # segmentation node in scene
SEGMENT_NAME           = "Posterior"                                 # segment to analyze

PATCH_MODE      = "geodesic"    # "geodesic" (recommended), "angle", or "plane"
APEX_MODE       = "interior"    # "interior" (deepest inside BFS) or "exterior" (deepest bulge)
TARGET_AREA_MM2 = 220.0         # <-- fixed surface patch area (mm^2) for geodesic mode
GEO_RADIUS_MM   = 10.0          # initial guess / used only if PATCH_MODE!="geodesic"
POSTERIOR_ANGLE_DEG = 46.0      # if PATCH_MODE="angle"
PLANE_D_MM      = 3.0           # if PATCH_MODE="plane"

TRIM_FRAC  = 0.18               # robust sphere fit trimming
TRIM_ITERS = 3
REFIT_ON_PATCH = True           # re-fit BFS on selected patch for stability

# Save to Desktop for convenience
DESKTOP_PATH = os.path.expanduser("~/Desktop")
os.makedirs(DESKTOP_PATH, exist_ok=True)

# -------------------- Helpers --------------------
def _export_closed_surface_poly(segmentationNode, segmentID):
    """Get closed-surface vtkPolyData for a segment across Slicer versions."""
    segmentationNode.CreateClosedSurfaceRepresentation()
    logic = slicer.modules.segmentations.logic()

    # 1) Newer API (plural)
    if hasattr(logic, "ExportSegmentsClosedSurfaceRepresentationToModels"):
        ids = vtk.vtkStringArray(); ids.InsertNextValue(segmentID)
        models = vtk.vtkCollection()
        logic.ExportSegmentsClosedSurfaceRepresentationToModels(segmentationNode, ids, models)
        m = models.GetItemAsObject(0)
        if m and m.GetPolyData() and m.GetPolyData().GetNumberOfPoints() > 0:
            return m.GetPolyData(), m

    # 2) Older API (singular)
    if hasattr(logic, "ExportSegmentsClosedSurfaceRepresentationToModel"):
        m = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "tmp_surface")
        logic.ExportSegmentsClosedSurfaceRepresentationToModel(segmentationNode, [segmentID], [m])
        if m.GetPolyData() and m.GetPolyData().GetNumberOfPoints() > 0:
            return m.GetPolyData(), m

    # 3) Fallback: read from vtkSegment
    seg     = segmentationNode.GetSegmentation()
    segment = seg.GetSegment(segmentID)
    assert segment is not None, f"Segment ID '{segmentID}' not found."
    repObj = None
    if hasattr(logic, "GetSegmentRepresentation"):
        try:
            repObj = logic.GetSegmentRepresentation(segment, "Closed surface")
        except TypeError:
            repObj = None
    if repObj is None and hasattr(segment, "GetRepresentation"):
        repObj = segment.GetRepresentation("Closed surface")
    poly = vtk.vtkPolyData.SafeDownCast(repObj)
    assert poly and poly.GetNumberOfPoints() > 0, "Could not obtain 'Closed surface' polydata."
    m = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "tmp_surface")
    copy = vtk.vtkPolyData(); copy.DeepCopy(poly)
    m.SetAndObservePolyData(copy); m.CreateDefaultDisplayNodes()
    return m.GetPolyData(), m

def _vtk_to_numpy_points(poly):
    pts = poly.GetPoints(); n = pts.GetNumberOfPoints()
    arr = np.empty((n, 3), dtype=np.float64)
    for i in range(n): pts.GetPoint(i, arr[i])
    return arr

def _robust_sphere_fit(points, trim_frac=0.15, iters=3):
    P = points.copy()
    for _ in range(iters):
        A = np.c_[2*P, np.ones(len(P))]; b = (P**2).sum(1)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        c = x[:3]; d = x[3]; R = np.sqrt((c**2).sum() + d)
        resid = np.abs(np.linalg.norm(P - c, axis=1) - R)
        keep_n = max(int((1.0 - trim_frac) * len(P)), 100)
        P = P[np.argsort(resid)[:keep_n]]
    A = np.c_[2*P, np.ones(len(P))]; b = (P**2).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:3]; d = x[3]; R = np.sqrt((c**2).sum() + d)
    return c, R

def _angle_cap_mask(points, center, apex_dir, angle_deg):
    V = points - center; V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    a = apex_dir / (np.linalg.norm(apex_dir) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(V.dot(a), -1.0, 1.0)))
    return ang <= angle_deg

def _plane_cap_mask(points, apex_point, apex_dir, d_mm):
    n = apex_dir / (np.linalg.norm(apex_dir) + 1e-9)
    o = apex_point - d_mm * n
    return (points - o).dot(n) >= 0.0

# ---- Geodesic disk on mesh (single-source Dijkstra across triangle edges) ----
def _build_edge_graph(poly, points):
    polys = poly.GetPolys(); polys.InitTraversal()
    idList = vtk.vtkIdList()
    n = points.shape[0]
    adj = [[] for _ in range(n)]
    while polys.GetNextCell(idList):
        ids = [idList.GetId(i) for i in range(idList.GetNumberOfIds())]
        for i in range(len(ids)):
            a, b = ids[i], ids[(i+1) % len(ids)]
            pa, pb = points[a], points[b]
            w = float(np.linalg.norm(pa - pb))
            adj[a].append((b, w)); adj[b].append((a, w))
    return adj

def _geodesic_distances(poly, points, seed_idx):
    adj = _build_edge_graph(poly, points)
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[seed_idx] = 0.0
    visited = np.zeros(n, dtype=bool)
    h = [(0.0, seed_idx)]
    while h:
        d,u = heapq.heappop(h)
        if visited[u]: continue
        visited[u] = True
        for v,w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist  # mm

def _estimate_patch_area(poly, keep_mask):
    pts = poly.GetPoints()
    polys = poly.GetPolys(); polys.InitTraversal()
    idList = vtk.vtkIdList()
    area = 0.0
    while polys.GetNextCell(idList):
        ids = [idList.GetId(i) for i in range(idList.GetNumberOfIds())]
        if all(keep_mask[i] for i in ids):
            p0 = np.array(pts.GetPoint(ids[0]))
            p1 = np.array(pts.GetPoint(ids[1]))
            p2 = np.array(pts.GetPoint(ids[2]))
            area += 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
    return float(area)

def _add_scalar(poly, name, values, fill_strategy="max+1"):
    # Ensure array has only finite numbers before attaching to polydata
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.all():
        if fill_strategy == "max+1" and np.any(finite):
            fill = np.max(vals[finite]) + 1.0
        else:
            fill = 0.0
        vals = np.where(finite, vals, fill)
    arr = vtk.vtkDoubleArray(); arr.SetName(name)
    for v in vals: arr.InsertNextValue(float(v))
    poly.GetPointData().AddArray(arr)

# ===================== Heatmap: Distance to BFS Center (and Residual) =====================

def add_bfs_distance_heatmaps(outs=None,
                              segmentationNodeName=SEGMENTATION_NODE_NAME,
                              segmentName=SEGMENT_NAME,
                              trim_frac=TRIM_FRAC,
                              trim_iters=TRIM_ITERS):
    """
    Create surface heatmaps for:
      - center_dist_mm : Euclidean distance from each vertex to BFS center
      - bfs_resid_mm   : distance - BFS radius (signed residual)
    Uses BFS center/radius from 'outs' if provided, otherwise refits.

    Returns:
      dict with keys: poly, modelNode, center_mm, radius_mm, dists_mm, resid_mm
    """
    # --- get segmentation + closed surface polydata ---
    segNode = slicer.util.getNode(segmentationNodeName)
    seg = segNode.GetSegmentation()
    segID = seg.GetSegmentIdBySegmentName(segmentName)
    if segID is None:
        raise ValueError(f"Segment '{segmentName}' not found in '{segmentationNodeName}'")

    poly, modelNode = _export_closed_surface_poly(segNode, segID)
    n_pts = poly.GetNumberOfPoints()
    if n_pts == 0:
        raise RuntimeError("Closed surface has 0 points; check segmentation and surface representation.")

    pts_np = _vtk_to_numpy_points(poly)

    # --- get BFS center + radius ---
    if outs is not None and "bfs_center_mm" in outs and "bfs_radius_mm" in outs:
        center = np.asarray(outs["bfs_center_mm"], dtype=float)
        radius = float(outs["bfs_radius_mm"])
        print(f"[Heatmap] Using BFS center/radius from outs: "
              f"center={center.tolist()}, R={radius:.3f} mm")
    else:
        center, radius = _robust_sphere_fit(pts_np, trim_frac=trim_frac, iters=trim_iters)
        print(f"[Heatmap] Refit BFS for heatmap: center={center.tolist()}, R={radius:.3f} mm")

    # --- compute distances + residuals ---
    dists = np.linalg.norm(pts_np - center[None, :], axis=1)
    resid = dists - radius

    # --- helper to push numpy → vtk array ---
    def _add_array(name, values):
        arr = vtk.vtkDoubleArray()
        arr.SetName(name)
        arr.SetNumberOfValues(len(values))
        for i, v in enumerate(values):
            arr.SetValue(i, float(v))
        poly.GetPointData().AddArray(arr)
        return arr

    _add_array("center_dist_mm", dists)
    _add_array("bfs_resid_mm", resid)

    # --- hook into display for visualization ---
    dispNode = modelNode.GetDisplayNode()
    if dispNode is None:
        modelNode.CreateDefaultDisplayNodes()
        dispNode = modelNode.GetDisplayNode()

    # Use residual by default (highlights bulge/flattening); distance is still available
    dispNode.SetActiveScalarName("bfs_resid_mm")
    dispNode.SetScalarVisibility(True)

    # Optional: try to use a diverging LUT if present
    colorNode = slicer.util.getFirstNodeByClassByName("vtkMRMLColorTableNode", "RedBlue")
    if colorNode:
        dispNode.SetAndObserveColorNodeID(colorNode.GetID())

    # --- quick validation printout ---
    rms_resid = math.sqrt(float(np.mean(resid**2)))
    print(f"[Heatmap] Added 'center_dist_mm' and 'bfs_resid_mm' on {n_pts} vertices.")
    print(f"          Dist range: {dists.min():.2f}–{dists.max():.2f} mm "
          f"(R≈{radius:.2f} mm); residual RMS={rms_resid:.3f} mm")

    return {
        "poly": poly,
        "modelNode": modelNode,
        "center_mm": center,
        "radius_mm": radius,
        "dists_mm": dists,
        "resid_mm": resid,
    }

# ===================== Minimal helper if you only want raw distance =====================

def add_center_distance_heatmap(poly, center, modelNode, array_name="center_dist_mm"):
    """
    Adds a scalar field 'array_name' of Euclidean distances from each point to 'center'.
    poly: vtkPolyData (closed surface)
    center: 3-element iterable in mm (RAS)
    modelNode: corresponding model node for display
    """
    pts = poly.GetPoints()
    n = pts.GetNumberOfPoints()
    if n == 0:
        raise RuntimeError("Polydata has 0 points; cannot build heatmap.")

    center = np.asarray(center, dtype=float)
    dists = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = np.array(pts.GetPoint(i), dtype=np.float64)
        dists[i] = np.linalg.norm(p - center)

    arr = vtk.vtkDoubleArray()
    arr.SetName(array_name)
    arr.SetNumberOfValues(n)
    for i, v in enumerate(dists):
        arr.SetValue(i, float(v))
    poly.GetPointData().AddArray(arr)

    dispNode = modelNode.GetDisplayNode()
    if dispNode is None:
        modelNode.CreateDefaultDisplayNodes()
        dispNode = modelNode.GetDisplayNode()
    dispNode.SetActiveScalarName(array_name)
    dispNode.SetScalarVisibility(True)

    print(f"[Heatmap] Added '{array_name}' on {n} vertices "
          f"(range {dists.min():.2f}–{dists.max():.2f} mm).")
    return dists

# -------------------- Main --------------------
def run_pipeline():
    # 1) Get the surface
    segNode = slicer.util.getNode(SEGMENTATION_NODE_NAME)
    segID   = segNode.GetSegmentation().GetSegmentIdBySegmentName(SEGMENT_NAME)
    assert segID, f"Segment '{SEGMENT_NAME}' not found in '{SEGMENTATION_NODE_NAME}'"
    poly, modelNode = _export_closed_surface_poly(segNode, segID)

    # 2) Initial robust BFS on full surface
    P = _vtk_to_numpy_points(poly)
    c0, R0 = _robust_sphere_fit(P, trim_frac=TRIM_FRAC, iters=TRIM_ITERS)

    # 3) Residuals and apex (interior vs exterior)
    dist0  = np.linalg.norm(P - c0, axis=1)
    resid0 = dist0 - R0
    apex_idx = int(np.argmin(resid0)) if APEX_MODE == "interior" else int(np.argmax(resid0))
    apex_point = P[apex_idx]
    apex_dir   = apex_point - c0

    # 4) Build the analysis patch
    if PATCH_MODE == "geodesic":
        geod = _geodesic_distances(poly, P, apex_idx)       # mm along surface
        finite = np.isfinite(geod)
        sign_mask = (resid0 <= 0.0 + 1e-6) if APEX_MODE == "interior" else (resid0 >= 0.0 - 1e-6)
        usable = finite & sign_mask

        # Upper bound for radius is the farthest usable vertex
        if not np.any(usable):
            raise RuntimeError("No usable vertices around apex (check segmentation).")
        hi = float(np.max(geod[usable]))
        # Area at upper bound
        keep_hi = (geod <= hi) & usable
        area_hi = _estimate_patch_area(poly, keep_hi)

        area_note = ""
        if TARGET_AREA_MM2 is not None:
            if area_hi < TARGET_AREA_MM2:
                # Cannot reach requested area; take the maximum available
                GEO_RADIUS_MM = hi
                keep = keep_hi
                area_note = f"(max reachable area {area_hi:.1f} < target {TARGET_AREA_MM2:.1f})"
            else:
                # Bisection to find radius that yields target area
                lo = 0.0
                for _ in range(18):  # ~1e-5 relative radius resolution
                    mid = 0.5*(lo+hi)
                    keep_test = (geod <= mid) & usable
                    area = _estimate_patch_area(poly, keep_test)
                    if area < TARGET_AREA_MM2:
                        lo = mid
                    else:
                        hi = mid
                GEO_RADIUS_MM = 0.5*(lo+hi)
                keep = (geod <= GEO_RADIUS_MM) & usable
        else:
            # No fixed area requested: fall back to fixed geodesic radius
            keep = (geod <= GEO_RADIUS_MM) & sign_mask

        patch_label = (f"geodesic area={TARGET_AREA_MM2:.1f} mm^2 (s~{GEO_RADIUS_MM:.2f} mm; {APEX_MODE})"
                      if TARGET_AREA_MM2 is not None else
                      f"geodesic {GEO_RADIUS_MM:.2f} mm ({APEX_MODE})")
                    

    elif PATCH_MODE == "angle":
        keep = _angle_cap_mask(P, c0, apex_dir, POSTERIOR_ANGLE_DEG)
        if APEX_MODE == "interior": keep = keep & (resid0 <= 0.0 + 1e-6)
        else:                        keep = keep & (resid0 >= 0.0 - 1e-6)
        patch_label = f"angle alpha={POSTERIOR_ANGLE_DEG:.1f} deg ({APEX_MODE})"


    elif PATCH_MODE == "plane":
        keep = _plane_cap_mask(P, apex_point, apex_dir, PLANE_D_MM)
        if APEX_MODE == "interior": keep = keep & (resid0 <= 0.0 + 1e-6)
        else:                        keep = keep & (resid0 >= 0.0 - 1e-6)
        patch_label = f"plane d={PLANE_D_MM:.2f} mm ({APEX_MODE})"

    else:
        raise ValueError("PATCH_MODE must be 'geodesic', 'angle', or 'plane'.")

    if np.count_nonzero(keep) < 100:
        raise RuntimeError("Patch too small; increase area/radius or relax constraints.")

    P_patch   = P[keep]
    resid_pat = resid0[keep]

    # 5) Optionally re-fit BFS on the patch
    if REFIT_ON_PATCH:
        c, R = _robust_sphere_fit(P_patch, trim_frac=min(TRIM_FRAC, 0.12), iters=TRIM_ITERS)
        dist_pat = np.linalg.norm(P_patch - c, axis=1)
        resid_pat = dist_pat - R
    else:
        c, R = c0, R0

    # 6) Metrics on the patch
    if APEX_MODE == "interior":
        apex_depth_mm = float(-np.min(resid_pat))      # interior depth as positive number
    else:
        apex_depth_mm = float(np.max(resid_pat))       # exterior bulge depth
    rms_resid_mm  = float(np.sqrt(np.mean(resid_pat**2)))
    patch_area_mm2 = _estimate_patch_area(poly, keep)

    # 7) QA outputs
    _add_scalar(poly, "residual_mm", np.linalg.norm(P - c, axis=1) - R)
    _add_scalar(poly, "patch_keep", keep.astype(float))
    if PATCH_MODE == "geodesic":
        _add_scalar(poly, "geodist_mm", geod)  # sanitized in _add_scalar

    # Save VTK + CSV to Desktop
    writer = vtk.vtkPolyDataWriter()
    vtk_path = os.path.join(DESKTOP_PATH, "posterior_surface_with_residuals.vtk")
    writer.SetFileName(vtk_path)
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    writer.Write()

    csv_path = os.path.join(DESKTOP_PATH, "posterior_BFS_metrics.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Patch","ApexMode","PatchArea_mm2","GEO_radius_mm","Angle_deg","Plane_d_mm",
                    "BFS_radius_mm","Apex_depth_mm","RMS_residual_mm",
                    "Apex_x_mm","Apex_y_mm","Apex_z_mm"])
        w.writerow([patch_label, APEX_MODE, patch_area_mm2,
                    (GEO_RADIUS_MM if PATCH_MODE=='geodesic' else ""),
                    (POSTERIOR_ANGLE_DEG if PATCH_MODE=='angle' else ""),
                    (PLANE_D_MM if PATCH_MODE=='plane' else ""),
                    float(R), apex_depth_mm, rms_resid_mm,
                    float(apex_point[0]), float(apex_point[1]), float(apex_point[2])])

    print("\n=== Posterior BFS metrics (patch) ===")
    print(f"Patch      : {patch_label} | Area ~ {patch_area_mm2:.1f} mm^2")
    if PATCH_MODE == "geodesic" and TARGET_AREA_MM2 is not None:
        if 'area_note' in locals() and area_note:
            print("Note       :", area_note)
    if PATCH_MODE == "angle":
        d_equiv = R * (1.0 - math.cos(math.radians(POSTERIOR_ANGLE_DEG)))
        print(f"Angle-cap  : alpha = {POSTERIOR_ANGLE_DEG:.1f} deg  (~d {d_equiv:.2f} mm for R={R:.2f})")
    if PATCH_MODE == "plane":
        alpha_equiv = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - PLANE_D_MM / R))))
        print(f"Plane-cap  : d = {PLANE_D_MM:.2f} mm  (~alpha {alpha_equiv:.1f} deg for R={R:.2f})")
    print(f"BFS radius : {R:.3f} mm")
    print(f"Apex depth : {apex_depth_mm:.3f} mm ({'interior cup' if APEX_MODE=='interior' else 'exterior bulge'})")
    print(f"RMS resid  : {rms_resid_mm:.3f} mm")
    print("VTK (QA)   :", vtk_path)
    print("CSV        :", csv_path)

    # ---- return dictionary ----
    return dict(
        patch=patch_label,
        apex_mode=APEX_MODE,
        patch_area_mm2=patch_area_mm2,
        R_mm=float(R),
        apex_depth_mm=apex_depth_mm,
        rms_resid_mm=rms_resid_mm,
        apex_point_ras=apex_point.tolist(),
        vtk_path=vtk_path,
        csv_path=csv_path
    )


# Example:
# outs = run_pipeline()
# outs
