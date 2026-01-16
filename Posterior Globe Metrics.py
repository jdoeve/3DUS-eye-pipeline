# ===================== Posterior Globe: Geodesic Interior Patch (Fixed Area) → BFS → Metrics =====================
# Paste into Slicer's Python Interactor. Edit CONFIG, then run: outs = run_pipeline()

import numpy as np, vtk, slicer, os, math, heapq

# -------------------- CONFIG --------------------
SEGMENTATION_NODE_NAME = "Segmentation"  # segmentation node in scene
SEGMENT_NAME           = "Posterior"                                 # segment to analyze

PATCH_MODE      = "geodesic"    # "geodesic" (recommended), "angle", or "plane"
APEX_MODE       = "interior"    # "interior" (deepest inside BFS) or "exterior" (deepest bulge)
TARGET_AREA_MM2 = 250.0         # <-- fixed surface patch area (mm^2) for geodesic mode
GEO_RADIUS_MM   = 10.0          # initial guess / used only if PATCH_MODE!="geodesic"
POSTERIOR_ANGLE_DEG = 45.0      # if PATCH_MODE="angle"
PLANE_D_MM      = 3.0           # if PATCH_MODE="plane"

# --- Robust Apex Options (recommended ON) ---

# 0) Manual override (best fail-safe):
# Create a Markups fiducial named MANUAL_APEX_FIDUCIAL_NAME (1 point).
# Script will snap it to nearest surface vertex and use that as apex.
USE_MANUAL_APEX_FIDUCIAL   = True
MANUAL_APEX_FIDUCIAL_NAME  = "PosteriorApex_Manual"
USE_BOUNDARY_DISTANCE_APEX = True

# 1) Posterior prior direction (surface-only). Helps prevent picking seams/rims.
USE_POSTERIOR_PRIOR        = True
POSTERIOR_PRIOR_CONE_DEG   = 35.0   # coarse cone around posterior prior

# 2) Residual smoothing on mesh (median over 1-ring neighbors).
USE_RESIDUAL_MEDIAN_SMOOTH = True

# 3) Optional second refinement cone around the coarse apex (you already have this)
# POSTERIOR_CONE_DEG is reused for the final refine stage

# ----- Variance reduction / standardization options -----

# 1) Apex refinement: only search for apex within this cone around initial apex direction
POSTERIOR_CONE_DEG       = 20.0   # degrees

# 2) Patch limitation: only keep points within this Euclidean radius of the apex (mm)
PATCH_MAX_RADIUS_MM      = 10.0   # posterior cap radius in mm (e.g. 8–12)

# 3) Seam exclusion: drop vertices whose |H| is in the top X% for the patch
SEAM_EXCLUDE_PERCENT     = 2.0    # e.g. remove top 5% largest |H| as seam/edge outliers

# 4) Local quadric: stronger residual trimming to reject local artifacts
LOCAL_QUADRIC_TRIM_FRAC  = 0.35   # keep best 65% points in the neighborhood (was ~0.2 before)

# 5) Surface smoothing before curvature
ENABLE_SURFACE_SMOOTHING = True
SMOOTH_ITERATIONS        = 15     # light smoothing
SMOOTH_PASSBAND          = 0.1    # for windowed sinc filter

# --- Curvature analysis options ---
LOCAL_QUADRIC_RADIUS_MM = 4.0   # neighborhood radius for quadric fit around apex
CURVATURE_TRIM_PERCENT  = 10.0   # trim top X% |H| values to downweight seam spikes

TRIM_FRAC  = 0.10               # robust sphere fit trimming
TRIM_ITERS = 3
REFIT_ON_PATCH = True           # re-fit BFS on selected patch for stability
CREATE_APEX_FIDUCIAL = True     # toggle creating/updating a Markups fiducial at the apex
APEX_FIDUCIAL_NAME   = "PosteriorApex"

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

# ===================== Curvature Helpers: Local Quadric + Discrete Mean =====================

def _build_local_frame(apex_point, center_point):
    """
    Build an orthonormal frame (t1, t2, n) at the apex.
    n is roughly the surface normal (center->apex), t1,t2 span tangent plane.
    """
    apex_point = np.asarray(apex_point, dtype=float)
    center_point = np.asarray(center_point, dtype=float)

    n = apex_point - center_point
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        # degenerate: pick arbitrary normal
        n = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        n = n / n_norm

    # pick reference not parallel to n
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)

    t1 = ref - np.dot(ref, n) * n
    t1_norm = np.linalg.norm(t1)
    if t1_norm < 1e-9:
        t1 = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        t1 = t1 / t1_norm

    t2 = np.cross(n, t1)
    t2_norm = np.linalg.norm(t2)
    if t2_norm < 1e-9:
        t2 = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        t2 = t2 / t2_norm

    return t1, t2, n

def _smooth_scalar_on_mesh(poly, values, n_iter=5, alpha=0.5):
    """
    Simple Laplacian smoothing for a per-vertex scalar field.

    poly   : vtkPolyData (used for connectivity)
    values : 1D NumPy array of length N (scalar at each point)
    n_iter : number of smoothing iterations
    alpha  : blending factor (0 < alpha < 1);
             alpha=0.5 means average half with neighbors each step.

    Returns smoothed values as NumPy array (same length).
    """
    vals = np.asarray(values, dtype=float)
    n = vals.shape[0]
    if n == 0:
        return vals

    # Build adjacency once
    points = _vtk_to_numpy_points(poly)
    adj = _build_edge_graph(poly, points)  # already defined in your script

    cur = vals.copy()
    for _ in range(n_iter):
        new = cur.copy()
        for i in range(n):
            neigh = adj[i]
            if not neigh:
                continue
            # neighbors indices only
            idxs = [j for j,_ in neigh]
            neigh_vals = cur[idxs]
            mean_neigh = np.mean(neigh_vals)
            new[i] = (1.0 - alpha) * cur[i] + alpha * mean_neigh
        cur = new

    return cur

def _fit_local_quadric_at_apex(P_patch, apex_point, center_point,
                               local_radius_mm=LOCAL_QUADRIC_RADIUS_MM,
                               outlier_trim_frac=0.20):
    """
    Fit z = a x^2 + b y^2 + c x y + d x + e y + f in a local neighborhood
    around the apex, using a tangent-plane coordinate system.

    Returns a dict with principal curvatures, mean/gaussian curvature, etc.
    Outlier trimming makes it robust to local artifacts (e.g., seam intruding
    into the neighborhood).
    """
    P_patch = np.asarray(P_patch, dtype=float)
    apex_point = np.asarray(apex_point, dtype=float)
    center_point = np.asarray(center_point, dtype=float)

    # Local frame
    t1, t2, n = _build_local_frame(apex_point, center_point)

    # Express patch points in local coordinates relative to apex
    V = P_patch - apex_point[None, :]
    X = V.dot(t1)
    Y = V.dot(t2)
    Z = V.dot(n)

    # Local neighborhood around apex (limit influence of seam)
    r = np.sqrt(X**2 + Y**2)
    mask = r <= local_radius_mm
    if np.count_nonzero(mask) < 30:
        # fallback: take nearest 30 points if patch is sparse
        idx = np.argsort(r)
        mask = np.zeros_like(r, dtype=bool)
        mask[idx[:30]] = True

    Xn = X[mask]; Yn = Y[mask]; Zn = Z[mask]

    # Design matrix for quadratic fit
    A = np.c_[Xn**2, Yn**2, Xn*Yn, Xn, Yn, np.ones_like(Xn)]

    # First pass fit
    coeffs, *_ = np.linalg.lstsq(A, Zn, rcond=None)
    pred = A.dot(coeffs)
    resid = Zn - pred

    # Robust trimming (downweight any weird local seam geometry)
    if 0.0 < outlier_trim_frac < 0.5 and len(Zn) > 20:
        keep_n = max(int((1.0 - outlier_trim_frac) * len(Zn)), 20)
        keep_idx = np.argsort(np.abs(resid))[:keep_n]
        A2 = A[keep_idx, :]
        Z2 = Zn[keep_idx]
        coeffs, *_ = np.linalg.lstsq(A2, Z2, rcond=None)

    a, b, c, d, e, f = coeffs

    # Second derivatives at origin (apex)
    z_xx = 2.0 * a
    z_yy = 2.0 * b
    z_xy = c

    Hmat = np.array([[z_xx, z_xy],
                     [z_xy, z_yy]], dtype=float)
    evals, _ = np.linalg.eig(Hmat)
    # Sort so k1 is the larger magnitude
    k1, k2 = np.sort(evals)[::-1]

    H_mean = 0.5 * (k1 + k2)
    K_gauss = (k1 * k2)

    R1 = (1.0 / k1) if abs(k1) > 1e-9 else np.inf
    R2 = (1.0 / k2) if abs(k2) > 1e-9 else np.inf
    R_equiv = (1.0 / H_mean) if abs(H_mean) > 1e-9 else np.inf

    anisotropy = (abs(k1 - k2) / (abs(k1) + abs(k2))) if (abs(k1) + abs(k2)) > 1e-9 else 0.0

    return dict(
        quad_k1_apex_invmm=float(k1),
        quad_k2_apex_invmm=float(k2),
        quad_H_apex_invmm=float(H_mean),
        quad_K_apex_invmm2=float(K_gauss),
        quad_R1_apex_mm=float(R1),
        quad_R2_apex_mm=float(R2),
        quad_R_equiv_apex_mm=float(R_equiv),
        quad_anisotropy_apex=float(anisotropy)
    )

def _compute_discrete_mean_curvature(poly):
    """
    Run vtkCurvatures (mean curvature) on the full surface and return
    a NumPy array of H (1/mm) for each vertex. Also stores a scalar
    array 'mean_curvature_invmm' on poly for QA visualization.

    We rely on robust summarization (median + trimmed mean) later
    to suppress seam-driven spikes.
    """
    curv = vtk.vtkCurvatures()
    curv.SetInputData(poly)
    curv.SetCurvatureTypeToMean()
    curv.Update()
    out = curv.GetOutput()

    npts = out.GetNumberOfPoints()
    arr = out.GetPointData().GetScalars()
    if arr is None or npts == 0:
        H = np.zeros(npts, dtype=float)
    else:
        H = np.array([arr.GetTuple1(i) for i in range(npts)], dtype=float)

    # Attach to original poly for QA
    _add_scalar(poly, "mean_curvature_invmm", H, fill_strategy="max+1")

    H_abs = np.abs(H)
    _add_scalar(poly, "abs_mean_curvature_invmm", H_abs, fill_strategy="max+1")

    # Optional: smoothed absolute curvature for nicer heatmaps (visualization only)
    H_abs_smooth = _smooth_scalar_on_mesh(poly, H_abs, n_iter=5, alpha=0.5)
    _add_scalar(poly, "abs_mean_curvature_smooth_invmm", H_abs_smooth, fill_strategy="max+1")

    return H


def _create_or_update_apex_fiducial(point_ras, name):
    """Create or update a fiducial node at the apex; returns the node or None."""
    node = None
    try:
        candidate = slicer.util.getNode(name)
        if candidate.GetClassName() == "vtkMRMLMarkupsFiducialNode":
            node = candidate
    except Exception:
        node = None

    if node is None:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        node.CreateDefaultDisplayNodes()
    node.RemoveAllControlPoints()
    node.AddControlPoint(vtk.vtkVector3d(float(point_ras[0]), float(point_ras[1]), float(point_ras[2])))
    node.SetLocked(True)  # avoid accidental edits
    return node

def _get_fiducial_point_world(name):
    """Return first control point world coords from a Markups fiducial node, or None."""
    try:
        n = slicer.util.getNode(name)
    except Exception:
        return None
    if n is None or n.GetClassName() != "vtkMRMLMarkupsFiducialNode":
        return None
    if n.GetNumberOfControlPoints() < 1:
        return None
    p = [0.0, 0.0, 0.0]
    n.GetNthControlPointPositionWorld(0, p)
    return np.array(p, dtype=float)

def _nearest_vertex_index(P, point):
    d = np.linalg.norm(P - point[None, :], axis=1)
    return int(np.argmin(d))

def _pca_axis(P):
    """Return first principal component unit vector from points P (Nx3)."""
    X = P - np.mean(P, axis=0, keepdims=True)
    C = (X.T @ X) / max(len(P) - 1, 1)
    w, V = np.linalg.eigh(C)
    a = V[:, np.argmax(w)]
    a = a / (np.linalg.norm(a) + 1e-12)
    return a

def _choose_posterior_prior_dir(P, c, resid, cone_deg=35.0, apex_mode="interior"):
    """
    Surface-only posterior prior:
    - Use PCA main axis (±a)
    - Choose the sign whose cone contains 'deeper' residuals for interior (or higher for exterior).
    """
    a = _pca_axis(P)
    c = np.asarray(c, dtype=float)

    # unit directions from center
    V = P - c[None, :]
    Vn = np.linalg.norm(V, axis=1)
    Vn[Vn < 1e-9] = 1e-9
    U = V / Vn[:, None]

    def score_for_dir(d):
        d = d / (np.linalg.norm(d) + 1e-12)
        cos_thresh = np.cos(np.deg2rad(cone_deg))
        m = (U @ d) >= cos_thresh
        if not np.any(m):
            return np.inf if apex_mode == "interior" else -np.inf
        r = resid[m]
        # robust statistic (median)
        return float(np.median(r))

    s_pos = score_for_dir(a)
    s_neg = score_for_dir(-a)

    if apex_mode == "interior":
        # want more negative median residual (deeper cup)
        return a if s_pos < s_neg else -a
    else:
        # want more positive median residual (bigger bulge)
        return a if s_pos > s_neg else -a

def _median_smooth_scalar_on_mesh(poly, values):
    """
    1-ring median smoothing of a scalar defined per-vertex.
    Excellent for killing seam spikes that hijack argmin/argmax.
    """
    P = _vtk_to_numpy_points(poly)
    adj = _build_edge_graph(poly, P)  # list of (neighbor, weight)
    vals = np.asarray(values, dtype=float)
    out = np.empty_like(vals)
    for i in range(len(adj)):
        nbrs = [i] + [j for (j, _) in adj[i]]
        out[i] = float(np.median(vals[nbrs]))
    return out

def _pick_apex_with_prior(P, c, resid, apex_mode, prior_dir, prior_cone_deg):
    """Pick apex index within a cone around prior_dir using resid (already smoothed if desired)."""
    c = np.asarray(c, dtype=float)
    prior_dir = np.asarray(prior_dir, dtype=float)
    prior_dir = prior_dir / (np.linalg.norm(prior_dir) + 1e-12)

    V = P - c[None, :]
    Vn = np.linalg.norm(V, axis=1)
    Vn[Vn < 1e-9] = 1e-9
    U = V / Vn[:, None]

    cos_thresh = np.cos(np.deg2rad(prior_cone_deg))
    m = (U @ prior_dir) >= cos_thresh
    if not np.any(m):
        m = np.ones(P.shape[0], dtype=bool)

    idxs = np.arange(P.shape[0])[m]
    r = resid[m]

    if apex_mode == "interior":
        best = int(idxs[np.argmin(r)])
    else:
        best = int(idxs[np.argmax(r)])
    return best

def _boundary_vertex_mask(poly):
    """Return boolean mask of vertices that lie on the mesh boundary (edges used by only 1 face)."""
    n = poly.GetNumberOfPoints()
    boundary = np.zeros(n, dtype=bool)

    edgeCount = {}
    polys = poly.GetPolys(); polys.InitTraversal()
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        ids = [idList.GetId(i) for i in range(idList.GetNumberOfIds())]
        for i in range(len(ids)):
            a = ids[i]
            b = ids[(i+1) % len(ids)]
            if a > b: a, b = b, a
            edgeCount[(a, b)] = edgeCount.get((a, b), 0) + 1

    for (a, b), cnt in edgeCount.items():
        if cnt == 1:
            boundary[a] = True
            boundary[b] = True
    return boundary

def _multi_source_geodesic(poly, points, seed_mask):
    """Dijkstra from many seeds at once. seed_mask True means distance=0 start."""
    adj = _build_edge_graph(poly, points)
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float64)
    h = []
    for i, s in enumerate(seed_mask):
        if s:
            dist[i] = 0.0
            h.append((0.0, i))
    heapq.heapify(h)
    visited = np.zeros(n, dtype=bool)

    while h:
        d,u = heapq.heappop(h)
        if visited[u]: 
            continue
        visited[u] = True
        for v,w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist

def _pseudo_boundary_from_radius(P, c, top_percent=5.0):
    """Return seed mask selecting the outermost top_percent of ||P-c|| as a pseudo-boundary."""
    r = np.linalg.norm(P - c[None, :], axis=1)
    thr = np.percentile(r, 100.0 - top_percent)
    return r >= thr


# ===================== Variance Reduction Helpers =====================

def _smooth_poly(poly):
    """
    Optional light smoothing of the closed surface before curvature computations.
    Uses vtkWindowedSincPolyDataFilter, which preserves overall shape better
    than naive Laplacian smoothing.
    """
    if not ENABLE_SURFACE_SMOOTHING:
        return poly

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetNumberOfIterations(SMOOTH_ITERATIONS)
    smoother.SetPassBand(SMOOTH_PASSBAND)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def _refine_apex_in_cone(P, c, apex_point, resid_all, apex_mode, cone_deg=POSTERIOR_CONE_DEG):
    """
    Refine the apex by restricting the search for min/max residual to points whose
    direction from the center lies within a cone around the initial apex direction.
    This stabilizes the apex location and avoids seam-driven extrema.

    P          : Nx3 points (surface)
    c          : 3-vector BFS center
    apex_point : initial apex (3-vector)
    resid_all  : residuals for all points (r - R)
    apex_mode  : "interior" or "exterior"
    """
    P = np.asarray(P, dtype=float)
    c = np.asarray(c, dtype=float)
    apex_point = np.asarray(apex_point, dtype=float)
    v0 = apex_point - c
    v0_norm = np.linalg.norm(v0)
    if v0_norm < 1e-9:
        # fall back to centroid direction if degenerate
        v0 = np.mean(P, axis=0) - c
        v0_norm = np.linalg.norm(v0)
    v0 = v0 / (v0_norm + 1e-12)

    V = P - c[None, :]
    V_norm = np.linalg.norm(V, axis=1)
    V_norm[V_norm < 1e-9] = 1e-9
    U = V / V_norm[:, None]

    cos_angle = U.dot(v0)
    cos_thresh = np.cos(np.deg2rad(cone_deg))
    mask = cos_angle >= cos_thresh

    # If cone is empty (shouldn't happen), fall back to full set
    if not np.any(mask):
        mask = np.ones_like(cos_angle, dtype=bool)

    resid_sub = resid_all[mask]
    idx_sub = np.arange(P.shape[0])[mask]

    if apex_mode == "interior":
        # interior cup: apex is most negative residual
        best_local = np.argmin(resid_sub)
    else:
        # exterior bulge: apex is most positive residual
        best_local = np.argmax(resid_sub)

    best_idx = int(idx_sub[best_local])
    return P[best_idx], best_idx


def _limit_patch_radius(P, apex_point, keep_mask, max_radius_mm=PATCH_MAX_RADIUS_MM):
    """
    Restrict the posterior patch to points within a fixed Euclidean radius around the apex.
    This standardizes patch size across eyes and reduces variance.
    """
    P = np.asarray(P, dtype=float)
    apex_point = np.asarray(apex_point, dtype=float)
    d = np.linalg.norm(P - apex_point[None, :], axis=1)
    radius_mask = d <= max_radius_mm
    return keep_mask & radius_mask

# -------------------- Main --------------------
def run_pipeline():
    # 1) Get the surface
    segNode = slicer.util.getNode(SEGMENTATION_NODE_NAME)
    segID   = segNode.GetSegmentation().GetSegmentIdBySegmentName(SEGMENT_NAME)
    assert segID, f"Segment '{SEGMENT_NAME}' not found in '{SEGMENTATION_NODE_NAME}'"
    poly, modelNode = _export_closed_surface_poly(segNode, segID)
    poly = _smooth_poly(poly)

    # 2) Initial robust BFS on full surface
    P = _vtk_to_numpy_points(poly)
    c0, R0 = _robust_sphere_fit(P, trim_frac=TRIM_FRAC, iters=TRIM_ITERS)

    # 3) Residuals (BFS)
    dist0  = np.linalg.norm(P - c0, axis=1)
    resid0 = dist0 - R0

    # -------------------- Apex selection (priority order) --------------------
    apex_fiducial = None  # avoids "referenced before assignment"

    # (1) Manual apex override (highest priority)
    manual_world = _get_fiducial_point_world(MANUAL_APEX_FIDUCIAL_NAME) if USE_MANUAL_APEX_FIDUCIAL else None
    if manual_world is not None:
        apex_idx = _nearest_vertex_index(P, manual_world)
        apex_reason = "manual_fiducial"

    # (2) Boundary-distance apex (geodesic farthest from boundary)
    elif USE_BOUNDARY_DISTANCE_APEX:
        bmask = _boundary_vertex_mask(poly)

        # If the surface is closed, there is no true boundary → use a pseudo-boundary ring
        if np.count_nonzero(bmask) < 10:
            bmask = _pseudo_boundary_from_radius(P, c0, top_percent=5.0)

        db = _multi_source_geodesic(poly, P, bmask)  # distance-to-boundary (mm)

        # Restrict to correct side of BFS if possible
        if APEX_MODE == "interior":
            valid = np.isfinite(db) & (resid0 <= 0.0 + 1e-6)
        else:
            valid = np.isfinite(db) & (resid0 >= 0.0 - 1e-6)

        if np.count_nonzero(valid) < 50:
            valid = np.isfinite(db)

        idxs = np.where(valid)[0]
        if idxs.size == 0:
            raise RuntimeError("Boundary-distance apex failed: no finite distance-to-boundary values.")

        apex_idx = int(idxs[np.argmax(db[idxs])])
        apex_reason = "boundary_distance"

    # (3) Fallback: posterior prior + median-smoothed residual minimum/maximum
    else:
        # choose posterior hemisphere using PCA prior (or residual-extremum direction if disabled)
        if USE_POSTERIOR_PRIOR:
            prior_dir = _choose_posterior_prior_dir(
                P=P, c=c0, resid=resid0, cone_deg=POSTERIOR_PRIOR_CONE_DEG, apex_mode=APEX_MODE
            )
        else:
            tmp_idx = int(np.argmin(resid0)) if APEX_MODE == "interior" else int(np.argmax(resid0))
            prior_dir = (P[tmp_idx] - c0)

        resid_for_pick = _median_smooth_scalar_on_mesh(poly, resid0) if USE_RESIDUAL_MEDIAN_SMOOTH else resid0

        apex_idx = _pick_apex_with_prior(
            P=P, c=c0, resid=resid_for_pick, apex_mode=APEX_MODE,
            prior_dir=prior_dir, prior_cone_deg=POSTERIOR_PRIOR_CONE_DEG
        )
        apex_reason = "prior_residual"

    # Initial apex point + direction
    apex_point = P[apex_idx]
    apex_dir   = apex_point - c0

    # -------------------- Optional refinement (narrow cone) --------------------
    # Refine using raw residuals (not smoothed) within a tighter cone around the initial apex direction
    apex_point, apex_idx = _refine_apex_in_cone(
        P=P,
        c=c0,
        apex_point=apex_point,
        resid_all=resid0,
        apex_mode=APEX_MODE,
        cone_deg=POSTERIOR_CONE_DEG
    )
    apex_dir = apex_point - c0

    # Create/update fiducial at FINAL apex
    if CREATE_APEX_FIDUCIAL:
        apex_fiducial = _create_or_update_apex_fiducial(apex_point, APEX_FIDUCIAL_NAME)

    # (Optional) Debug prints
    print(f"[Apex] method={apex_reason}  idx={apex_idx}  resid0={resid0[apex_idx]:.3f} mm")
    if USE_BOUNDARY_DISTANCE_APEX and 'db' in locals():
        print(f"[Apex] db (dist-to-boundary)={db[apex_idx]:.3f} mm")

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
    
    # --- Limit posterior patch to fixed radius around apex (variance reduction step 2) ---
    keep = _limit_patch_radius(P, apex_point, keep, max_radius_mm=PATCH_MAX_RADIUS_MM)

    if np.count_nonzero(keep) < 100:
        raise RuntimeError("Patch too small after radius limit; increase PATCH_MAX_RADIUS_MM or relax constraints.")

    # Updated patch points
    P_patch = P[keep]

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

    # 6a) Local quadric analysis at apex (robust, local neighborhood)
    quad_metrics = _fit_local_quadric_at_apex(
        P_patch,
        apex_point,
        c,
        local_radius_mm=LOCAL_QUADRIC_RADIUS_MM,
        outlier_trim_frac=LOCAL_QUADRIC_TRIM_FRAC     # <-- stronger trimming (step 4)
    )

    # 6b) Discrete mean curvature on the full surface, summarized on patch
    H_all = _compute_discrete_mean_curvature(poly)   # one H value per vertex

    # 3a) Seam exclusion: drop global outliers likely due to the reconstruction seam
    H_abs_all = np.abs(H_all[np.isfinite(H_all)])
    if H_abs_all.size > 0 and SEAM_EXCLUDE_PERCENT > 0.0:
        seam_cutoff = np.percentile(H_abs_all, 100.0 - SEAM_EXCLUDE_PERCENT)
        seam_ok = np.abs(H_all) <= seam_cutoff
    else:
        seam_ok = np.isfinite(H_all)

    # Only keep patch points that are also not seam outliers
    keep_curv = keep & seam_ok

    H_patch = H_all[keep_curv]
    H_patch = H_patch[np.isfinite(H_patch)]

    if H_patch.size > 0:
        H_abs = np.abs(H_patch)
        # Existing intra-patch trimming to guard against any remaining spikes
        cutoff = np.percentile(H_abs, 100.0 - CURVATURE_TRIM_PERCENT)
        trim_mask = H_abs <= cutoff
        H_trim = H_patch[trim_mask]
        if H_trim.size == 0:
            H_trim = H_patch  # fallback if everything was trimmed

        disc_H_median = float(np.median(H_trim))
        disc_H_abs_median = float(np.median(np.abs(H_trim)))
        disc_H_trimmed_mean_abs = float(np.mean(np.abs(H_trim)))
        disc_R_equiv_patch = (1.0 / disc_H_median) if abs(disc_H_median) > 1e-9 else np.inf
    else:
        disc_H_median = 0.0
        disc_H_abs_median = 0.0
        disc_H_trimmed_mean_abs = 0.0
        disc_R_equiv_patch = np.inf

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
        w.writerow([
            "Patch","ApexMode","PatchArea_mm2","GEO_radius_mm","Angle_deg","Plane_d_mm",
            "BFS_radius_mm","Apex_depth_mm","RMS_residual_mm",
            "Quad_R_equiv_apex_mm","Quad_anisotropy_apex",
            "Disc_R_equiv_patch_mm","Disc_H_trimmed_mean_abs_invmm",
            "Apex_x_mm","Apex_y_mm","Apex_z_mm"
        ])
        w.writerow([
            patch_label, APEX_MODE, patch_area_mm2,
            (GEO_RADIUS_MM if PATCH_MODE=='geodesic' else ""),
            (POSTERIOR_ANGLE_DEG if PATCH_MODE=='angle' else ""),
            (PLANE_D_MM if PATCH_MODE=='plane' else ""),
            float(R), apex_depth_mm, rms_resid_mm,
            quad_metrics["quad_R_equiv_apex_mm"],
            quad_metrics["quad_anisotropy_apex"],
            disc_R_equiv_patch,
            disc_H_trimmed_mean_abs,
            float(apex_point[0]), float(apex_point[1]), float(apex_point[2])
        ])

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
    print(f"Quad R_eq  : {quad_metrics['quad_R_equiv_apex_mm']:.3f} mm (local apex radius)")
    print(f"Quad aniso : {quad_metrics['quad_anisotropy_apex']:.3f} (0=spherical, 1=highly toric)")
    print(f"Disc R_eq  : {disc_R_equiv_patch:.3f} mm (patch-wide curvature, seam-trimmed)")
    print(f"Disc |H|   : {disc_H_trimmed_mean_abs:.4f} 1/mm (trimmed mean |H| on patch)")
    print("VTK (QA)   :", vtk_path)
    print("CSV        :", csv_path)

    if apex_fiducial:
        print("Fiducial   :", apex_fiducial.GetName(), "(locked)")

    # ---- return dictionary ----
    return dict(
        patch=patch_label,
        apex_mode=APEX_MODE,
        patch_area_mm2=patch_area_mm2,
        R_mm=float(R),
        apex_depth_mm=apex_depth_mm,
        rms_resid_mm=rms_resid_mm,
        quad_metrics=quad_metrics,
        disc_R_equiv_patch_mm=disc_R_equiv_patch,
        disc_H_trimmed_mean_abs_invmm=disc_H_trimmed_mean_abs,
        apex_point_ras=apex_point.tolist(),
        vtk_path=vtk_path,
        csv_path=csv_path
    )



# Example:
# outs = run_pipeline()
# outs
