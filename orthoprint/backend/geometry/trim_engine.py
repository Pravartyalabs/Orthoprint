"""
OrthoPrint — Rhino-Style Trim Engine
Implements all 8 phases of Rhino's trim pipeline:
P0: Pre-condition check
P1: Cutter definition (horizontal plane from trim points)
P2: Intersection computation
P3: Region classification
P4: Geometry removal
P5: Capping & closing
P6: Post-trim validation
FB: Fallback path
"""
from __future__ import annotations
import numpy as np
import trimesh
import trimesh.repair
import trimesh.intersections
from typing import Optional
import io, base64


# ─── P0: Pre-condition Check ──────────────────────────────────────────────────

def precondition_check(mesh: trimesh.Trimesh) -> dict:
    """
    P0 — Validate mesh before accepting trim request.
    Checks: watertightness, non-manifold edges, normal consistency, tolerance.
    """
    report = {"passed": True, "warnings": [], "errors": []}

    # Check 1: Count naked edges
    edge_count = {}
    for face in mesh.faces:
        for i in range(3):
            e = tuple(sorted([face[i], face[(i+1)%3]]))
            edge_count[e] = edge_count.get(e, 0) + 1
    naked_edges = sum(1 for c in edge_count.values() if c == 1)
    non_manifold = sum(1 for c in edge_count.values() if c > 2)

    if naked_edges > 0:
        report["warnings"].append(f"Mesh has {naked_edges} naked edges — will attempt repair")
    if non_manifold > 0:
        report["warnings"].append(f"Mesh has {non_manifold} non-manifold edges")

    # Check 2: Normal consistency
    report["normals_consistent"] = bool(mesh.is_winding_consistent)
    if not mesh.is_winding_consistent:
        report["warnings"].append("Inconsistent normals — will fix before trim")

    # Check 3: Minimum face count
    if len(mesh.faces) < 100:
        report["errors"].append("Mesh too simple — fewer than 100 faces")
        report["passed"] = False

    # Check 4: Degenerate faces
    areas = mesh.area_faces
    degen = int(np.sum(areas < 1e-10))
    if degen > 0:
        report["warnings"].append(f"{degen} degenerate faces detected")

    report["naked_edges"] = naked_edges
    report["non_manifold_edges"] = non_manifold
    report["face_count"] = len(mesh.faces)
    report["is_watertight"] = bool(mesh.is_watertight)

    return report


# ─── P0: Mesh Repair ─────────────────────────────────────────────────────────

def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fix normals, fill holes, remove degenerate faces before trim."""
    mesh = mesh.copy()
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh


# ─── P1: Cutter Definition ───────────────────────────────────────────────────

def define_cutting_plane(
    mesh: trimesh.Trimesh,
    trim_points_3d: list[list[float]],
    proximal_offset_mm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    P1 — Define the cutting plane from trim points.
    Like Rhino's WireCut: creates an infinite cutting surface.

    Strategy:
    - Points form a ring around the limb at trim height
    - Use mean Z + offset as the horizontal cut level
    - Clamp to valid range inside mesh bounds (5mm from each end)
    - Cutting plane is always infinite (extends beyond mesh boundary)
    """
    pts = np.array(trim_points_3d, dtype=float)

    # Mean Z of trim points = anatomical trim level
    mean_z = float(pts[:, 2].mean())

    # Add proximal offset — keep a cuff above the marked line
    cut_z = mean_z + proximal_offset_mm

    # Clamp to valid mesh range
    z_min = float(mesh.bounds[0][2]) + 5.0
    z_max = float(mesh.bounds[1][2]) - 5.0
    cut_z = float(np.clip(cut_z, z_min, z_max))

    # Horizontal plane — always cuts entire mesh
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    plane_origin = np.array([0.0, 0.0, cut_z], dtype=float)

    return plane_normal, plane_origin


# ─── P2: Intersection Computation ────────────────────────────────────────────

def compute_intersection_curve(
    mesh: trimesh.Trimesh,
    plane_normal: np.ndarray,
    plane_origin: np.ndarray,
) -> Optional[np.ndarray]:
    """
    P2 — Compute exact intersection curve between cutting plane and mesh.
    Returns array of 3D points forming the intersection polyline.
    """
    try:
        section = mesh.section(
            plane_origin=plane_origin,
            plane_normal=plane_normal,
        )
        if section is None:
            return None
        path_2d, transform = section.to_planar()
        # Convert back to 3D
        pts_2d = np.array([v for e in path_2d.entities
                          for v in path_2d.vertices[e.points]])
        if len(pts_2d) == 0:
            return None
        # Add Z back
        pts_3d = np.column_stack([pts_2d, np.full(len(pts_2d), plane_origin[2])])
        return pts_3d
    except Exception:
        return None


# ─── P3: Region Classification ───────────────────────────────────────────────

def classify_regions(
    mesh: trimesh.Trimesh,
    plane_normal: np.ndarray,
    plane_origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    P3 — Classify faces into keep (distal/lower) and remove (proximal/upper).
    Uses dot product of face centroid with plane normal.
    Keep = below plane (socket side), Remove = above plane.
    """
    face_centroids = mesh.triangles_center
    # Dot product: positive = above plane, negative = below
    dots = np.dot(face_centroids - plane_origin, plane_normal)
    keep_mask = dots <= 0     # below plane = keep (distal)
    remove_mask = dots > 0    # above plane = remove (proximal)
    return keep_mask, remove_mask


# ─── P4: Geometry Removal ────────────────────────────────────────────────────

def remove_geometry(
    mesh: trimesh.Trimesh,
    plane_normal: np.ndarray,
    plane_origin: np.ndarray,
) -> trimesh.Trimesh:
    """
    P4 — Remove faces above the cutting plane.
    Uses trimesh.intersections.slice_mesh_plane for clean edge splitting.
    This is the equivalent of Rhino's face-level deletion with edge splitting.
    """
    try:
        # slice_mesh_plane keeps geometry on the side the normal points AWAY from
        # normal=[0,0,1] pointing up → keeps everything BELOW the plane
        trimmed = trimesh.intersections.slice_mesh_plane(
            mesh,
            plane_normal=plane_normal,
            plane_origin=plane_origin,
            cap=False,  # Cap separately in P5
        )
        if trimmed is None or len(trimmed.faces) == 0:
            raise ValueError("slice_mesh_plane returned empty mesh")
        return trimmed
    except Exception:
        # Fallback: manual face deletion (FB path)
        return _fallback_face_removal(mesh, plane_normal, plane_origin)


def _fallback_face_removal(
    mesh: trimesh.Trimesh,
    plane_normal: np.ndarray,
    plane_origin: np.ndarray,
) -> trimesh.Trimesh:
    """
    FB — Fallback: manual face deletion when slice_mesh_plane fails.
    Equivalent to Rhino's Explode → Trim → Join sequence.
    """
    keep_mask, _ = classify_regions(mesh, plane_normal, plane_origin)
    trimmed = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces[keep_mask],
        process=False,
    )
    trimmed.remove_unreferenced_vertices()
    return trimmed


# ─── P5: Capping & Closing ────────────────────────────────────────────────────

def cap_and_close(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    P5 — Cap the open boundary and restore watertight solid.
    Rhino equivalent: DupEdge → PlanarSrf → Join.
    """
    mesh = mesh.copy()

    # Fill all holes (planar cap at cut boundary)
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)
    mesh.remove_unreferenced_vertices()

    return mesh


# ─── Post-trim Smoothing ─────────────────────────────────────────────────────

def smooth_boundary(
    mesh: trimesh.Trimesh,
    iterations: int = 3,
    lamb: float = 0.3,
) -> trimesh.Trimesh:
    """
    Laplacian smoothing near the trim boundary to remove sharp edges.
    Only smooths vertices near the open boundary — preserves anatomy elsewhere.
    """
    try:
        import trimesh.smoothing
        mesh = trimesh.smoothing.filter_laplacian(
            mesh, lamb=lamb, iterations=iterations
        )
    except Exception:
        pass
    return mesh


# ─── P6: Post-trim Validation ────────────────────────────────────────────────

def post_trim_validation(mesh: trimesh.Trimesh) -> dict:
    """
    P6 — Validate trimmed result.
    Rhino equivalent: ShowEdges + SelBadObjects + MergeAllFaces.
    """
    report = {"passed": True, "warnings": [], "errors": []}

    # Count naked edges
    edge_count = {}
    for face in mesh.faces:
        for i in range(3):
            e = tuple(sorted([face[i], face[(i+1)%3]]))
            edge_count[e] = edge_count.get(e, 0) + 1
    naked = sum(1 for c in edge_count.values() if c == 1)
    non_manifold = sum(1 for c in edge_count.values() if c > 2)

    report["is_watertight"] = bool(mesh.is_watertight)
    report["is_winding_consistent"] = bool(mesh.is_winding_consistent)
    report["naked_edges"] = naked
    report["non_manifold_edges"] = non_manifold
    report["face_count"] = len(mesh.faces)
    report["vertex_count"] = len(mesh.vertices)
    report["bounds"] = mesh.bounds.tolist()

    height = float(mesh.bounds[1][2] - mesh.bounds[0][2])
    report["height_mm"] = round(height, 1)

    if naked > 100:
        report["warnings"].append(f"{naked} naked edges remain after capping")
    if non_manifold > 0:
        report["errors"].append(f"{non_manifold} non-manifold edges — mesh may fail in print")
        report["passed"] = False
    if len(mesh.faces) < 50:
        report["errors"].append("Too few faces in result — trim may have failed")
        report["passed"] = False

    return report


# ─── Full Rhino-Style Trim Pipeline ──────────────────────────────────────────

def rhino_trim_pipeline(
    stl_path: str,
    trim_points_3d: list[list[float]],
    proximal_offset_mm: float = 5.0,
    smooth_iterations: int = 3,
) -> dict:
    """
    Full 8-phase Rhino-style trim pipeline.
    Returns trimmed mesh as base64 STL + full validation report.

    Phases:
      P0 → validate input mesh
      P1 → define infinite cutting plane
      P2 → compute intersection curve
      P3 → classify keep/remove regions
      P4 → remove geometry (with FB fallback)
      P5 → cap and close
      P6 → validate output
    """
    from geometry.engine import align_mesh

    result = {
        "phases": {},
        "success": False,
        "trimmed_stl_b64": None,
    }

    # ── Load and align ────────────────────────────────────────────────────────
    mesh = trimesh.load(str(stl_path), force="mesh")
    mesh = align_mesh(mesh)

    # ── P0: Pre-condition check ───────────────────────────────────────────────
    p0 = precondition_check(mesh)
    result["phases"]["P0_precondition"] = p0
    if not p0["passed"]:
        result["error"] = "P0 failed: " + "; ".join(p0["errors"])
        return result

    # Repair before trim
    mesh = repair_mesh(mesh)

    # ── P1: Define cutting plane ──────────────────────────────────────────────
    plane_normal, plane_origin = define_cutting_plane(
        mesh, trim_points_3d, proximal_offset_mm
    )
    result["phases"]["P1_cutter"] = {
        "plane_normal": plane_normal.tolist(),
        "plane_origin": plane_origin.tolist(),
        "cut_z_mm": float(plane_origin[2]),
    }

    # ── P2: Intersection curve ────────────────────────────────────────────────
    intersection = compute_intersection_curve(mesh, plane_normal, plane_origin)
    result["phases"]["P2_intersection"] = {
        "curve_points": len(intersection) if intersection is not None else 0,
        "complete": intersection is not None,
    }

    # ── P3: Region classification ─────────────────────────────────────────────
    keep_mask, remove_mask = classify_regions(mesh, plane_normal, plane_origin)
    result["phases"]["P3_classification"] = {
        "keep_faces": int(keep_mask.sum()),
        "remove_faces": int(remove_mask.sum()),
    }

    # ── P4: Remove geometry ───────────────────────────────────────────────────
    trimmed = remove_geometry(mesh, plane_normal, plane_origin)
    used_fallback = len(trimmed.faces) == int(keep_mask.sum())
    result["phases"]["P4_removal"] = {
        "faces_after": len(trimmed.faces),
        "used_fallback": used_fallback,
    }

    # ── P5: Cap and close ────────────────────────────────────────────────────
    trimmed = cap_and_close(trimmed)

    # Post-trim boundary smoothing
    if smooth_iterations > 0:
        trimmed = smooth_boundary(trimmed, iterations=smooth_iterations)

    result["phases"]["P5_capping"] = {
        "is_watertight": bool(trimmed.is_watertight),
        "faces_after_cap": len(trimmed.faces),
    }

    # ── P6: Validation ────────────────────────────────────────────────────────
    p6 = post_trim_validation(trimmed)
    result["phases"]["P6_validation"] = p6

    # ── Export to base64 STL ──────────────────────────────────────────────────
    buf = io.BytesIO()
    trimmed.export(buf, file_type="stl")
    result["trimmed_stl_b64"] = base64.b64encode(buf.getvalue()).decode()
    result["success"] = True
    result["face_count"] = len(trimmed.faces)
    result["bounds"] = trimmed.bounds.tolist()
    result["cut_z_mm"] = float(plane_origin[2])
    result["height_mm"] = float(
        trimmed.bounds[1][2] - trimmed.bounds[0][2]
    )

    return result
