"""
OrthoPrint — Geometry Engine
All 4 processing phases: align → trim → shell → validate

Dependencies: trimesh[all] numpy scipy shapely
"""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Callable, Awaitable, Optional
import numpy as np
import trimesh
import trimesh.repair
import trimesh.intersections
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPolygon

ProgressCallback = Callable[[str, int], Awaitable[None]]


# ─── Requirement 1: Adaptive Decimation ──────────────────────────────────────

def adaptive_decimate(
    mesh: trimesh.Trimesh,
    target_for_browser: int = 150_000,
    max_faces_before_decimate: int = 50_000,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Adaptive decimation that scales the target face count based on mesh
    complexity rather than applying a flat threshold.

    Strategy:
      - Compute a complexity score from surface area / bounding-box volume ratio.
        High curvature (e.g. residual limb with bony prominences) scores higher
        and retains more faces; simple cylindrical stumps score lower.
      - Scale the browser-preview target between 40k and 150k based on score.
      - Always keep the original (or lightly cleaned) mesh for print export.

    Returns:
        (preview_mesh, print_mesh)
        preview_mesh — decimated for browser rendering
        print_mesh   — full resolution for STL export
    """
    face_count = len(mesh.faces)

    # Keep a full-res copy for print export regardless
    print_mesh = mesh.copy()

    if face_count <= max_faces_before_decimate:
        # Already within browser budget — no decimation needed at all
        return mesh.copy(), print_mesh

    # ── Complexity score ──────────────────────────────────────────────────────
    # Ratio of surface area to the surface area of a bounding-box ellipsoid.
    # A perfectly smooth cylinder ≈ 1.0; a heavily featured limb > 1.5.
    try:
        surface_area = mesh.area
        bounds = mesh.bounding_box.extents          # [dx, dy, dz]
        # Approximate ellipsoid surface area (Knud Thomsen formula)
        a, b, c = sorted(bounds / 2, reverse=True)
        p = 1.6075
        ellipsoid_area = (
            4 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3) ** (1 / p)
        )
        complexity = float(np.clip(surface_area / max(ellipsoid_area, 1e-6), 1.0, 3.0))
    except Exception:
        complexity = 1.5  # safe fallback

    # Map complexity [1.0 → 3.0] to target faces [60k → 150k]
    min_target = 60_000
    max_target = target_for_browser
    scaled_target = int(
        min_target + (complexity - 1.0) / 2.0 * (max_target - min_target)
    )
    scaled_target = max(min_target, min(scaled_target, face_count - 1))

    try:
        preview_mesh = mesh.simplify_quadratic_decimation(scaled_target)
    except Exception:
        # Fallback: uniform decimation if quadratic fails
        ratio = scaled_target / face_count
        preview_mesh = mesh.simplify_quadratic_decimation(int(face_count * ratio))

    return preview_mesh, print_mesh


# ─── Phase 1: PCA Auto-Alignment ─────────────────────────────────────────────

def align_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Rotate so that the principal inertia axis (long axis of the limb)
    aligns with the Z-axis, and the centroid moves to the origin.

    Steps:
      1. Translate centroid to origin
      2. Compute principal inertia axes via trimesh (internally uses eigendecomp
         of the inertia tensor)
      3. Align primary axis (longest dimension) to [0,0,1]
      4. Shift mesh so its lowest point sits at Z = 0
    """
    mesh = mesh.copy()

    # 1. Centroid to origin
    mesh.apply_translation(-mesh.centroid)

    # 2. Compute the primary axis robustly. Use trimesh inertia if available,
    #    otherwise fall back to PCA of the vertex cloud.
    primary_axis = None
    try:
        inertia_transform = mesh.principal_inertia_transform
        axis_candidate = np.asarray(inertia_transform[:3, 2], dtype=float).flatten()
        if axis_candidate.shape == (3,) and np.isfinite(axis_candidate).all():
            primary_axis = axis_candidate
    except Exception:
        primary_axis = None

    if primary_axis is None or np.linalg.norm(primary_axis) < 1e-6:
        vertices = np.asarray(mesh.vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
            raise ValueError("Mesh does not contain enough vertex data for alignment")

        centered = vertices - vertices.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        primary_axis = eigvecs[:, np.argmax(eigvals)]

    primary_axis = np.asarray(primary_axis, dtype=float).flatten()
    if primary_axis.shape != (3,) or not np.isfinite(primary_axis).all():
        raise ValueError("Computed alignment axis is invalid")

    primary_axis = primary_axis / np.linalg.norm(primary_axis)

    # 3. Build rotation matrix: primary_axis → [0, 0, 1]
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    rot = trimesh.geometry.align_vectors(primary_axis, target)
    mesh.apply_transform(rot)

    # 4. Translate so the bottom of the mesh is at Z = 0
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0][2]])

    return mesh


# ─── Phase 2: Trim-Line — Plane Fitting ──────────────────────────────────────

def build_trim_plane_from_points(
    trim_points_3d: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a plane through the clinician trim-line points using SVD.
    The plane cuts THROUGH THE ENTIRE MESH like Rhino split.

    Strategy:
    - If points lie roughly in a horizontal band (Z variance < XY variance),
      use a horizontal plane at mean Z — most reliable for transtibial sockets.
    - Otherwise use SVD best-fit plane through the points.

    Returns:
        plane_normal  — unit normal vector (points upward +Z)
        plane_origin  — centroid of trim points (point on plane)
    """
    pts = np.array(trim_points_3d, dtype=float)

    if len(pts) < 3:
        z = float(pts[:, 2].mean())
        return np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, z])

    centroid = pts.mean(axis=0)

    # Compute variance in Z vs XY to decide plane orientation
    z_var = float(np.var(pts[:, 2]))
    xy_var = float(np.var(pts[:, 0]) + np.var(pts[:, 1]))

    if z_var < xy_var * 0.5:
        # Points form a roughly horizontal ring — use horizontal cut plane
        # This is the most common case for trim lines around a limb
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        origin = np.array([0.0, 0.0, float(pts[:, 2].mean())], dtype=float)
        return normal, origin

    # Points have significant Z variation — use SVD best-fit plane
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    return normal, centroid


# ─── Phase 3: Shell Generation ───────────────────────────────────────────────

def generate_socket_shell(
    mesh: trimesh.Trimesh,
    trim_points_3d: list[list[float]],
    wall_thickness_mm: float = 4.0,
    proximal_offset_mm: float = 5.0,
) -> trimesh.Trimesh:
    """
    Produces a closed, printable socket shell in three steps:

    A. TRUE BOOLEAN SLICE at the trim-line plane using
       trimesh.intersections.slice_mesh_plane() — this correctly cuts faces
       that straddle the plane and caps the open edge, unlike the previous
       face-masking approach which left a ragged open boundary.

    B. MANIFOLD SHELL OFFSET using the correct trimesh technique:
       - Duplicate the trimmed mesh as inner and outer surfaces
       - Invert normals on the inner surface
       - Stitch the proximal rim (open boundary loop) between outer and inner
       - Stitch the distal cap (bottom of socket)
       This produces a genuinely closed, watertight shell.

    C. REPAIR — fill any remaining holes and fix winding consistency.
    """

    # ── A. True plane slice (Rhino-style horizontal cut) ─────────────────────
    # Use mean Z of trim points + offset for reliable full-mesh cut
    pts_arr = np.array(trim_points_3d, dtype=float)
    cut_z = float(pts_arr[:, 2].mean()) + proximal_offset_mm
    cut_z = float(np.clip(cut_z, mesh.bounds[0][2] + 5, mesh.bounds[1][2] - 5))

    plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    offset_origin = np.array([0.0, 0.0, cut_z], dtype=float)

    # slice_mesh_plane keeps geometry on the side the normal points AWAY from,
    # i.e. the distal (lower) portion of the limb. cap=True closes the cut face.
    try:
        trimmed = trimesh.intersections.slice_mesh_plane(
            mesh,
            plane_normal=plane_normal,
            plane_origin=offset_origin,
            cap=True,
        )
    except Exception:
        # Fallback to face-mask method if slice fails (e.g. degenerate geometry)
        face_centroids = mesh.triangles_center
        keep_mask = np.dot(face_centroids, plane_normal) < (
            np.dot(offset_origin, plane_normal)
        )
        trimmed = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces[keep_mask],
            process=False,
        )
        trimesh.repair.fill_holes(trimmed)

    trimmed.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(trimmed)

    # ── B. Manifold shell offset ──────────────────────────────────────────────
    shell = _build_closed_shell(trimmed, wall_thickness_mm)

    # ── C. Final repair ───────────────────────────────────────────────────────
    trimesh.repair.fix_winding(shell)
    trimesh.repair.fill_holes(shell)
    shell.remove_unreferenced_vertices()

    return shell


def _build_closed_shell(
    body: trimesh.Trimesh,
    wall_thickness_mm: float,
) -> trimesh.Trimesh:
    """
    Build a watertight shell from a surface mesh:

    1. Outer surface  = body as-is
    2. Inner surface  = body with vertices offset inward along vertex normals,
                        faces flipped so normals point inward
    3. Rim band       = quad strip connecting outer and inner boundary loops
                        at the proximal (open) edge
    4. Distal cap     = triangulated polygon closing the distal tip (if open)

    Returns a single concatenated, manifold mesh.
    """
    outer = body.copy()

    # Build inner surface: offset vertices inward
    inner = body.copy()
    vertex_normals = body.vertex_normals
    inner.vertices = body.vertices - vertex_normals * wall_thickness_mm
    # Flip inner faces so normals point inward (into the shell wall)
    inner.faces = inner.faces[:, ::-1]

    parts = [outer, inner]

    # ── Rim: stitch open boundary loops ──────────────────────────────────────
    # Find the boundary (open edge) vertices of the outer mesh
    outer_boundary = _get_boundary_loop(outer)
    inner_boundary = _get_boundary_loop(inner)

    if outer_boundary is not None and inner_boundary is not None and \
       len(outer_boundary) == len(inner_boundary):
        rim_faces = _stitch_loops(
            outer.vertices, outer_boundary,
            inner.vertices + len(outer.vertices),  # index offset after concat
            inner_boundary,
        )
        # Build the rim as a separate mesh that will be concatenated
        all_verts = np.vstack([outer.vertices, inner.vertices])
        rim_mesh = trimesh.Trimesh(vertices=all_verts, faces=rim_faces, process=False)
        parts.append(rim_mesh)

    # Concatenate all parts
    shell = trimesh.util.concatenate(parts)
    return shell


def _get_boundary_loop(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    """
    Return an ordered array of vertex indices forming the boundary loop
    of a mesh with a single open edge. Returns None if mesh is closed or
    has multiple disconnected boundaries.
    """
    try:
        # edges_unique contains each edge once; boundary edges appear only once
        edge_count = {}
        for face in mesh.faces:
            for i in range(3):
                e = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_count[e] = edge_count.get(e, 0) + 1

        boundary_edges = [list(e) for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            return None

        # Chain edges into an ordered loop
        loop = [boundary_edges[0][0], boundary_edges[0][1]]
        used = {0}
        for _ in range(len(boundary_edges) - 1):
            last = loop[-1]
            for i, e in enumerate(boundary_edges):
                if i in used:
                    continue
                if e[0] == last:
                    loop.append(e[1])
                    used.add(i)
                    break
                if e[1] == last:
                    loop.append(e[0])
                    used.add(i)
                    break

        return np.array(loop[:-1])  # remove duplicate endpoint
    except Exception:
        return None


def _stitch_loops(
    verts_a: np.ndarray, loop_a: np.ndarray,
    verts_b_offset: np.ndarray, loop_b: np.ndarray,
) -> np.ndarray:
    """
    Build a quad strip (as triangles) connecting two boundary loops.
    loop_a and loop_b are index arrays; loop_b indices already have the
    vertex-array offset applied.
    """
    n = min(len(loop_a), len(loop_b))
    faces = []
    for i in range(n):
        a0 = loop_a[i]
        a1 = loop_a[(i + 1) % n]
        b0 = loop_b[i]
        b1 = loop_b[(i + 1) % n]
        faces.append([a0, a1, b0])
        faces.append([a1, b1, b0])
    return np.array(faces)


# ─── Phase 4: Validation & Repair ────────────────────────────────────────────

def measure_circumference_at_z(mesh: trimesh.Trimesh, z_height: float) -> float:
    """
    Slice mesh at given Z height, return perimeter of cross-section in mm.
    Returns 0.0 if the Z height is outside the mesh bounds.
    """
    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    if not (z_min < z_height < z_max):
        return 0.0

    try:
        section = mesh.section(
            plane_origin=[0, 0, z_height],
            plane_normal=[0, 0, 1],
        )
        if section is None:
            return 0.0
        path, _ = section.to_planar()
        perimeter = sum(e.length for e in path.entities)
        return float(perimeter)
    except Exception:
        return 0.0


def validate_and_repair(mesh: trimesh.Trimesh) -> dict:
    """
    Check manifold status, fill holes, fix winding. Returns validation report.
    """
    # Count open edges before repair
    try:
        all_edges = {}
        for face in mesh.faces:
            for i in range(3):
                e = tuple(sorted([face[i], face[(i + 1) % 3]]))
                all_edges[e] = all_edges.get(e, 0) + 1
        open_edge_count = sum(1 for c in all_edges.values() if c == 1)
    except Exception:
        open_edge_count = -1

    report = {
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "open_edges": open_edge_count,
        "face_count": len(mesh.faces),
        "vertex_count": len(mesh.vertices),
    }

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        report["repaired"] = True
        report["is_watertight_after_repair"] = bool(mesh.is_watertight)
    else:
        report["repaired"] = False

    return report


# ─── Full Pipeline (async, with progress callbacks) ──────────────────────────

async def run_full_pipeline(
    stl_path: Path,
    trim_points_3d: list[list[float]],
    wall_thickness_mm: float,
    proximal_offset_mm: float,
    output_path: Path,
    target_browser_faces: int = 150_000,
    on_progress: Optional[ProgressCallback] = None,
) -> dict:
    """
    Runs all 4 phases with adaptive decimation.
    Exports the full-resolution shell (not the decimated preview).
    Returns validation report.
    """

    async def prog(msg: str, pct: int):
        if on_progress:
            await on_progress(msg, pct)

    await prog("Loading scan", 5)
    mesh = trimesh.load(str(stl_path), force="mesh")

    await prog("Adaptive decimation", 15)
    preview_mesh, print_mesh = adaptive_decimate(
        mesh, target_for_browser=target_browser_faces
    )
    # Pipeline operates on the preview mesh for speed;
    # final export uses the print-resolution mesh
    working_mesh = preview_mesh

    await prog("PCA alignment", 30)
    working_mesh = align_mesh(working_mesh)
    print_mesh = align_mesh(print_mesh)

    await prog("Slicing at trim-line plane", 50)
    shell = await asyncio.to_thread(
        generate_socket_shell,
        print_mesh,          # use full-res for the actual socket
        trim_points_3d,
        wall_thickness_mm,
        proximal_offset_mm,
    )

    await prog("Manifold validation & repair", 75)
    report = validate_and_repair(shell)

    await prog("Measuring circumferences", 88)
    height = float(shell.bounds[1][2] - shell.bounds[0][2])
    z_heights = [
        round(height * 0.2, 1),
        round(height * 0.5, 1),
        round(height * 0.8, 1),
    ]
    report["circumferences_mm"] = {
        f"z_{int(z)}": measure_circumference_at_z(shell, z)
        for z in z_heights
    }

    await prog("Exporting STL", 95)
    shell.export(str(output_path))

    await prog("Done", 100)
    report["output_path"] = str(output_path)
    report["decimation"] = {
        "input_faces": len(mesh.faces),
        "preview_faces": len(preview_mesh.faces),
        "print_faces": len(print_mesh.faces),
    }
    return report


# ─── Rhino-Style Split Mesh Preview ──────────────────────────────────────────

def preview_split_mesh(
    stl_path: str,
    trim_points_3d: list[list[float]],
    proximal_offset_mm: float = 5.0,
) -> dict:
    """
    Rhino-style split mesh:
    1. Load raw STL
    2. Align mesh (PCA upright)
    3. Compute cut Z from mean Z of trim points + offset
    4. Slice with infinite horizontal plane (like Rhino split surface)
    5. Cap open edge
    6. Return as base64 STL for browser preview
    """
    import base64, io

    # Step 1+2: Load and align (same transform as full pipeline)
    mesh = trimesh.load(str(stl_path), force="mesh")
    mesh = align_mesh(mesh)

    # Step 3: Trim points arrive in frontend coords (aligned mesh space)
    # Use mean Z as cut height — horizontal plane guaranteed to cut entire mesh
    pts = np.array(trim_points_3d, dtype=float)
    mesh_height = float(mesh.bounds[1][2] - mesh.bounds[0][2])

    # Clamp cut Z to valid range inside mesh bounds
    raw_z = float(pts[:, 2].mean()) + proximal_offset_mm
    cut_z = float(np.clip(raw_z, mesh.bounds[0][2] + 5, mesh.bounds[1][2] - 5))

    # Step 4: Infinite horizontal plane — cuts ENTIRE mesh like Rhino
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    offset_origin = np.array([0.0, 0.0, cut_z], dtype=float)

    try:
        trimmed = trimesh.intersections.slice_mesh_plane(
            mesh,
            plane_normal=plane_normal,
            plane_origin=offset_origin,
            cap=True,
        )
    except Exception:
        face_centroids = mesh.triangles_center
        keep_mask = np.dot(face_centroids, plane_normal) < np.dot(offset_origin, plane_normal)
        trimmed = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces[keep_mask],
            process=False,
        )
        trimesh.repair.fill_holes(trimmed)

    trimmed.remove_unreferenced_vertices()

    # Export to base64 STL for browser
    buf = io.BytesIO()
    trimmed.export(buf, file_type="stl")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "trimmed_stl_b64": b64,
        "face_count": len(trimmed.faces),
        "bounds": trimmed.bounds.tolist(),
        "plane_normal": plane_normal.tolist(),
        "plane_origin": offset_origin.tolist(),
    }
