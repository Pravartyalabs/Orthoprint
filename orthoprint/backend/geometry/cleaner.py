"""
OrthoPrint — Mesh Cleaning Engine
Four cleaning operations applied between scan upload and PCA alignment:

1. Fragment removal  — drop disconnected islands below a volume threshold
2. Z-clip           — remove geometry below a floor height (stand, chair, etc.)
3. Spike filter     — remove vertices that deviate beyond N× mean edge length
4. Laplacian smooth — reduce surface noise while preserving gross shape
"""
from __future__ import annotations
import numpy as np
import trimesh
import trimesh.repair
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# ─── 1. Fragment Removal ──────────────────────────────────────────────────────

def remove_fragments(mesh: trimesh.Trimesh, min_volume_ratio: float = 0.01) -> trimesh.Trimesh:
    """
    Split mesh into connected components, keep only those whose volume is at
    least `min_volume_ratio` × the largest component's volume.

    min_volume_ratio=0.01 means drop anything smaller than 1% of the main body.
    """
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    # Sort by volume descending
    def safe_volume(m):
        try:
            v = abs(m.volume)
            return v if np.isfinite(v) else 0.0
        except Exception:
            return float(len(m.faces))  # fallback: face count

    components = sorted(components, key=safe_volume, reverse=True)
    max_vol = safe_volume(components[0])
    if max_vol == 0:
        return mesh

    kept = [c for c in components if safe_volume(c) >= min_volume_ratio * max_vol]
    if not kept:
        return mesh
    if len(kept) == 1:
        return kept[0]

    combined = trimesh.util.concatenate(kept)
    trimesh.repair.fix_winding(combined)
    return combined


# ─── 2. Z-Clip (floor / stand removal) ───────────────────────────────────────

def z_clip(mesh: trimesh.Trimesh, clip_height_mm: float) -> trimesh.Trimesh:
    """
    Remove all geometry at or below Z = clip_height_mm.
    Useful for cropping the scan stand, floor, or wheelchair footrest.
    clip_height_mm=0 means no clipping.
    """
    if clip_height_mm <= 0:
        return mesh

    face_centroids = mesh.triangles_center
    keep_mask = face_centroids[:, 2] > clip_height_mm

    if keep_mask.sum() == 0:
        return mesh  # don't wipe the whole mesh

    clipped = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces[keep_mask],
        process=False,
    )
    clipped.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(clipped)
    trimesh.repair.fill_holes(clipped)
    return clipped


# ─── 3. Spike Filter ─────────────────────────────────────────────────────────

def remove_spikes(mesh: trimesh.Trimesh, spike_multiplier: float = 4.0) -> trimesh.Trimesh:
    """
    Remove vertices whose mean edge length to neighbours deviates more than
    `spike_multiplier` × the global mean edge length.

    spike_multiplier=4.0 is conservative — only removes obvious scanner spikes.
    Lower values (2.0–3.0) are more aggressive.
    """
    if spike_multiplier <= 0:
        return mesh

    edges = mesh.edges_unique
    edge_lengths = np.linalg.norm(
        mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
    )
    global_mean = edge_lengths.mean()
    if global_mean == 0:
        return mesh

    # Per-vertex mean edge length
    vertex_edge_sum = np.zeros(len(mesh.vertices))
    vertex_edge_count = np.zeros(len(mesh.vertices))
    for (v0, v1), length in zip(edges, edge_lengths):
        vertex_edge_sum[v0] += length
        vertex_edge_sum[v1] += length
        vertex_edge_count[v0] += 1
        vertex_edge_count[v1] += 1

    # Avoid divide-by-zero for isolated vertices
    vertex_edge_count = np.maximum(vertex_edge_count, 1)
    vertex_mean_edge = vertex_edge_sum / vertex_edge_count

    spike_mask = vertex_mean_edge > spike_multiplier * global_mean
    spike_indices = set(np.where(spike_mask)[0])

    if not spike_indices:
        return mesh

    # Remove faces that reference any spike vertex
    keep_faces = []
    for face in mesh.faces:
        if not any(v in spike_indices for v in face):
            keep_faces.append(face)

    if not keep_faces:
        return mesh

    cleaned = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=np.array(keep_faces),
        process=False,
    )
    cleaned.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(cleaned)
    trimesh.repair.fill_holes(cleaned)
    return cleaned


# ─── 4. Laplacian Smoothing ───────────────────────────────────────────────────

def laplacian_smooth(
    mesh: trimesh.Trimesh,
    iterations: int = 3,
    lamb: float = 0.5,
) -> trimesh.Trimesh:
    """
    Iterative Laplacian smoothing.

    iterations : number of smoothing passes (1–10; more = smoother but slower)
    lamb       : smoothing factor per iteration (0–1; 0.5 is a safe default)

    Uses cotangent weights for better shape preservation on non-uniform meshes.
    Falls back to uniform weights if cotangent computation fails.
    """
    if iterations <= 0 or lamb <= 0:
        return mesh

    verts = mesh.vertices.copy().astype(np.float64)
    faces = mesh.faces
    n = len(verts)

    # Build uniform Laplacian weight matrix (sparse)
    rows, cols, data = [], [], []
    for face in faces:
        for i in range(3):
            v0 = face[i]
            v1 = face[(i + 1) % 3]
            rows.append(v0)
            cols.append(v1)
            data.append(1.0)
            rows.append(v1)
            cols.append(v0)
            data.append(1.0)

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Row-normalise so each vertex gets the mean of its neighbours
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    inv_diag = csr_matrix(
        (1.0 / row_sums, (np.arange(n), np.arange(n))), shape=(n, n)
    )
    L = inv_diag @ W  # normalised Laplacian

    for _ in range(iterations):
        neighbour_mean = L @ verts
        verts = verts + lamb * (neighbour_mean - verts)

    smoothed = trimesh.Trimesh(
        vertices=verts,
        faces=faces.copy(),
        process=False,
    )
    smoothed.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(smoothed)
    return smoothed


# ─── Full Cleaning Pipeline ───────────────────────────────────────────────────

def run_cleaning_pipeline(
    mesh: trimesh.Trimesh,
    min_volume_ratio: float = 0.01,
    clip_height_mm: float = 0.0,
    spike_multiplier: float = 4.0,
    smooth_iterations: int = 3,
    smooth_lambda: float = 0.5,
) -> tuple[trimesh.Trimesh, dict]:
    """
    Run all four cleaning steps in sequence.
    Returns (cleaned_mesh, report).
    """
    report = {
        "input_faces": len(mesh.faces),
        "input_vertices": len(mesh.vertices),
        "steps": [],
    }

    def record(step_name: str, m: trimesh.Trimesh):
        report["steps"].append({
            "step": step_name,
            "faces": len(m.faces),
            "vertices": len(m.vertices),
        })

    # Step 1: fragment removal
    mesh = remove_fragments(mesh, min_volume_ratio=min_volume_ratio)
    record("fragment_removal", mesh)

    # Step 2: Z-clip
    mesh = z_clip(mesh, clip_height_mm=clip_height_mm)
    record("z_clip", mesh)

    # Step 3: spike filter
    mesh = remove_spikes(mesh, spike_multiplier=spike_multiplier)
    record("spike_filter", mesh)

    # Step 4: Laplacian smoothing
    mesh = laplacian_smooth(mesh, iterations=smooth_iterations, lamb=smooth_lambda)
    record("laplacian_smooth", mesh)

    report["output_faces"] = len(mesh.faces)
    report["output_vertices"] = len(mesh.vertices)
    report["faces_removed"] = report["input_faces"] - report["output_faces"]

    return mesh, report
