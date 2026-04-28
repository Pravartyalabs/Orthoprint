"""
OrthoPrint — Parametric Adjustment Engine

Three clinically-driven adjustment operations:

1. Global volume adjustment  — uniform scale in XY to expand/contract
                               the socket across the full height
2. Circumferential ring adj  — level-specific XY scaling at Z-height
                               rings with smooth falloff between rings
3. Localised relief/build-up — Gaussian-weighted radial displacement
                               at a specific 3D point (bony prominence,
                               scar tissue, tender area)
"""
from __future__ import annotations
import numpy as np
import trimesh
import trimesh.repair
from scipy.ndimage import gaussian_filter1d


# ─── 1. Global Volume Adjustment ─────────────────────────────────────────────

def apply_global_volume_adjustment(
    mesh: trimesh.Trimesh,
    adjustment_pct: float,
) -> trimesh.Trimesh:
    """
    Uniformly scale the mesh in X and Y (not Z) to expand or contract
    the socket volume. Z is preserved to maintain socket height.

    adjustment_pct:
        Negative = reduce volume (tighter socket)
        Positive = expand volume (looser socket)
        Range: -20% to +20%
        e.g. -5.0 means 5% volume reduction

    Clinical use: global fit adjustments for residual limb volume changes
    (e.g. post-operative oedema reduction).
    """
    if abs(adjustment_pct) < 0.01:
        return mesh

    scale_xy = 1.0 + (adjustment_pct / 100.0)
    scale_xy = max(0.5, min(2.0, scale_xy))  # safety clamp

    mesh = mesh.copy()
    # Scale X and Y around the mesh centroid in XY
    cx, cy = mesh.centroid[0], mesh.centroid[1]

    mesh.vertices[:, 0] = cx + (mesh.vertices[:, 0] - cx) * scale_xy
    mesh.vertices[:, 1] = cy + (mesh.vertices[:, 1] - cy) * scale_xy

    trimesh.repair.fix_winding(mesh)
    return mesh


# ─── 2. Circumferential Ring Adjustments ─────────────────────────────────────

class RingAdjustment:
    """A single circumferential ring adjustment."""
    def __init__(self, z_height_mm: float, adjustment_mm: float):
        """
        z_height_mm  : Z position of the ring on the aligned mesh
        adjustment_mm: radial change in mm
                       Negative = tighten (relief), Positive = expand (build-up)
        """
        self.z = z_height_mm
        self.delta = adjustment_mm


def apply_ring_adjustments(
    mesh: trimesh.Trimesh,
    rings: list[RingAdjustment],
    falloff_mm: float = 15.0,
) -> trimesh.Trimesh:
    """
    Apply level-specific circumferential adjustments with smooth Gaussian
    falloff between rings so adjacent levels blend naturally.

    rings      : list of RingAdjustment objects
    falloff_mm : half-width of the Gaussian influence zone around each ring.
                 15mm gives a smooth transition; reduce for sharper changes.

    Algorithm:
      For each vertex, compute the total radial displacement as the sum of
      contributions from all rings, weighted by a Gaussian centred at each
      ring's Z height. Displace vertices outward/inward along their
      projection from the mesh centroid XY.
    """
    if not rings:
        return mesh

    mesh = mesh.copy()
    verts = mesh.vertices.copy()

    cx = np.mean(verts[:, 0])
    cy = np.mean(verts[:, 1])

    # Radial direction (unit vector in XY plane from centroid)
    dx = verts[:, 0] - cx
    dy = verts[:, 1] - cy
    r  = np.sqrt(dx**2 + dy**2)
    r_safe = np.maximum(r, 1e-6)
    nx = dx / r_safe   # unit radial X
    ny = dy / r_safe   # unit radial Y

    z_verts = verts[:, 2]
    sigma = falloff_mm

    total_radial_delta = np.zeros(len(verts))

    for ring in rings:
        # Gaussian weight: vertices near ring.z get full adjustment
        weight = np.exp(-0.5 * ((z_verts - ring.z) / sigma) ** 2)
        total_radial_delta += weight * ring.delta

    # Apply radial displacement
    verts[:, 0] += nx * total_radial_delta
    verts[:, 1] += ny * total_radial_delta

    mesh.vertices = verts
    trimesh.repair.fix_winding(mesh)
    return mesh


# ─── 3. Localised Relief and Build-Up ────────────────────────────────────────

def apply_local_modification(
    mesh: trimesh.Trimesh,
    centre_3d: list[float],
    radius_mm: float,
    depth_mm: float,
) -> trimesh.Trimesh:
    """
    Apply a localised Gaussian-weighted radial modification at a specific
    3D point on the mesh surface.

    centre_3d : [x, y, z] — the focus point (e.g. bony prominence)
    radius_mm : influence radius — vertices within this sphere are affected
    depth_mm  : displacement magnitude along vertex normal
                Negative = relief (push socket outward = more space for prominence)
                Positive = build-up (push socket inward = add padding)

    Clinical use:
      Relief:    fibula head, patella, tibial crest, bony prominences
      Build-up:  medial tibial flare, areas needing more contact/support
    """
    if abs(depth_mm) < 0.01 or radius_mm <= 0:
        return mesh

    mesh = mesh.copy()
    centre = np.array(centre_3d, dtype=float)
    verts  = mesh.vertices.copy()
    normals = mesh.vertex_normals

    # Distance from each vertex to the centre point
    dist = np.linalg.norm(verts - centre, axis=1)

    # Gaussian weight — falls to ~0 at radius_mm
    sigma = radius_mm / 2.5
    weight = np.exp(-0.5 * (dist / sigma) ** 2)
    weight[dist > radius_mm] = 0.0   # hard cutoff outside radius

    # Displace along vertex normals
    displacement = weight[:, np.newaxis] * normals * depth_mm
    verts += displacement

    mesh.vertices = verts
    trimesh.repair.fix_winding(mesh)
    return mesh


# ─── Combined Adjustment Pipeline ────────────────────────────────────────────

def apply_all_adjustments(
    mesh: trimesh.Trimesh,
    global_volume_pct: float = 0.0,
    ring_adjustments: list[dict] | None = None,
    local_modifications: list[dict] | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """
    Apply all adjustments in sequence:
      1. Global volume
      2. Ring-based circumferential
      3. Localised relief/build-up

    ring_adjustments  : list of {"z_height_mm": float, "adjustment_mm": float}
    local_modifications: list of {"centre_3d": [x,y,z], "radius_mm": float, "depth_mm": float}

    Returns (adjusted_mesh, report).
    """
    report = {"steps": []}

    # 1. Global
    if abs(global_volume_pct) >= 0.01:
        mesh = apply_global_volume_adjustment(mesh, global_volume_pct)
        report["steps"].append({
            "step": "global_volume",
            "adjustment_pct": global_volume_pct,
        })

    # 2. Ring-based
    if ring_adjustments:
        rings = [
            RingAdjustment(r["z_height_mm"], r["adjustment_mm"])
            for r in ring_adjustments
        ]
        mesh = apply_ring_adjustments(mesh, rings)
        report["steps"].append({
            "step": "ring_adjustments",
            "count": len(rings),
            "rings": ring_adjustments,
        })

    # 3. Local modifications
    if local_modifications:
        for mod in local_modifications:
            mesh = apply_local_modification(
                mesh,
                centre_3d=mod["centre_3d"],
                radius_mm=mod["radius_mm"],
                depth_mm=mod["depth_mm"],
            )
        report["steps"].append({
            "step": "local_modifications",
            "count": len(local_modifications),
        })

    report["output_faces"] = len(mesh.faces)
    return mesh, report
