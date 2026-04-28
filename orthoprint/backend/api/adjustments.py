"""
OrthoPrint — /api/adjustments router
Global volume, circumferential ring, and localised relief/build-up adjustments.
Applies to the working STL in-place (non-destructively via versioning).
"""
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import trimesh

from geometry.adjustments import apply_all_adjustments

router = APIRouter()

UPLOAD_DIR = Path("uploads")
VERSION_DIR = Path("versions")
VERSION_DIR.mkdir(exist_ok=True)


# ─── Request models ───────────────────────────────────────────────────────────

class RingAdjustmentIn(BaseModel):
    z_height_mm: float = Field(..., description="Z position of adjustment ring on aligned mesh")
    adjustment_mm: float = Field(
        ..., ge=-20.0, le=20.0,
        description="Radial change in mm. Negative = tighten, Positive = expand."
    )
    label: str = Field("", description="Optional clinical label e.g. 'patella tendon level'")


class LocalModificationIn(BaseModel):
    centre_3d: list[float] = Field(..., min_length=3, max_length=3,
        description="[x, y, z] focus point on mesh surface")
    radius_mm: float = Field(..., ge=5.0, le=60.0,
        description="Influence radius in mm")
    depth_mm: float = Field(..., ge=-10.0, le=10.0,
        description="Displacement depth. Negative = relief, Positive = build-up.")
    label: str = Field("", description="Clinical label e.g. 'fibula head relief'")


class AdjustmentRequest(BaseModel):
    scan_id: str
    global_volume_pct: float = Field(
        0.0, ge=-20.0, le=20.0,
        description="Global volume % change. Negative = reduce, Positive = expand."
    )
    ring_adjustments: list[RingAdjustmentIn] = Field(
        default_factory=list,
        description="Circumferential ring adjustments at specific Z heights"
    )
    local_modifications: list[LocalModificationIn] = Field(
        default_factory=list,
        description="Localised relief or build-up zones"
    )


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/apply")
def apply_adjustments(req: AdjustmentRequest):
    """
    Apply global volume, ring, and localised adjustments to the working scan.
    Saves a versioned backup before modifying.
    Returns adjustment report.
    """
    stl_path = UPLOAD_DIR / f"{req.scan_id}.stl"
    if not stl_path.exists():
        raise HTTPException(404, f"Scan {req.scan_id} not found")

    # Version backup — preserves pre-adjustment state for undo
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = VERSION_DIR / f"{req.scan_id}_{ts}.stl"
    shutil.copy2(stl_path, backup_path)

    # Load mesh
    mesh = trimesh.load(str(stl_path), force="mesh")

    # Apply all adjustments
    adjusted, report = apply_all_adjustments(
        mesh,
        global_volume_pct=req.global_volume_pct,
        ring_adjustments=[r.model_dump() for r in req.ring_adjustments],
        local_modifications=[m.model_dump() for m in req.local_modifications],
    )

    # Save adjusted mesh back
    adjusted.export(str(stl_path))

    report["scan_id"] = req.scan_id
    report["backup_version"] = backup_path.name
    return report


@router.get("/versions/{scan_id}")
def list_versions(scan_id: str):
    """List all saved versions for a scan (for undo support)."""
    versions = sorted(VERSION_DIR.glob(f"{scan_id}_*.stl"))
    return {
        "scan_id": scan_id,
        "versions": [v.name for v in versions],
        "count": len(versions),
    }


@router.post("/revert/{scan_id}/{version_name}")
def revert_to_version(scan_id: str, version_name: str):
    """Restore a previous version of the scan (undo adjustment)."""
    version_path = VERSION_DIR / version_name
    if not version_path.exists():
        raise HTTPException(404, "Version not found")
    if not version_name.startswith(scan_id):
        raise HTTPException(400, "Version does not belong to this scan")

    dest = UPLOAD_DIR / f"{scan_id}.stl"
    shutil.copy2(version_path, dest)
    return {"ok": True, "reverted_to": version_name}
