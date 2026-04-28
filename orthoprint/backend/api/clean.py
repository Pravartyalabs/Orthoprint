"""
OrthoPrint — /api/clean router
Mesh cleaning endpoint: fragment removal, Z-clip, spike filter, smoothing.
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import trimesh

from geometry.cleaner import run_cleaning_pipeline

router = APIRouter()

UPLOAD_DIR = Path("uploads")


class CleanRequest(BaseModel):
    scan_id: str
    min_volume_ratio: float = Field(
        0.01, ge=0.0, le=1.0,
        description="Drop fragments smaller than this fraction of main body volume"
    )
    clip_height_mm: float = Field(
        0.0, ge=0.0, le=200.0,
        description="Remove everything at or below this Z height (mm). 0 = no clip."
    )
    spike_multiplier: float = Field(
        4.0, ge=1.0, le=20.0,
        description="Remove vertices whose edge length exceeds N× global mean. Higher = less aggressive."
    )
    smooth_iterations: int = Field(
        3, ge=0, le=10,
        description="Number of Laplacian smoothing passes. 0 = no smoothing."
    )
    smooth_lambda: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Smoothing strength per iteration (0–1)."
    )


@router.post("/")
def clean_scan(req: CleanRequest):
    """
    Run the cleaning pipeline on an uploaded scan.
    Overwrites the stored STL with the cleaned version.
    Returns a per-step report.
    """
    stl_path = UPLOAD_DIR / f"{req.scan_id}.stl"
    if not stl_path.exists():
        raise HTTPException(404, f"Scan {req.scan_id} not found")

    mesh = trimesh.load(str(stl_path), force="mesh")

    cleaned, report = run_cleaning_pipeline(
        mesh,
        min_volume_ratio=req.min_volume_ratio,
        clip_height_mm=req.clip_height_mm,
        spike_multiplier=req.spike_multiplier,
        smooth_iterations=req.smooth_iterations,
        smooth_lambda=req.smooth_lambda,
    )

    cleaned.export(str(stl_path))

    return {
        "scan_id": req.scan_id,
        "cleaned": True,
        **report,
    }


@router.get("/preview/{scan_id}")
def cleaning_preview(scan_id: str):
    """
    Return stats about the current scan without modifying it.
    Used to show the clinician the mesh state before cleaning.
    """
    stl_path = UPLOAD_DIR / f"{scan_id}.stl"
    if not stl_path.exists():
        raise HTTPException(404, f"Scan {scan_id} not found")

    mesh = trimesh.load(str(stl_path), force="mesh")
    components = mesh.split(only_watertight=False)

    return {
        "scan_id": scan_id,
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "is_watertight": mesh.is_watertight,
        "component_count": len(components),
        "bounds_mm": mesh.bounds.tolist(),
        "z_min": float(mesh.bounds[0][2]),
        "z_max": float(mesh.bounds[1][2]),
        "height_mm": float(mesh.bounds[1][2] - mesh.bounds[0][2]),
    }
