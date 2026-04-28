"""
OrthoPrint — /api/geometry router
Handles geometry job dispatch and result retrieval.
"""
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from geometry.engine import run_full_pipeline, align_mesh
from api.jobs import job_registry, JobStatus
import trimesh

router = APIRouter()

UPLOAD_DIR = Path("uploads")
EXPORT_DIR = Path("exports")


class GeometryRequest(BaseModel):
    scan_id: str
    trim_points_3d: list[list[float]] = Field(
        ..., description="List of [x,y,z] points defining the trim-line"
    )
    wall_thickness_mm: float = Field(4.0, ge=1.0, le=12.0)
    proximal_offset_mm: float = Field(5.0, ge=0.0, le=30.0)
    target_browser_faces: int = Field(
        150_000, ge=20_000, le=500_000,
        description="Adaptive decimation target for browser preview (scales with mesh complexity)"
    )


class GeometryResponse(BaseModel):
    job_id: str
    status: str
    ws_url: str


@router.post("/process", response_model=GeometryResponse)
async def process_geometry(req: GeometryRequest, request: Request):
    """
    Dispatch a geometry processing job. 
    Returns job_id — connect via WebSocket ws://<host>/ws/{job_id} for progress.
    """
    stl_path = UPLOAD_DIR / f"{req.scan_id}.stl"
    if not stl_path.exists():
        raise HTTPException(404, f"Scan {req.scan_id} not found")

    job_id = str(uuid.uuid4())
    output_path = EXPORT_DIR / f"{job_id}_socket.stl"

    job_registry[job_id] = {"status": JobStatus.PENDING, "result": None}

    broadcaster = request.app.state.broadcast

    async def _run():
        job_registry[job_id]["status"] = JobStatus.RUNNING
        try:
            async def on_progress(stage: str, pct: int):
                await broadcaster(job_id, {"stage": stage, "pct": pct})

            result = await run_full_pipeline(
                stl_path=stl_path,
                trim_points_3d=req.trim_points_3d,
                wall_thickness_mm=req.wall_thickness_mm,
                proximal_offset_mm=req.proximal_offset_mm,
                output_path=output_path,
                target_browser_faces=req.target_browser_faces,
                on_progress=on_progress,
            )
            result["job_id"] = job_id  # so frontend can navigate to /validate/{job_id}
            job_registry[job_id]["status"] = JobStatus.DONE
            job_registry[job_id]["result"] = result
            await broadcaster(job_id, {"stage": "Done", "pct": 100, "result": result})
        except Exception as e:
            job_registry[job_id]["status"] = JobStatus.FAILED
            job_registry[job_id]["error"] = str(e)
            await broadcaster(job_id, {"stage": "Error", "pct": -1, "error": str(e)})

    asyncio.create_task(_run())

    host = request.headers.get("host", "localhost:8000")
    return GeometryResponse(
        job_id=job_id,
        status="queued",
        ws_url=f"ws://{host}/ws/{job_id}",
    )


@router.get("/result/{job_id}")
def get_result(job_id: str):
    """Poll for job result (alternative to WebSocket)."""
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/download/{job_id}")
def download_socket(job_id: str):
    """Download the generated socket STL."""
    path = EXPORT_DIR / f"{job_id}_socket.stl"
    if not path.exists():
        raise HTTPException(404, "Export not ready")
    return FileResponse(path, filename=f"socket_{job_id}.stl", media_type="application/octet-stream")


class AlignRequest(BaseModel):
    scan_id: str


@router.post("/align")
def align_scan(req: AlignRequest):
    """
    Run PCA alignment on an uploaded scan in-place.
    Overwrites the stored STL with the aligned version.
    """
    stl_path = UPLOAD_DIR / f"{req.scan_id}.stl"
    if not stl_path.exists():
        raise HTTPException(404, f"Scan {req.scan_id} not found")

    mesh = trimesh.load(str(stl_path), force="mesh")
    aligned = align_mesh(mesh)
    aligned.export(str(stl_path))

    return {
        "scan_id": req.scan_id,
        "aligned": True,
        "bounds": aligned.bounds.tolist(),
        "face_count": len(aligned.faces),
    }



class SplitPreviewRequest(BaseModel):
    scan_id: str
    trim_points_3d: list[list[float]]
    proximal_offset_mm: float = Field(5.0, ge=0.0, le=30.0)


@router.post("/split-preview")
def split_preview(req: SplitPreviewRequest):
    """
    Returns a preview of the trimmed mesh using Rhino-style split technique.
    Frontend displays this before committing to full shell generation.
    """
    from geometry.engine import preview_split_mesh
    stl_path = UPLOAD_DIR / f"{req.scan_id}.stl"
    if not stl_path.exists():
        raise HTTPException(404, f"Scan {req.scan_id} not found")
    try:
        result = preview_split_mesh(
            str(stl_path),
            req.trim_points_3d,
            req.proximal_offset_mm,
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


class RhinoTrimRequest(BaseModel):
    scan_id: str
    trim_points_3d: list[list[float]]
    proximal_offset_mm: float = Field(5.0, ge=0.0, le=30.0)
    smooth_iterations: int = Field(3, ge=0, le=10)


@router.post("/rhino-trim")
def rhino_trim(req: RhinoTrimRequest):
    """
    Full Rhino-style 8-phase trim pipeline.
    P0 validate → P1 cutter → P2 intersect → P3 classify →
    P4 remove → P5 cap → P6 validate → return trimmed STL
    """
    from geometry.trim_engine import rhino_trim_pipeline
    stl_path = UPLOAD_DIR / f"{req.scan_id}.stl"
    if not stl_path.exists():
        raise HTTPException(404, f"Scan {req.scan_id} not found")
    try:
        result = rhino_trim_pipeline(
            str(stl_path),
            req.trim_points_3d,
            req.proximal_offset_mm,
            req.smooth_iterations,
        )
        if not result["success"]:
            raise HTTPException(500, result.get("error", "Trim failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
