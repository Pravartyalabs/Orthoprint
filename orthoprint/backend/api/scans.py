"""
OrthoPrint — /api/scans router
Handles scan upload (STL, OBJ, PLY), file serving, and scan metadata.
"""

import uuid
import struct
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import trimesh
import numpy as np

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SUPPORTED_FORMATS = {".stl", ".obj", ".ply"}


@router.post("/upload")
async def upload_scan(file: UploadFile = File(...)):
    """
    Accept 3D scan in STL, OBJ, or PLY format.
    Converts OBJ and PLY to STL internally for a unified pipeline.
    Handles both mesh and point-cloud PLY files.
    """
    ext = Path(file.filename).suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. Accepted: STL, OBJ, PLY"
        )

    scan_id = str(uuid.uuid4())
    content = await file.read()

    # Save original file
    original_path = UPLOAD_DIR / f"{scan_id}_original{ext}"
    original_path.write_bytes(content)

    stl_path = UPLOAD_DIR / f"{scan_id}.stl"
    original_face_count = 0
    conversion_note = None

    # ---------------------------
    # CASE 1: STL (fast path)
    # ---------------------------
    if ext == ".stl":
        stl_path.write_bytes(content)
        original_face_count = _estimate_stl_face_count(content)

    # ---------------------------
    # CASE 2: OBJ / PLY
    # ---------------------------
    else:
        try:
            mesh = trimesh.load(str(original_path), force="mesh")

            # ⚠️ Handle PLY point cloud case
            if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
                raise ValueError(
                    "File does not contain a valid mesh (possibly point cloud PLY). "
                    "Please export mesh with faces."
                )

            # Clean mesh (important for prosthetic pipeline)
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_faces()

            original_face_count = len(mesh.faces)

            # Export to STL
            mesh.export(str(stl_path))

            conversion_note = (
                f"Converted from {ext.upper()} to STL "
                f"({original_face_count:,} faces)"
            )

        except Exception as e:
            original_path.unlink(missing_ok=True)
            raise HTTPException(
                422,
                f"Could not parse {ext.upper()} file: {str(e)}"
            )

    return {
        "scan_id": scan_id,
        "filename": file.filename,
        "original_format": ext.upper().lstrip("."),
        "size_bytes": len(content),
        "face_count_raw": original_face_count,
        "needs_decimation": original_face_count > 150_000,
        "conversion_note": conversion_note,
    }


@router.get("/{scan_id}/file")
def serve_scan(scan_id: str):
    """Serve the working STL for the 3D viewport."""
    path = UPLOAD_DIR / f"{scan_id}.stl"

    if not path.exists():
        raise HTTPException(404, "Scan not found")

    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{scan_id}.stl",
    )


@router.get("/{scan_id}/info")
def scan_info(scan_id: str):
    """Return basic mesh stats for a scan."""
    path = UPLOAD_DIR / f"{scan_id}.stl"

    if not path.exists():
        raise HTTPException(404, "Scan not found")

    try:
        mesh = trimesh.load(str(path), force="mesh")

        return {
            "scan_id": scan_id,
            "faces": len(mesh.faces),
            "vertices": len(mesh.vertices),
            "is_watertight": bool(mesh.is_watertight),
            "bounds_mm": mesh.bounds.tolist(),
            "height_mm": float(mesh.bounds[1][2] - mesh.bounds[0][2]),
        }

    except Exception as e:
        raise HTTPException(500, f"Could not read scan: {e}")


def _estimate_stl_face_count(data: bytes) -> int:
    """Fast face count from binary STL header (no full parse needed)."""
    if len(data) < 84:
        return 0
    try:
        return struct.unpack_from("<I", data, 80)[0]
    except Exception:
        return 0