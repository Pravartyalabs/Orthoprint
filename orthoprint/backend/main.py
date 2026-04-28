"""
OrthoPrint Backend — FastAPI
Pravartya Labs
"""
import uuid
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.auth        import router as auth_router
from api.patients    import router as patients_router
from api.scans       import router as scans_router
from api.geometry    import router as geometry_router
from api.clean       import router as clean_router
from api.adjustments import router as adjustments_router
from api.cases       import router as cases_router
from api.jobs        import router as jobs_router, job_registry

UPLOAD_DIR = Path("uploads")
EXPORT_DIR = Path("exports")
UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="OrthoPrint API",
    version="1.1.0",
    description="AI-assisted O&P digital design platform — Pravartya Labs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(auth_router,        prefix="/api/auth",        tags=["auth"])
app.include_router(patients_router,    prefix="/api/patients",    tags=["patients"])
app.include_router(scans_router,       prefix="/api/scans",       tags=["scans"])
app.include_router(clean_router,       prefix="/api/clean",       tags=["clean"])
app.include_router(adjustments_router, prefix="/api/adjustments", tags=["adjustments"])
app.include_router(geometry_router,    prefix="/api/geometry",    tags=["geometry"])
app.include_router(cases_router,       prefix="/api/cases",       tags=["cases"])
app.include_router(jobs_router,        prefix="/api/jobs",        tags=["jobs"])


# ─── WebSocket progress channel ──────────────────────────────────────────────

active_ws: dict[str, WebSocket] = {}


@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    active_ws[job_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        active_ws.pop(job_id, None)


async def broadcast_progress(job_id: str, payload: dict):
    ws = active_ws.get(job_id)
    if ws:
        try:
            import json
            await ws.send_text(json.dumps(payload))
        except Exception:
            active_ws.pop(job_id, None)


app.state.broadcast = broadcast_progress


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "orthoprint-api", "version": "1.1.0"}
