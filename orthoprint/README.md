# OrthoPrint — Cloud-to-Print Toolchain
**Pravartya Labs | Confidential & Proprietary**

A full-stack web application for designing and 3D-printing orthotic & prosthetic
(O&P) socket devices — from raw limb scan to print-ready STL.

---

## Stack

| Layer | Technology |
|---|---|
| Frontend UI | React 18 + Tailwind CSS |
| 3D Engine | React Three Fiber (R3F) + Three.js |
| State | Zustand (trim-line undo/redo) |
| Backend API | FastAPI + Uvicorn (Python 3.11) |
| Geometry Engine | Trimesh + NumPy + SciPy + Shapely |
| Real-time | WebSocket (job progress) |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Containerisation | Docker + docker-compose |

---

## Architecture Overview

```
Browser (React + R3F)
  ├── Patient intake & records
  ├── STL upload & preview (decimated to 150k faces)
  ├── 3D viewport — orbit, pan, zoom, measurement overlays
  ├── Trim-line raycasting (Catmull-Rom spline) + undo/redo
  └── Parameters panel → dispatch geometry job

FastAPI Backend
  ├── POST /api/scans/upload          — ingest STL from Creality Raptor
  ├── POST /api/geometry/process      — dispatch geometry job → job_id
  ├── WS   /ws/{job_id}               — real-time progress feed
  ├── GET  /api/geometry/result/{id}  — poll for result
  ├── GET  /api/geometry/download/{id}— download socket STL
  └── CRUD /api/patients/             — patient records

Geometry Engine (trimesh + numpy + scipy)
  ├── Phase 1: PCA auto-alignment (long axis → Z)
  ├── Phase 2: Catmull-Rom trim-line spline fitting
  ├── Phase 3: Boolean slice + variable-thickness shell offset
  └── Phase 4: Circumference validation + manifold repair + STL export

Hardware
  ├── Scanner : Creality Raptor (Blue Laser / NIR mode)
  ├── Printer : Elegoo Neptune 4 Max (Klipper firmware)
  └── Material: PETG or Carbon Fiber PLA
```

---

## Quick Start

### 1. Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 2. Frontend

```bash
cd frontend
npm install
npm run dev          # starts on http://localhost:3000
```

### 3. Docker (recommended)

```bash
docker-compose up --build
```

---

## Workflow Steps

| Step | Description |
|---|---|
| 01 Scan | Upload STL from Creality Raptor; auto-decimated to 150k faces |
| 02 Align | PCA auto-alignment (long axis → Z), manual fine-tune |
| 03 Trim-line | Click 6+ points on 3D mesh → Catmull-Rom spline |
| 04 Design | Set wall thickness, proximal offset, infill → run geometry engine |
| 05 Validate | Circumference check (digital vs tape), manifold status |
| 06 Print | Clinical sign-off → export STL + Elegoo Neptune 4 Max Klipper config |

---

## Improvements over original design doc

1. **Patient management layer** — full CRUD for patient records, scan history
2. **Trim-line undo/redo** — full command history stack (Zustand store)
3. **WebSocket progress** — real-time geometry job feedback to UI
4. **Configurable decimation** — threshold is a tunable parameter, default 150k
5. **Clinical sign-off step** — explicit approval gate before print dispatch
6. **Klipper config export** — Neptune 4 Max profiles bundled with STL
7. **Docker deployment** — single `docker-compose up` for full stack
8. **Async job queue** — geometry runs in background; UI stays responsive

---

## Key Files

```
orthoprint/
├── backend/
│   ├── main.py                  # FastAPI app, WebSocket, scan upload
│   ├── geometry/engine.py       # All 4 geometry phases
│   ├── api/geometry.py          # Geometry job router
│   ├── api/patients.py          # Patient records router
│   ├── api/jobs.py              # Job registry
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Routes
│   │   ├── components/
│   │   │   └── Viewport3D.jsx   # R3F canvas + raycasting + trim overlay
│   │   ├── pages/
│   │   │   └── DesignPage.jsx   # Parameters + job dispatch
│   │   ├── hooks/
│   │   │   └── useGeometryJob.js # WS-based job hook
│   │   └── store/
│   │       └── trimLineStore.js  # Zustand trim-line + undo/redo
│   └── package.json
└── docker-compose.yml
```
