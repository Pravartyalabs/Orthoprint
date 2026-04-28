"""
OrthoPrint — /api/cases router
Full case lifecycle: assessment → scan → design → review → approved → fabrication → delivered

Tracks design sessions, version history, assigned users, and fabrication readiness.
"""
import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.auth import get_current_user

router = APIRouter()

# In-memory — replace with PostgreSQL in production
_cases: dict[str, dict] = {}

VALID_TRANSITIONS = {
    "assessment":  ["scan-ready"],
    "scan-ready":  ["in-design", "assessment"],
    "in-design":   ["review", "scan-ready"],
    "review":      ["approved", "in-design"],
    "approved":    ["fabrication", "review"],
    "fabrication": ["delivered", "approved"],
    "delivered":   ["follow-up"],
    "follow-up":   [],
}


class CaseCreate(BaseModel):
    patient_id: str
    device_type: str
    scan_id: Optional[str] = None
    assigned_designer: Optional[str] = None
    assigned_clinician: Optional[str] = None
    priority: str = "routine"
    notes: Optional[str] = None


class CaseNote(BaseModel):
    note: str
    author: str


@router.post("/", status_code=201)
def create_case(data: CaseCreate, user: dict = Depends(get_current_user)):
    case_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    case = {
        "case_id":            case_id,
        "patient_id":         data.patient_id,
        "device_type":        data.device_type,
        "scan_id":            data.scan_id,
        "status":             "assessment",
        "priority":           data.priority,
        "assigned_designer":  data.assigned_designer,
        "assigned_clinician": data.assigned_clinician,
        "created_by":         user["user_id"],
        "created_at":         now,
        "updated_at":         now,
        "design_job_id":      None,
        "approved_by":        None,
        "approved_at":        None,
        "notes":              [{"note": data.notes, "author": user["name"], "at": now}] if data.notes else [],
        "status_history":     [{"status": "assessment", "by": user["name"], "at": now}],
    }
    _cases[case_id] = case
    return case


@router.get("/")
def list_cases(
    status: Optional[str] = None,
    patient_id: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    result = list(_cases.values())
    # Fabricators only see fabrication/delivered cases
    if user["role"] == "fabricator":
        result = [c for c in result if c["status"] in ("fabrication", "delivered")]
    if status:
        result = [c for c in result if c["status"] == status]
    if patient_id:
        result = [c for c in result if c["patient_id"] == patient_id]
    return result


@router.get("/{case_id}")
def get_case(case_id: str, user: dict = Depends(get_current_user)):
    case = _cases.get(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    return case


@router.patch("/{case_id}/status")
def advance_status(
    case_id: str,
    new_status: str,
    user: dict = Depends(get_current_user),
):
    """Advance case through the workflow. Validates allowed transitions."""
    case = _cases.get(case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    current = case["status"]
    allowed = VALID_TRANSITIONS.get(current, [])
    if new_status not in allowed:
        raise HTTPException(
            400,
            f"Cannot move from '{current}' to '{new_status}'. Allowed next states: {allowed}"
        )

    now = datetime.utcnow().isoformat()
    case["status"]     = new_status
    case["updated_at"] = now
    case["status_history"].append({"status": new_status, "by": user["name"], "at": now})

    # Record sign-off metadata when approved
    if new_status == "approved":
        case["approved_by"] = user["name"]
        case["approved_at"] = now

    return case


@router.patch("/{case_id}/assign")
def assign_case(
    case_id: str,
    designer: Optional[str] = None,
    clinician: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    """Assign a designer and/or clinician to a case."""
    case = _cases.get(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    if designer:
        case["assigned_designer"] = designer
    if clinician:
        case["assigned_clinician"] = clinician
    case["updated_at"] = datetime.utcnow().isoformat()
    return case


@router.patch("/{case_id}/scan")
def attach_scan_to_case(
    case_id: str,
    scan_id: str,
    user: dict = Depends(get_current_user),
):
    case = _cases.get(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    case["scan_id"]    = scan_id
    case["status"]     = "scan-ready"
    case["updated_at"] = datetime.utcnow().isoformat()
    case["status_history"].append(
        {"status": "scan-ready", "by": user["name"], "at": case["updated_at"]}
    )
    return case


@router.patch("/{case_id}/design-job")
def attach_design_job(
    case_id: str,
    job_id: str,
    user: dict = Depends(get_current_user),
):
    case = _cases.get(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    case["design_job_id"] = job_id
    case["status"]        = "review"
    case["updated_at"]    = datetime.utcnow().isoformat()
    case["status_history"].append(
        {"status": "review", "by": user["name"], "at": case["updated_at"]}
    )
    return case


@router.post("/{case_id}/notes")
def add_note(
    case_id: str,
    note: CaseNote,
    user: dict = Depends(get_current_user),
):
    case = _cases.get(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    case["notes"].append({
        "note":   note.note,
        "author": note.author or user["name"],
        "at":     datetime.utcnow().isoformat(),
    })
    return {"ok": True, "note_count": len(case["notes"])}


@router.get("/stats/summary")
def case_stats(user: dict = Depends(get_current_user)):
    """Dashboard summary: counts per status."""
    counts: dict[str, int] = {}
    for case in _cases.values():
        s = case["status"]
        counts[s] = counts.get(s, 0) + 1
    return {
        "total": len(_cases),
        "by_status": counts,
        "fabrication_queue": sum(
            1 for c in _cases.values() if c["status"] == "fabrication"
        ),
    }
