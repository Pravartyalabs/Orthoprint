"""
OrthoPrint — /api/patients router
Full clinical patient record per requirements document.
"""
import uuid
from datetime import datetime
from typing import Optional, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# In-memory store — replace with SQLAlchemy + PostgreSQL in production
_patients: dict[str, dict] = {}


# ─── Enums / Literals ─────────────────────────────────────────────────────────

AmputationSide  = Literal["left", "right", "bilateral"]
AmputationLevel = Literal[
    "below-knee", "above-knee",
    "below-elbow", "above-elbow",
    "partial-foot", "hip-disarticulation",
    "shoulder-disarticulation",
]
ActivityLevel   = Literal["low", "moderate", "high", "variable"]
SuspensionType  = Literal[
    "pin-lock", "vacuum", "seal-in", "sleeve",
    "suction", "strap", "harness", "other",
]
SoftTissueCondition = Literal[
    "good", "moderate-atrophy", "severe-atrophy",
    "oedematous", "scarring", "bony-prominence", "mixed",
]
CaseStatus = Literal[
    "assessment",   # patient registered, awaiting scan
    "scan-ready",   # scan uploaded, not yet cleaned
    "in-design",    # design workflow in progress
    "review",       # design complete, awaiting clinical review
    "approved",     # clinically signed off
    "fabrication",  # sent to printer/fabrication
    "delivered",    # device delivered to patient
    "follow-up",    # post-delivery review
]


# ─── Models ───────────────────────────────────────────────────────────────────

class ClinicalAssessment(BaseModel):
    """BK/AK-specific clinical inputs required for AI-assisted design."""
    residual_limb_length_mm: Optional[float] = Field(
        None, description="Measured residual limb length in mm"
    )
    body_weight_kg: Optional[float] = Field(
        None, ge=1, le=300, description="Patient body weight in kg"
    )
    activity_level: Optional[ActivityLevel] = None
    suspension_type: Optional[SuspensionType] = None
    soft_tissue_condition: Optional[SoftTissueCondition] = None
    # Circumference measurements (tape) at key Z-heights
    circumference_measurements: Optional[dict[str, float]] = Field(
        None,
        description="Tape measurements keyed by label e.g. {'patella_tendon': 280.0, 'mid_tibia': 260.0}"
    )
    liner_thickness_mm: Optional[float] = Field(
        None, description="Liner thickness in mm (affects socket clearance)"
    )
    volume_reduction_pct: Optional[float] = Field(
        None, ge=-20, le=20,
        description="Global volume adjustment % — negative = reduce, positive = expand"
    )
    preferred_wall_thickness_mm: Optional[float] = Field(
        4.0, ge=3.0, le=6.0,
        description="Preferred socket wall thickness (3, 4, 5, or 6 mm)"
    )


class PatientCreate(BaseModel):
    # Demographics
    name: str = Field(..., min_length=2)
    date_of_birth: Optional[str] = Field(None, description="ISO date YYYY-MM-DD")
    age: int = Field(..., ge=0, le=120)
    sex: Literal["M", "F", "Other"]
    # Device prescription
    device_type: Literal[
        "prosthetic-socket", "insole", "afo-smo",
        "spinal-brace", "cranial-helmet", "liner-mould",
    ] = "prosthetic-socket"
    amputation_side: AmputationSide
    amputation_level: AmputationLevel
    # Clinical assessment
    clinical: Optional[ClinicalAssessment] = None
    # Admin
    referring_clinician: Optional[str] = None
    clinician_notes: Optional[str] = None
    priority: Literal["routine", "urgent", "emergency"] = "routine"


class PatientUpdate(BaseModel):
    """Partial update — all fields optional."""
    clinical: Optional[ClinicalAssessment] = None
    clinician_notes: Optional[str] = None
    priority: Optional[Literal["routine", "urgent", "emergency"]] = None
    status: Optional[CaseStatus] = None
    referring_clinician: Optional[str] = None


class PatientOut(PatientCreate):
    patient_id: str
    created_at: str
    updated_at: str
    status: CaseStatus = "assessment"
    scans: list[str] = []
    design_sessions: list[str] = []


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/", response_model=PatientOut, status_code=201)
def create_patient(data: PatientCreate):
    pid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    record = {
        "patient_id": pid,
        "created_at": now,
        "updated_at": now,
        "status": "assessment",
        "scans": [],
        "design_sessions": [],
        **data.model_dump(),
    }
    _patients[pid] = record
    return record


@router.get("/", response_model=list[PatientOut])
def list_patients(
    status: Optional[str] = None,
    device_type: Optional[str] = None,
):
    """List all patients, optionally filtered by status or device type."""
    result = list(_patients.values())
    if status:
        result = [p for p in result if p["status"] == status]
    if device_type:
        result = [p for p in result if p["device_type"] == device_type]
    return result


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str):
    p = _patients.get(patient_id)
    if not p:
        raise HTTPException(404, "Patient not found")
    return p


@router.patch("/{patient_id}", response_model=PatientOut)
def update_patient(patient_id: str, data: PatientUpdate):
    """Partial update — update clinical data, status, notes."""
    p = _patients.get(patient_id)
    if not p:
        raise HTTPException(404, "Patient not found")
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    p.update(updates)
    p["updated_at"] = datetime.utcnow().isoformat()
    return p


@router.patch("/{patient_id}/status")
def update_status(patient_id: str, status: CaseStatus):
    """Advance the case status in the fabrication workflow."""
    p = _patients.get(patient_id)
    if not p:
        raise HTTPException(404, "Patient not found")
    p["status"] = status
    p["updated_at"] = datetime.utcnow().isoformat()
    return {"patient_id": patient_id, "status": status}


@router.patch("/{patient_id}/scan")
def attach_scan(patient_id: str, scan_id: str):
    p = _patients.get(patient_id)
    if not p:
        raise HTTPException(404, "Patient not found")
    if scan_id not in p["scans"]:
        p["scans"].append(scan_id)
    p["status"] = "scan-ready"
    p["updated_at"] = datetime.utcnow().isoformat()
    return {"ok": True, "scan_count": len(p["scans"])}
