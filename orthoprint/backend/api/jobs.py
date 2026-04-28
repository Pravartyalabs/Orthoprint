"""
OrthoPrint — Job registry and status tracking
"""
from enum import Enum
from fastapi import APIRouter

router = APIRouter()

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

# Simple in-memory registry; swap for Redis in production
job_registry: dict[str, dict] = {}

@router.get("/")
def list_jobs():
    return [
        {"job_id": jid, "status": info["status"]}
        for jid, info in job_registry.items()
    ]
