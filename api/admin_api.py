"""Operational endpoints.

Foreign keys and single-transaction writes rule out orphaned clips or
embeddings. What remains worth checking is a person with no embedding, or a
clip whose embedding is missing.
"""

from fastapi import APIRouter, HTTPException

from src.voice_service import integrity_report

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/integrity")
def check_integrity():
    """Read-only report of incomplete records."""
    try:
        report = integrity_report()
        return {
            "success": True,
            "message": "Consistent." if report["consistent"] else "Incomplete records found.",
            "data": report,
            "errors": [],
            **report,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Integrity check failed: {exc}")
