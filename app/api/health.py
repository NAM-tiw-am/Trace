from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.database import get_db, is_sqlite
from app.core.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint confirming database and model availability."""
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy" if "unhealthy" not in db_status else "degraded",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": {
            "status": db_status,
            "engine": "sqlite" if is_sqlite else "postgresql+pgvector",
        },
        "model": settings.INSIGHTFACE_MODEL,
        "thresholds": {
            "match": settings.MATCH_THRESHOLD,
            "high_confidence": settings.HIGH_CONFIDENCE_THRESHOLD,
            "cooldown_seconds": settings.ALERT_COOLDOWN_SECONDS,
        }
    }
