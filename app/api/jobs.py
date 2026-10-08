from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.processing_job import ProcessingJob
from app.schemas.processing_job import ProcessingJobResponse

router = APIRouter(prefix="/jobs", tags=["Jobs"])

@router.get("", response_model=List[ProcessingJobResponse])
def list_jobs(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """List recent video processing jobs."""
    return db.query(ProcessingJob).order_by(ProcessingJob.id.desc()).offset(skip).limit(limit).all()

@router.get("/{id}", response_model=ProcessingJobResponse)
def get_job(id: int, db: Session = Depends(get_db)):
    """Get status and progress of a processing job."""
    job = db.query(ProcessingJob).filter(ProcessingJob.id == id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found.")
    return job
