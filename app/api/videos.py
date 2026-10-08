import os
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.video import Video
from app.models.sighting import Sighting
from app.schemas.video import VideoResponse
from app.schemas.processing_job import ProcessingJobResponse
from app.schemas.sighting import SightingResponse
from app.services.storage_service import StorageService
from app.services.video_service import VideoService
from app.services.processing_service import ProcessingService

router = APIRouter(prefix="/videos", tags=["Videos"])

@router.post("", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
def upload_video(
    camera_name: str = Form(...),
    location: Optional[str] = Form(None),
    case_id: Optional[int] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a CCTV video footage file.
    Stores the video file, extracts frame rate and duration metadata, and registers it.
    """
    filename, file_path = StorageService.save_video(file)

    video = VideoService.register_video(
        db=db,
        camera_name=camera_name,
        filename=filename,
        file_path=file_path,
        location=location,
        recording_time=datetime.utcnow(),
        case_id=case_id,
    )
    return video

@router.get("", response_model=List[VideoResponse])
def list_videos(case_id: Optional[int] = None, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List uploaded CCTV videos."""
    query = db.query(Video)
    if case_id:
        query = query.filter(Video.case_id == case_id)
    return query.offset(skip).limit(limit).all()

@router.get("/{id}", response_model=VideoResponse)
def get_video(id: int, db: Session = Depends(get_db)):
    """Get video details."""
    video = db.query(Video).filter(Video.id == id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")
    return video

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(id: int, db: Session = Depends(get_db)):
    """Delete a video and its file."""
    video = db.query(Video).filter(Video.id == id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")

    if os.path.exists(video.file_path):
        try:
            os.remove(video.file_path)
        except OSError:
            pass

    db.delete(video)
    db.commit()
    return None

@router.post("/{id}/process", response_model=ProcessingJobResponse, status_code=status.HTTP_202_ACCEPTED)
def process_video(
    id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Trigger CCTV video processing job to detect faces and match against registered persons.
    Runs asynchronously in the background.
    """
    video = db.query(Video).filter(Video.id == id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")

    job = ProcessingService.create_job(db=db, video_id=id)

    # Launch processing in background
    background_tasks.add_task(ProcessingService.execute_job, job.id)

    return job

@router.get("/{id}/sightings", response_model=List[SightingResponse])
def get_video_sightings(id: int, db: Session = Depends(get_db)):
    """Retrieve all sightings detected in this CCTV video."""
    video = db.query(Video).filter(Video.id == id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")

    return db.query(Sighting).filter(Sighting.video_id == id).order_by(Sighting.timestamp_seconds.asc()).all()
