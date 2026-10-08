from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

class ProcessingJobResponse(BaseModel):
    id: int
    video_id: int
    status: str
    progress: float
    current_frame: int
    total_frames: int
    faces_detected: int
    matches_found: int
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
