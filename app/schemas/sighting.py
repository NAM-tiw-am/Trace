from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict
from app.schemas.person import PersonResponse
from app.schemas.video import VideoResponse

class SightingBase(BaseModel):
    person_id: int
    video_id: int
    job_id: Optional[int] = None
    timestamp_seconds: float
    timestamp_formatted: str
    frame_number: int
    similarity_score: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    snapshot_path: str
    match_status: Optional[str] = "POTENTIAL_MATCH"
    notes: Optional[str] = None

class SightingCreate(SightingBase):
    pass

class SightingStatusUpdate(BaseModel):
    match_status: str  # POTENTIAL_MATCH, REVIEWED, CONFIRMED, REJECTED
    notes: Optional[str] = None

class SightingResponse(SightingBase):
    id: int
    created_at: datetime
    person: Optional[PersonResponse] = None
    video: Optional[VideoResponse] = None

    model_config = ConfigDict(from_attributes=True)
