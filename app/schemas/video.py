from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

class VideoBase(BaseModel):
    camera_name: str
    location: Optional[str] = None
    recording_time: Optional[datetime] = None
    case_id: Optional[int] = None

class VideoCreate(VideoBase):
    pass

class VideoResponse(VideoBase):
    id: int
    filename: str
    file_path: str
    fps: float
    total_frames: int
    duration_seconds: float
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
