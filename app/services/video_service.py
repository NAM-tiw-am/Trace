import logging
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from app.models.video import Video
from app.ai.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

class VideoService:
    @staticmethod
    def register_video(
        db: Session,
        camera_name: str,
        filename: str,
        file_path: str,
        location: Optional[str] = None,
        recording_time: Optional[datetime] = None,
        case_id: Optional[int] = None,
    ) -> Video:
        """Inspect video metadata and create Video record."""
        fps = 25.0
        total_frames = 0
        duration = 0.0

        try:
            with VideoProcessor(file_path) as vp:
                fps = vp.fps
                total_frames = vp.total_frames
                duration = vp.duration
        except Exception as e:
            logger.warning("Could not read video metadata via OpenCV: %s", e)

        video = Video(
            case_id=case_id,
            camera_name=camera_name,
            location=location,
            recording_time=recording_time or datetime.utcnow(),
            filename=filename,
            file_path=file_path,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            status="UPLOADED",
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        return video
