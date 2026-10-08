from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="SET NULL"), nullable=True, index=True)
    camera_name = Column(String(100), nullable=False)
    location = Column(String(255), nullable=True)
    recording_time = Column(DateTime, nullable=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    fps = Column(Float, default=25.0)
    total_frames = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    status = Column(String(50), default="UPLOADED", index=True)  # UPLOADED, PROCESSING, COMPLETED, FAILED
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    case = relationship("Case", back_populates="videos")
    jobs = relationship("ProcessingJob", back_populates="video", cascade="all, delete-orphan")
    sightings = relationship("Sighting", back_populates="video", cascade="all, delete-orphan")
