from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

class Sighting(Base):
    __tablename__ = "sightings"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("processing_jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    timestamp_seconds = Column(Float, nullable=False)
    timestamp_formatted = Column(String(50), nullable=False)
    frame_number = Column(Integer, nullable=False)
    similarity_score = Column(Float, nullable=False)
    bbox_x1 = Column(Integer, default=0)
    bbox_y1 = Column(Integer, default=0)
    bbox_x2 = Column(Integer, default=0)
    bbox_y2 = Column(Integer, default=0)
    snapshot_path = Column(String(500), nullable=False)
    match_status = Column(String(50), default="POTENTIAL_MATCH", index=True)  # POTENTIAL_MATCH, REVIEWED, CONFIRMED, REJECTED
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    person = relationship("Person", back_populates="sightings")
    video = relationship("Video", back_populates="sightings")
    job = relationship("ProcessingJob", back_populates="sightings")
