from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    description = Column(Text, nullable=True)
    last_known_location = Column(String(255), nullable=True)
    last_known_time = Column(DateTime, nullable=True)
    status = Column(String(50), default="MISSING", index=True)  # MISSING, FOUND, INACTIVE
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    case = relationship("Case", back_populates="persons")
    photos = relationship("PersonPhoto", back_populates="person", cascade="all, delete-orphan")
    embeddings = relationship("FaceEmbedding", back_populates="person", cascade="all, delete-orphan")
    sightings = relationship("Sighting", back_populates="person", cascade="all, delete-orphan")
