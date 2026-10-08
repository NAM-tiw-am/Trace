from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import relationship
from app.core.database import Base

class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    case_number = Column(String(50), unique=True, index=True, nullable=False)
    title = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="ACTIVE", index=True)  # ACTIVE, CLOSED, ARCHIVED
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    persons = relationship("Person", back_populates="case", cascade="all, delete-orphan")
    videos = relationship("Video", back_populates="case")
