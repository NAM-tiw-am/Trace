from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base, get_vector_type

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    photo_id = Column(Integer, ForeignKey("person_photos.id", ondelete="CASCADE"), nullable=True, index=True)
    embedding = Column(get_vector_type(512), nullable=False)
    model_name = Column(String(100), default="buffalo_l/w600k_r50")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    person = relationship("Person", back_populates="embeddings")
    photo = relationship("PersonPhoto", back_populates="embeddings")
