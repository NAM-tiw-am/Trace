import logging
from pathlib import Path
from typing import Optional
import numpy as np
from sqlalchemy.orm import Session
from app.ai.detector import FaceDetector
from app.ai.recognizer import FaceRecognizer
from app.models.embedding import FaceEmbedding
from app.core.config import settings

logger = logging.getLogger(__name__)

class FaceService:
    _detector: Optional[FaceDetector] = None
    _recognizer: Optional[FaceRecognizer] = None

    @classmethod
    def get_detector(cls) -> FaceDetector:
        if cls._detector is None:
            cls._detector = FaceDetector(
                det_size=settings.FACE_DETECTION_SIZE,
                min_confidence=settings.MIN_FACE_CONFIDENCE,
                model_name=settings.INSIGHTFACE_MODEL,
            )
        return cls._detector

    @classmethod
    def get_recognizer(cls) -> FaceRecognizer:
        if cls._recognizer is None:
            cls._recognizer = FaceRecognizer(detector=cls.get_detector())
        return cls._recognizer

    @classmethod
    def process_and_store_photo(
        cls,
        db: Session,
        person_id: int,
        photo_id: int,
        image_path: str,
    ) -> Optional[FaceEmbedding]:
        """Detect face in photo, compute 512-D embedding, and save to DB."""
        recognizer = cls.get_recognizer()
        emb = recognizer.get_embedding_from_path(image_path)
        if emb is None:
            logger.warning("No face detected in photo: %s", image_path)
            return None

        # Convert numpy array to list for database persistence
        emb_list = emb.tolist()
        db_embedding = FaceEmbedding(
            person_id=person_id,
            photo_id=photo_id,
            embedding=emb_list,
            model_name=f"{settings.INSIGHTFACE_MODEL}/w600k_r50",
        )
        db.add(db_embedding)
        db.commit()
        db.refresh(db_embedding)
        logger.info("Saved embedding for person_id=%d from photo_id=%d", person_id, photo_id)
        return db_embedding
