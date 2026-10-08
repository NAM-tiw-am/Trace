from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from app.ai.detector import FaceDetector

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """Generate 512-D ArcFace embedding vectors from face images."""

    def __init__(self, detector: Optional[FaceDetector] = None) -> None:
        self._detector = detector or FaceDetector()

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding of the most prominent face from an image array."""
        detections = self._detector.detect(image)
        if not detections:
            return None

        # Choose face with largest bounding box area
        best_face = max(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
        )

        embedding = best_face.get("embedding")
        if embedding is None and "raw_face" in best_face:
            embedding = best_face["raw_face"].embedding

        if embedding is not None:
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = (embedding / norm).astype(np.float32)

        return embedding

    def get_embedding_from_path(self, path: str | Path) -> Optional[np.ndarray]:
        """Load image from path and return normalized embedding."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Could not decode image: {p}")
        return self.get_embedding(img)
