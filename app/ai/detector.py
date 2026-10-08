from __future__ import annotations
import logging
from pathlib import Path
from typing import Sequence, Optional
import cv2
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

class FaceDetector:
    """InsightFace RetinaFace wrapper."""

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        min_confidence: float = 0.5,
        model_name: str = "buffalo_l",
    ) -> None:
        self._det_size = det_size
        self._min_confidence = min_confidence
        self._model_name = model_name
        self._app: Optional[FaceAnalysis] = None
        self._initialise()

    def _initialise(self) -> None:
        logger.info("Initializing FaceDetector with model '%s' ...", self._model_name)
        # Try CPU/CUDA gracefully
        for ctx_id in [0, -1]:
            try:
                app = FaceAnalysis(name=self._model_name)
                app.prepare(ctx_id=ctx_id, det_size=self._det_size)
                self._app = app
                logger.info("FaceDetector initialized successfully (ctx_id=%d)", ctx_id)
                return
            except Exception as e:
                logger.warning("Failed to initialize FaceAnalysis with ctx_id=%d: %s", ctx_id, e)

        raise RuntimeError("Failed to initialize FaceAnalysis with both GPU and CPU.")

    def detect(self, image: np.ndarray) -> list[dict]:
        """Detect faces in BGR image array."""
        if image is None or image.size == 0:
            return []

        raw_faces = self._app.get(image)
        results = []
        for face in raw_faces:
            conf = float(face.det_score) if hasattr(face, "det_score") else 1.0
            if conf < self._min_confidence:
                continue

            bbox = face.bbox.astype(int).tolist() if hasattr(face, "bbox") else [0, 0, 0, 0]
            # Clip bbox to image dimensions
            h, w = image.shape[:2]
            x1 = max(0, min(w - 1, bbox[0]))
            y1 = max(0, min(h - 1, bbox[1]))
            x2 = max(0, min(w, bbox[2]))
            y2 = max(0, min(h, bbox[3]))

            embedding = face.embedding if hasattr(face, "embedding") else None
            if embedding is not None:
                # Ensure unit L2 normalization
                norm = np.linalg.norm(embedding)
                if norm > 1e-6:
                    embedding = embedding / norm

            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "landmarks": face.kps.tolist() if hasattr(face, "kps") and face.kps is not None else [],
                "embedding": embedding,
                "raw_face": face,
            })

        return results

    def detect_from_path(self, path: str | Path) -> list[dict]:
        """Load image from disk and detect faces."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Could not decode image: {p}")
        return self.detect(img)
