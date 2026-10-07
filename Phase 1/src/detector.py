"""
detector.py — Face Detection Module for the Missing-Person Identification Pipeline.

This module provides the ``FaceDetector`` class, which wraps InsightFace's
``FaceAnalysis`` application (``buffalo_l`` model pack) to detect faces in images
and return bounding boxes, confidence scores, landmarks, and cropped face
regions.

It is designed as the first stage of the pipeline:

    Image → FaceDetector → Bounding Box + Confidence + Landmarks → Face Crop

The output is intended for downstream consumption by an ArcFace-based
recognition / embedding module.

Typical usage::

    from detector import FaceDetector

    detector = FaceDetector()
    detections = detector.detect_from_path("photo.jpg")

    for det in detections:
        print(det["confidence"], det["bbox"])

Author:  Phase 1 – Face Detection Team
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class ImageLoadError(Exception):
    """Raised when an image cannot be loaded from disk.

    Possible causes include a missing file, an unreadable format, a corrupt
    file, or insufficient permissions.
    """


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------
class FaceDetector:
    """High-level face detector backed by InsightFace ``FaceAnalysis``.

    The detector automatically attempts to use CUDA (GPU) acceleration and
    falls back to CPU if CUDA is not available or fails to initialise.

    Parameters
    ----------
    det_size : tuple[int, int]
        Detection input size passed to ``FaceAnalysis.prepare``.
        Larger sizes can improve accuracy at the cost of speed.
        Default is ``(640, 640)`` — the standard for ``buffalo_l``.
    min_confidence : float
        Minimum detection confidence score.  Faces scoring below this
        threshold are silently filtered out of results.  Default ``0.5``.
    model_name : str
        InsightFace model pack name.  Default ``"buffalo_l"``.

    Raises
    ------
    RuntimeError
        If InsightFace fails to initialise on *both* CUDA and CPU providers.

    Examples
    --------
    >>> detector = FaceDetector(det_size=(320, 320), min_confidence=0.6)
    >>> faces = detector.detect(cv2.imread("test.jpg"))
    >>> len(faces)
    1
    """

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        min_confidence: float = 0.5,
        model_name: str = "buffalo_l",
    ) -> None:
        self._det_size = det_size
        self._min_confidence = min_confidence
        self._model_name = model_name
        self._provider: str = "unknown"

        self._app = self._init_face_analysis()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_face_analysis(self) -> FaceAnalysis:
        """Create and prepare the ``FaceAnalysis`` application.

        Checks which ONNX Runtime execution providers are genuinely
        available (via ``onnxruntime.get_available_providers()``) and
        selects providers accordingly.  If ``CUDAExecutionProvider`` is
        listed *and* initialisation succeeds, GPU is used; otherwise the
        method falls back to ``CPUExecutionProvider``.

        Returns
        -------
        FaceAnalysis
            A ready-to-use ``FaceAnalysis`` instance.

        Raises
        ------
        RuntimeError
            If initialisation fails on *all* providers.
        """
        import onnxruntime as ort

        available = ort.get_available_providers()
        cuda_available = "CUDAExecutionProvider" in available
        logger.debug("ONNX Runtime available providers: %s", available)

        # --- Attempt CUDA if genuinely present ---
        if cuda_available:
            try:
                app = FaceAnalysis(
                    name=self._model_name,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=self._det_size)
                self._provider = "CUDAExecutionProvider"
                logger.info(
                    "InsightFace initialised with CUDAExecutionProvider "
                    "(GPU acceleration enabled)."
                )
                return app
            except Exception as cuda_err:
                logger.warning(
                    "CUDAExecutionProvider listed but initialisation failed "
                    "(%s). Falling back to CPUExecutionProvider.",
                    cuda_err,
                )
        else:
            logger.info(
                "CUDAExecutionProvider not available in this ONNX Runtime "
                "build. Using CPUExecutionProvider."
            )

        # --- Fallback to CPU-only ---
        try:
            app = FaceAnalysis(
                name=self._model_name,
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=self._det_size)
            self._provider = "CPUExecutionProvider"
            logger.info(
                "InsightFace initialised with CPUExecutionProvider "
                "(CPU-only mode)."
            )
            return app
        except Exception as cpu_err:
            raise RuntimeError(
                "Failed to initialise InsightFace on both CUDA and CPU "
                f"providers. CPU error: {cpu_err}"
            ) from cpu_err

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def provider(self) -> str:
        """Return the ONNX execution provider currently in use."""
        return self._provider

    @property
    def det_size(self) -> tuple[int, int]:
        """Return the configured detection input size."""
        return self._det_size

    @property
    def min_confidence(self) -> float:
        """Return the configured minimum confidence threshold."""
        return self._min_confidence

    # ------------------------------------------------------------------
    # Image pre-processing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        """Convert a grayscale or single-channel image to 3-channel BGR.

        If the image is already 3-channel it is returned unchanged.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or BGR).

        Returns
        -------
        np.ndarray
            3-channel BGR image.
        """
        if image.ndim == 2:
            # Grayscale → BGR
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            # Single-channel with explicit dim → BGR
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            # BGRA → BGR (drop alpha)
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> list[dict]:
        """Detect faces in a BGR OpenCV image.

        Parameters
        ----------
        image : np.ndarray
            A BGR image as returned by ``cv2.imread``.  Grayscale images
            are automatically converted to BGR before detection.

        Returns
        -------
        list[dict]
            A list of detection dictionaries, one per face, sorted by
            confidence (highest first).  Each dict contains:

            - ``bbox`` – ``[x1, y1, x2, y2]`` as Python ``int`` values.
            - ``confidence`` – ``float`` detection score.
            - ``landmarks`` – Landmark array (typically 5×2 for
              ``buffalo_l``) as a ``numpy.ndarray``.
            - ``face_crop`` – The cropped face region as a BGR
              ``numpy.ndarray``.

            Returns an **empty list** (never ``None``) if no face is
            detected above ``min_confidence``.

        Notes
        -----
        Very small or low-resolution images will not crash the detector,
        but detection quality may be significantly reduced.
        """
        image = self._ensure_bgr(image)

        # Run InsightFace detection
        try:
            faces = self._app.get(image)
        except Exception:
            logger.exception(
                "InsightFace inference failed. Returning empty detections."
            )
            return []

        if not faces:
            logger.info("No face detected in the image.")
            return []

        # Build result list, filtering by confidence
        h, w = image.shape[:2]
        results: list[dict] = []

        for face in faces:
            score: float = float(face.det_score)
            if score < self._min_confidence:
                logger.debug(
                    "Skipping face with confidence %.4f (below threshold %.2f).",
                    score,
                    self._min_confidence,
                )
                continue

            # Clamp bounding box to image dimensions
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            # Guard against degenerate bounding boxes
            if x2 <= x1 or y2 <= y1:
                logger.debug(
                    "Skipping degenerate bbox [%d, %d, %d, %d].",
                    x1, y1, x2, y2,
                )
                continue

            face_crop = image[y1:y2, x1:x2].copy()

            # Landmarks: InsightFace stores 5-point kps by default on
            # the ``kps`` attribute (shape 5×2).  Some models also
            # expose ``landmark_2d_106`` for 106-point landmarks.
            landmarks = face.kps  # np.ndarray, shape (5, 2) typically
            if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                landmarks = face.landmark_2d_106  # (106, 2) if available

            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": score,
                    "landmarks": np.asarray(landmarks),
                    "face_crop": face_crop,
                }
            )

        # Sort by confidence, highest first
        results.sort(key=lambda d: d["confidence"], reverse=True)

        logger.info(
            "Detected %d face(s) above confidence threshold %.2f.",
            len(results),
            self._min_confidence,
        )
        return results

    # ------------------------------------------------------------------
    # Convenience: load-from-path
    # ------------------------------------------------------------------
    def detect_from_path(self, image_path: str) -> list[dict]:
        """Load an image from disk and run face detection.

        This is a convenience wrapper around :meth:`detect` that handles
        file validation so callers don't have to deal with ``cv2.imread``
        returning ``None`` on failure.

        Parameters
        ----------
        image_path : str
            Filesystem path to the image file.

        Returns
        -------
        list[dict]
            Same structure as :meth:`detect`.

        Raises
        ------
        ImageLoadError
            If the file does not exist, is not a file, or cannot be
            decoded by OpenCV.
        """
        path = Path(image_path)

        if not path.exists():
            raise ImageLoadError(
                f"Image file not found: '{image_path}'"
            )
        if not path.is_file():
            raise ImageLoadError(
                f"Path is not a regular file: '{image_path}'"
            )

        image = cv2.imread(str(path))

        if image is None:
            raise ImageLoadError(
                f"Failed to decode image (file may be corrupt or in an "
                f"unsupported format): '{image_path}'"
            )

        return self.detect(image)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def draw_detections(
        self,
        image: np.ndarray,
        detections: list[dict],
    ) -> np.ndarray:
        """Draw bounding boxes, confidence scores, and landmarks on an image.

        Useful for visual debugging and demo purposes.

        Parameters
        ----------
        image : np.ndarray
            The original BGR image (will **not** be modified in-place).
        detections : list[dict]
            Detection dicts as returned by :meth:`detect`.

        Returns
        -------
        np.ndarray
            A **copy** of the image with annotations drawn.
        """
        canvas = image.copy()
        canvas = self._ensure_bgr(canvas)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            landmarks = det["landmarks"]

            # -- Bounding box (green) --
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # -- Label with index and confidence --
            label = f"Face {i + 1}: {confidence:.3f}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            # Draw a filled rectangle behind the label for readability
            cv2.rectangle(
                canvas,
                (x1, y1 - label_size[1] - baseline - 4),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                cv2.FILLED,
            )
            cv2.putText(
                canvas,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # -- Landmarks (small coloured circles) --
            if landmarks is not None and len(landmarks) > 0:
                # Use distinct colours for the first 5 canonical landmarks
                # (left_eye, right_eye, nose, left_mouth, right_mouth)
                landmark_colours: list[tuple[int, int, int]] = [
                    (255, 0, 0),    # blue  — left eye
                    (0, 0, 255),    # red   — right eye
                    (0, 255, 0),    # green — nose tip
                    (255, 255, 0),  # cyan  — left mouth corner
                    (0, 255, 255),  # yellow — right mouth corner
                ]
                for idx, point in enumerate(landmarks):
                    colour = landmark_colours[idx % len(landmark_colours)]
                    cx, cy = int(point[0]), int(point[1])
                    cv2.circle(canvas, (cx, cy), 3, colour, -1)

        return canvas


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Configure root logger for console output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- Determine test image path ----
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        # Hardcoded placeholder — adjust as needed for quick testing
        test_image_path = os.path.join(
            "data", "test", "person_01", "image_01.jpg"
        )
        logger.info(
            "No image path supplied via argv.  Using default: %s",
            test_image_path,
        )

    # ---- Initialise detector ----
    detector = FaceDetector()
    logger.info("Execution provider: %s", detector.provider)

    # ---- Run detection ----
    try:
        detections = detector.detect_from_path(test_image_path)
    except ImageLoadError as err:
        logger.error("Could not load test image: %s", err)
        sys.exit(1)

    # ---- Print results ----
    print(f"\n{'='*60}")
    print(f"  Test image : {test_image_path}")
    print(f"  Faces found: {len(detections)}")
    print(f"{'='*60}")
    for i, det in enumerate(detections):
        print(
            f"  Face {i + 1}  |  confidence: {det['confidence']:.4f}  "
            f"|  bbox: {det['bbox']}  "
            f"|  landmarks shape: {det['landmarks'].shape}"
        )
    if not detections:
        print("  (no faces detected)")
    print(f"{'='*60}\n")

    # ---- Save annotated image ----
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "detection_test_output.jpg"

    image = cv2.imread(test_image_path)
    annotated = detector.draw_detections(image, detections)
    cv2.imwrite(str(output_path), annotated)
    logger.info("Annotated image saved to %s", output_path)
