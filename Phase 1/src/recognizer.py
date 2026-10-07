"""
recognizer.py -- Face Recognition / Embedding Module for the Missing-Person
Identification Pipeline.

This module provides the ``FaceRecognizer`` class, which turns a detected face
into a numerical embedding vector using the ArcFace recognition model bundled
in InsightFace's ``buffalo_l`` model pack (``w600k_r50``).

It sits at the second stage of the pipeline::

    Image -> Face Detection -> Detected Face -> ArcFace -> Embedding Vector

This module is responsible ONLY for embedding generation.  It does NOT
perform similarity comparison or threshold-based matching -- that logic
belongs to a separate ``similarity.py`` module downstream.

Typical usage::

    from recognizer import FaceRecognizer

    recognizer = FaceRecognizer()
    embedding = recognizer.get_embedding_from_path("photo.jpg")

    if embedding is not None:
        print(embedding.shape)   # e.g. (512,)
        print(np.linalg.norm(embedding))  # ~1.0 (L2-normalised)

Embedding normalisation
-----------------------
The ``w600k_r50`` ArcFace model in ``buffalo_l`` returns embeddings that are
**already L2-normalised** (norm ~= 1.0).  This class verifies that at runtime
and, as a safety net, explicitly re-normalises any embedding whose L2 norm
deviates from 1.0 by more than a small tolerance (1e-3).  Downstream code
can therefore safely use **cosine similarity** (equivalent to a dot product
on unit vectors) or **Euclidean distance** for comparison.

Author:  Phase 1 -- Face Recognition Team
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

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
# FaceRecognizer
# ---------------------------------------------------------------------------
class FaceRecognizer:
    """Generate ArcFace embedding vectors from face images.

    Wraps InsightFace's ``FaceAnalysis`` application (``buffalo_l``) and its
    bundled ``w600k_r50`` recognition model to produce a fixed-length 1-D
    embedding vector for any detected face.

    Parameters
    ----------
    detector : object or None
        An optional pre-existing ``FaceDetector`` instance (from
        ``detector.py``) to reuse for face detection.  If ``None``
        (the default), the recognizer lazily creates its own
        ``FaceDetector`` instance when needed.
    model_name : str
        InsightFace model pack name.  Default ``"buffalo_l"``.
    det_size : tuple[int, int]
        Detection input size for the internal ``FaceAnalysis`` app.
        Default ``(640, 640)``.

    Raises
    ------
    RuntimeError
        If InsightFace fails to initialise on both CUDA and CPU providers.

    Notes
    -----
    *   The class maintains its own ``FaceAnalysis`` instance specifically
        for embedding generation.  This is separate from any ``FaceDetector``
        instance used for detection because the two have different usage
        patterns (detection returns bounding boxes; embedding requires the
        full ``app.get()`` pipeline which internally detects + aligns +
        embeds).
    *   Embeddings are **L2-normalised** before being returned.  The
        ``w600k_r50`` model normally outputs normalised vectors, but this
        class applies an explicit safety-net normalisation.

    Examples
    --------
    >>> recognizer = FaceRecognizer()
    >>> emb = recognizer.get_embedding_from_path("test.jpg")
    >>> emb.shape
    (512,)
    >>> round(float(np.linalg.norm(emb)), 1)
    1.0
    """

    # Tolerance for checking whether a vector is already unit-length.
    _NORM_TOLERANCE: float = 1e-3

    def __init__(
        self,
        detector: object | None = None,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._model_name = model_name
        self._det_size = det_size
        self._provider: str = "unknown"

        # Internal FaceAnalysis app used for embedding extraction.
        self._app = self._init_face_analysis()

        # External or lazily-created FaceDetector for path-based helpers.
        self._detector = detector
        self._detector_initialised = detector is not None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_face_analysis(self) -> FaceAnalysis:
        """Create and prepare the ``FaceAnalysis`` application.

        Uses ``onnxruntime.get_available_providers()`` to determine whether
        CUDA is genuinely available before attempting GPU initialisation.

        Returns
        -------
        FaceAnalysis
            A prepared ``FaceAnalysis`` instance.

        Raises
        ------
        RuntimeError
            If initialisation fails on all providers.
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
                    "FaceRecognizer initialised with CUDAExecutionProvider "
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
                "FaceRecognizer initialised with CPUExecutionProvider "
                "(CPU-only mode)."
            )
            return app
        except Exception as cpu_err:
            raise RuntimeError(
                "Failed to initialise InsightFace on both CUDA and CPU "
                f"providers. CPU error: {cpu_err}"
            ) from cpu_err

    def _get_detector(self):
        """Return the ``FaceDetector`` instance, lazily creating one if needed.

        The import is deferred so that ``recognizer.py`` does not hard-fail
        at import time if ``detector.py`` is temporarily unavailable.

        Returns
        -------
        FaceDetector
            A ready-to-use detector instance.
        """
        if not self._detector_initialised:
            from detector import FaceDetector
            self._detector = FaceDetector()
            self._detector_initialised = True
            logger.info(
                "FaceRecognizer created its own FaceDetector instance "
                "(provider: %s).",
                self._detector.provider,
            )
        return self._detector

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def provider(self) -> str:
        """Return the ONNX execution provider currently in use."""
        return self._provider

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        """Convert a grayscale / single-channel / BGRA image to 3-ch BGR."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        """Load an image from disk with validation.

        Parameters
        ----------
        image_path : str
            Filesystem path to the image.

        Returns
        -------
        np.ndarray
            BGR image.

        Raises
        ------
        ImageLoadError
            If the file does not exist, is not a file, or cannot be decoded.
        """
        path = Path(image_path)
        if not path.exists():
            raise ImageLoadError(f"Image file not found: '{image_path}'")
        if not path.is_file():
            raise ImageLoadError(f"Path is not a regular file: '{image_path}'")

        image = cv2.imread(str(path))
        if image is None:
            raise ImageLoadError(
                f"Failed to decode image (file may be corrupt or in an "
                f"unsupported format): '{image_path}'"
            )
        return image

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    @classmethod
    def _normalise(cls, embedding: np.ndarray) -> np.ndarray:
        """L2-normalise an embedding vector.

        The ``w600k_r50`` model in ``buffalo_l`` typically returns vectors
        that are already unit-length.  This method acts as a safety net:
        if the norm is close to 1.0 it logs at DEBUG level; otherwise it
        re-normalises and logs at INFO level.

        Parameters
        ----------
        embedding : np.ndarray
            Raw 1-D embedding vector.

        Returns
        -------
        np.ndarray
            L2-normalised 1-D embedding (norm == 1.0).
        """
        norm = float(np.linalg.norm(embedding))
        if norm < 1e-10:
            logger.warning(
                "Embedding has near-zero L2 norm (%.6e); "
                "cannot normalise. Returning raw vector.",
                norm,
            )
            return embedding

        if abs(norm - 1.0) <= cls._NORM_TOLERANCE:
            logger.debug(
                "Embedding already L2-normalised (norm=%.6f).", norm
            )
        else:
            logger.info(
                "Embedding L2 norm is %.6f (not unit-length); "
                "re-normalising.", norm
            )

        return embedding / norm

    # ------------------------------------------------------------------
    # Core embedding generation
    # ------------------------------------------------------------------
    def get_embedding(
        self,
        image: np.ndarray,
        bbox: list | None = None,
    ) -> np.ndarray | None:
        """Generate an ArcFace embedding vector for a single face.

        Flexible input handling:

        *   If *bbox* is provided (``[x1, y1, x2, y2]``), *image* is
            treated as a **full, uncropped** image.  Detection is run on
            the full image and the detected face with the highest IoU
            overlap with the given bbox is selected for embedding.
        *   If *bbox* is ``None``, *image* is treated as a standalone
            image (either a pre-cropped face or a scene with one face).
            Detection runs on the image as-is and the largest face is
            used if multiple are detected.

        In all cases, ``app.get()`` runs on the **full image** so that
        InsightFace has sufficient context for detection and alignment,
        which is critical for small or low-resolution inputs.

        Parameters
        ----------
        image : np.ndarray
            BGR image -- either a full scene (when *bbox* is given) or
            an already-cropped face.
        bbox : list or None
            Optional ``[x1, y1, x2, y2]`` bounding box as ``int`` pixel
            coordinates.  When provided, the method selects the detected
            face that best overlaps with this bbox (by IoU).

        Returns
        -------
        np.ndarray or None
            A 1-D **L2-normalised** embedding vector (e.g. shape
            ``(512,)`` for ``w600k_r50``).  The exact dimensionality is
            read from the model output, not hardcoded.

            Returns ``None`` if no usable face could be found in the
            input (e.g. the crop was too small or distorted for the
            model to detect a face).

        Notes
        -----
        *   The returned embedding is always L2-normalised (unit length),
            so **cosine similarity** is equivalent to a simple dot product
            and **Euclidean distance** is valid for comparison.
        *   Very low-resolution or low-quality crops will not crash this
            method, but the resulting embedding quality may be poor.
        """
        image = self._ensure_bgr(image)

        # Run InsightFace's full pipeline (detect + align + embed) on the
        # complete image.  This gives InsightFace the full context for
        # detection and alignment, which is critical for small or
        # low-resolution images where a tight bbox crop would be too
        # small for the model to re-detect.
        try:
            faces = self._app.get(image)
        except Exception:
            logger.exception(
                "InsightFace inference failed during embedding extraction."
            )
            return None

        if not faces:
            logger.warning(
                "No face detected in the provided image. "
                "Cannot generate embedding."
            )
            return None

        # Select the target face.
        if bbox is not None:
            # A bbox was provided -- pick the detected face that overlaps
            # most with it (by IoU) so the caller gets the embedding for
            # the specific face they pointed at.
            primary = self._match_face_to_bbox(faces, bbox)
            if primary is None:
                logger.warning(
                    "None of the %d detected faces overlap with the "
                    "requested bbox %s. Returning None.",
                    len(faces),
                    bbox,
                )
                return None
        elif len(faces) > 1:
            # No bbox and multiple faces -- pick the largest.
            logger.info(
                "Multiple faces (%d) found in image; selecting the "
                "largest by bbox area.",
                len(faces),
            )
            faces.sort(
                key=lambda f: (
                    (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                ),
                reverse=True,
            )
            primary = faces[0]
        else:
            primary = faces[0]

        raw_embedding: np.ndarray = primary.embedding

        if raw_embedding is None:
            logger.warning(
                "InsightFace returned a face object with no embedding "
                "(recognition model may not have loaded). Returning None."
            )
            return None

        # Flatten to 1-D (should already be, but be safe).
        embedding = raw_embedding.flatten().astype(np.float32)

        # L2-normalise (safety net -- the model usually does this already).
        embedding = self._normalise(embedding)

        logger.debug(
            "Generated embedding: shape=%s, dtype=%s, L2 norm=%.6f.",
            embedding.shape,
            embedding.dtype,
            float(np.linalg.norm(embedding)),
        )
        return embedding

    # ------------------------------------------------------------------
    # Bbox matching helper
    # ------------------------------------------------------------------
    @staticmethod
    def _match_face_to_bbox(faces, bbox: list) -> object | None:
        """Find the detected face that best overlaps with a given bbox.

        Uses Intersection-over-Union (IoU) to match.  Returns ``None``
        if no face has IoU > 0 with the target bbox.

        Parameters
        ----------
        faces : list
            InsightFace face objects with ``.bbox`` attributes.
        bbox : list
            Target ``[x1, y1, x2, y2]`` as ints/floats.

        Returns
        -------
        object or None
            The best-matching InsightFace face object, or ``None``.
        """
        bx1, by1, bx2, by2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        target_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if target_area <= 0:
            return None

        best_face = None
        best_iou = 0.0

        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox[:4]
            # Intersection
            ix1 = max(bx1, float(fx1))
            iy1 = max(by1, float(fy1))
            ix2 = min(bx2, float(fx2))
            iy2 = min(by2, float(fy2))
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            # Union
            face_area = max(0.0, float(fx2 - fx1)) * max(0.0, float(fy2 - fy1))
            union = target_area + face_area - inter
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_face = face

        if best_iou > 0:
            logger.debug(
                "Matched face to bbox %s with IoU=%.4f.",
                bbox, best_iou,
            )
        return best_face

    # ------------------------------------------------------------------
    # Convenience: load-from-path
    # ------------------------------------------------------------------
    def get_embedding_from_path(
        self,
        image_path: str,
    ) -> np.ndarray | None:
        """Load an image from disk, detect the primary face, and return
        its embedding.

        Uses the ``FaceDetector`` (either injected via the constructor or
        lazily created) to find faces, selects the **largest** face by
        bounding-box area if multiple are present, and generates the
        embedding.

        Parameters
        ----------
        image_path : str
            Filesystem path to the image file.

        Returns
        -------
        np.ndarray or None
            L2-normalised embedding vector, or ``None`` if no face was
            found.

        Raises
        ------
        ImageLoadError
            If the file does not exist, is not a file, or cannot be
            decoded by OpenCV.
        """
        image = self._load_image(image_path)
        detector = self._get_detector()

        detections = detector.detect(image)

        if not detections:
            logger.warning(
                "No face detected in '%s'. Returning None.", image_path
            )
            return None

        # Select the largest face by bbox area.
        if len(detections) > 1:
            detections.sort(
                key=lambda d: (
                    (d["bbox"][2] - d["bbox"][0])
                    * (d["bbox"][3] - d["bbox"][1])
                ),
                reverse=True,
            )
            logger.info(
                "Multiple faces (%d) in '%s'; using the largest "
                "(bbox=%s, confidence=%.4f).",
                len(detections),
                image_path,
                detections[0]["bbox"],
                detections[0]["confidence"],
            )

        primary = detections[0]

        # Generate embedding via get_embedding, passing the full image and
        # the detection bbox so alignment is done on the full-resolution
        # image.
        return self.get_embedding(image, bbox=primary["bbox"])

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def get_embeddings_batch(
        self,
        image_paths: list[str],
    ) -> dict[str, np.ndarray | None]:
        """Generate embeddings for a batch of images.

        Processes each image independently.  A failure on one image does
        NOT stop processing of the rest -- errors are caught, logged, and
        the corresponding entry is set to ``None``.

        Parameters
        ----------
        image_paths : list[str]
            List of filesystem paths to image files.

        Returns
        -------
        dict[str, np.ndarray | None]
            Mapping of image path -> embedding (or ``None`` on failure).
        """
        results: dict[str, np.ndarray | None] = {}
        total = len(image_paths)

        for idx, path in enumerate(image_paths, start=1):
            logger.info(
                "Processing image %d/%d: '%s'", idx, total, path
            )
            try:
                embedding = self.get_embedding_from_path(path)
                results[path] = embedding
            except ImageLoadError as err:
                logger.error(
                    "Failed to load image '%s': %s", path, err
                )
                results[path] = None
            except Exception:
                logger.exception(
                    "Unexpected error processing '%s'.", path
                )
                results[path] = None

        succeeded = sum(1 for v in results.values() if v is not None)
        logger.info(
            "Batch complete: %d/%d images produced embeddings.",
            succeeded,
            total,
        )
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Configure root logger for console output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- Determine test image path ----
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        test_image_path = os.path.join(
            "data", "test", "person_01", "image_01.jpg"
        )
        logger.info(
            "No image path supplied via argv.  Using default: %s",
            test_image_path,
        )

    # ---- Initialise recognizer ----
    recognizer = FaceRecognizer()
    logger.info("Execution provider: %s", recognizer.provider)

    # ---- Generate embedding ----
    try:
        embedding = recognizer.get_embedding_from_path(test_image_path)
    except ImageLoadError as err:
        logger.error("Could not load test image: %s", err)
        sys.exit(1)

    # ---- Print results ----
    print(f"\n{'=' * 60}")
    print(f"  Test image : {test_image_path}")
    if embedding is not None:
        l2_norm = float(np.linalg.norm(embedding))
        print(f"  Shape      : {embedding.shape}")
        print(f"  Dtype      : {embedding.dtype}")
        print(f"  L2 norm    : {l2_norm:.6f}")
        print(f"  First 5    : {embedding[:5]}")
        print(f"  Normalised : {'Yes' if abs(l2_norm - 1.0) < 1e-3 else 'No'}")
    else:
        print("  Embedding  : None (no face detected)")
    print(f"{'=' * 60}\n")
