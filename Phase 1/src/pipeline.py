"""
pipeline.py -- End-to-End Face Match Pipeline for Phase 1.

This is the final integration module that wires together detection,
recognition, and similarity into one program:

    Image 1 --> Detect Face --> Embedding --.
                                            +--> Cosine Similarity --> Threshold --> Verdict
    Image 2 --> Detect Face --> Embedding --'

Usage (CLI demo)::

    python pipeline.py photo_a.jpg photo_b.jpg
    python pipeline.py photo_a.jpg photo_b.jpg --threshold 0.45

Usage (as a library)::

    from pipeline import FaceMatchPipeline

    pipeline = FaceMatchPipeline()
    result   = pipeline.compare_images("photo_a.jpg", "photo_b.jpg")
    pipeline.print_result(result)

Author:  Phase 1 -- Integration Team
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure sibling modules are importable when run as a script
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from recognizer import FaceRecognizer, ImageLoadError  # noqa: E402
from similarity import FaceSimilarity  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experimentally validated match threshold
# ---------------------------------------------------------------------------
# Derived from a threshold experiment on the Phase 1 test dataset
# (5 persons, 15 images -- 75 genuine pairs, 300 impostor pairs):
#   - Genuine pairs  mean similarity : 0.6874
#   - Impostor pairs mean similarity : 0.0061
#   - Midpoint threshold             : 0.3468
#   - At this threshold: 0 false positives, 97.3% true positive rate
#
# This value is a conservative starting point.  For missing-person systems
# it is better to bias toward fewer false positives (higher threshold)
# than to risk misidentifying someone.  Override via the constructor's
# ``match_threshold`` parameter after future re-calibration.
MATCH_THRESHOLD: float = 0.3468


# ---------------------------------------------------------------------------
# Verdict strings (constants for consistency)
# ---------------------------------------------------------------------------
_VERDICT_MATCH = "Potential Match"
_VERDICT_NO_MATCH = "Unlikely Match"
_VERDICT_ERR_IMG1 = "Error: No face detected in image 1"
_VERDICT_ERR_IMG2 = "Error: No face detected in image 2"
_VERDICT_ERR_BOTH = "Error: No face detected in either image"


# ---------------------------------------------------------------------------
# FaceMatchPipeline
# ---------------------------------------------------------------------------
class FaceMatchPipeline:
    """End-to-end face comparison pipeline for Phase 1.

    Owns a single ``FaceRecognizer`` (which internally owns a
    ``FaceDetector``) and a ``FaceSimilarity`` instance.  Provides a
    clean, crash-safe API that always returns a populated result dict
    rather than raising exceptions for expected failure modes.

    Parameters
    ----------
    match_threshold : float
        Cosine-similarity threshold above which two embeddings are
        considered a potential match.  Default is the experimentally
        validated value from the Phase 1 threshold experiment.

    Examples
    --------
    >>> pipe = FaceMatchPipeline()
    >>> result = pipe.compare_images("a.jpg", "b.jpg")
    >>> result["is_match"]
    True
    >>> result["similarity"]
    0.7523
    """

    def __init__(self, match_threshold: float = MATCH_THRESHOLD) -> None:
        self._threshold = match_threshold

        logger.info("Initialising FaceMatchPipeline ...")
        logger.info("Match threshold: %.4f", self._threshold)

        self._recognizer = FaceRecognizer()
        self._similarity = FaceSimilarity()

        logger.info(
            "Pipeline ready (provider: %s).", self._recognizer.provider
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def threshold(self) -> float:
        """Return the current match threshold."""
        return self._threshold

    @property
    def provider(self) -> str:
        """Return the ONNX execution provider in use."""
        return self._recognizer.provider

    # ------------------------------------------------------------------
    # Result factory
    # ------------------------------------------------------------------
    @staticmethod
    def _make_result(
        image_1: str = "",
        image_2: str = "",
        face_1_detected: bool = False,
        face_2_detected: bool = False,
        similarity: float | None = None,
        threshold_used: float = MATCH_THRESHOLD,
        is_match: bool | None = None,
        verdict: str = "",
        error: str | None = None,
    ) -> dict:
        """Build a standardised result dictionary."""
        return {
            "image_1": image_1,
            "image_2": image_2,
            "face_1_detected": face_1_detected,
            "face_2_detected": face_2_detected,
            "similarity": similarity,
            "threshold_used": threshold_used,
            "is_match": is_match,
            "verdict": verdict,
            "error": error,
        }

    # ------------------------------------------------------------------
    # Core: compare two image paths
    # ------------------------------------------------------------------
    def compare_images(
        self,
        image_path_1: str,
        image_path_2: str,
    ) -> dict:
        """Compare two face images end-to-end.

        Loads each image, detects the primary face, generates an
        L2-normalised embedding, computes cosine similarity, and
        applies the match threshold.

        This method **never raises** for expected failures (missing
        file, no face, corrupt image).  It always returns a fully
        populated result dict with the ``error`` field set when
        something goes wrong.

        Parameters
        ----------
        image_path_1 : str
            Filesystem path to the first image.
        image_path_2 : str
            Filesystem path to the second image.

        Returns
        -------
        dict
            Result dictionary with keys: ``image_1``, ``image_2``,
            ``face_1_detected``, ``face_2_detected``, ``similarity``,
            ``threshold_used``, ``is_match``, ``verdict``, ``error``.
        """
        emb1: np.ndarray | None = None
        emb2: np.ndarray | None = None
        face_1_ok = False
        face_2_ok = False
        error_msg: str | None = None

        # ---- Embedding 1 ----
        try:
            emb1 = self._recognizer.get_embedding_from_path(image_path_1)
            face_1_ok = emb1 is not None
            if not face_1_ok:
                logger.warning(
                    "No face detected in image 1: '%s'", image_path_1
                )
        except ImageLoadError as err:
            logger.error("Image 1 load error: %s", err)
            error_msg = f"Image 1: {err}"
        except Exception:
            logger.exception(
                "Unexpected error processing image 1: '%s'", image_path_1
            )
            error_msg = f"Image 1: unexpected error (see log for details)"

        # ---- Embedding 2 ----
        try:
            emb2 = self._recognizer.get_embedding_from_path(image_path_2)
            face_2_ok = emb2 is not None
            if not face_2_ok:
                logger.warning(
                    "No face detected in image 2: '%s'", image_path_2
                )
        except ImageLoadError as err:
            logger.error("Image 2 load error: %s", err)
            err_text = f"Image 2: {err}"
            error_msg = f"{error_msg}; {err_text}" if error_msg else err_text
        except Exception:
            logger.exception(
                "Unexpected error processing image 2: '%s'", image_path_2
            )
            err_text = "Image 2: unexpected error (see log for details)"
            error_msg = f"{error_msg}; {err_text}" if error_msg else err_text

        # ---- Determine verdict ----
        if not face_1_ok and not face_2_ok:
            verdict = _VERDICT_ERR_BOTH
            if error_msg is None:
                error_msg = "No face detected in either image"
            return self._make_result(
                image_1=image_path_1,
                image_2=image_path_2,
                face_1_detected=False,
                face_2_detected=False,
                threshold_used=self._threshold,
                verdict=verdict,
                error=error_msg,
            )

        if not face_1_ok:
            verdict = _VERDICT_ERR_IMG1
            if error_msg is None:
                error_msg = "No face detected in image 1"
            return self._make_result(
                image_1=image_path_1,
                image_2=image_path_2,
                face_1_detected=False,
                face_2_detected=face_2_ok,
                threshold_used=self._threshold,
                verdict=verdict,
                error=error_msg,
            )

        if not face_2_ok:
            verdict = _VERDICT_ERR_IMG2
            if error_msg is None:
                error_msg = "No face detected in image 2"
            return self._make_result(
                image_1=image_path_1,
                image_2=image_path_2,
                face_1_detected=face_1_ok,
                face_2_detected=False,
                threshold_used=self._threshold,
                verdict=verdict,
                error=error_msg,
            )

        # ---- Both faces detected -- compute similarity ----
        return self.compare_embeddings(
            emb1, emb2,  # type: ignore[arg-type]
            image_path_1=image_path_1,
            image_path_2=image_path_2,
        )

    # ------------------------------------------------------------------
    # Compare pre-computed embeddings
    # ------------------------------------------------------------------
    def compare_embeddings(
        self,
        embedding_1: np.ndarray,
        embedding_2: np.ndarray,
        image_path_1: str = "<pre-computed>",
        image_path_2: str = "<pre-computed>",
    ) -> dict:
        """Compare two pre-computed embedding vectors.

        Useful for future phases where embeddings are stored in a
        database (e.g. a gallery of missing-person embeddings).

        Parameters
        ----------
        embedding_1 : np.ndarray
            L2-normalised 1-D embedding for face 1.
        embedding_2 : np.ndarray
            L2-normalised 1-D embedding for face 2.
        image_path_1 : str
            Optional label for result dict (defaults to "<pre-computed>").
        image_path_2 : str
            Optional label for result dict (defaults to "<pre-computed>").

        Returns
        -------
        dict
            Same structure as :meth:`compare_images`.
        """
        try:
            score = self._similarity.cosine_similarity(embedding_1, embedding_2)
        except (ValueError, TypeError) as err:
            logger.error("Embedding comparison failed: %s", err)
            return self._make_result(
                image_1=image_path_1,
                image_2=image_path_2,
                face_1_detected=True,
                face_2_detected=True,
                threshold_used=self._threshold,
                verdict="Error: embedding comparison failed",
                error=str(err),
            )

        is_match = score >= self._threshold
        verdict = _VERDICT_MATCH if is_match else _VERDICT_NO_MATCH

        logger.info(
            "Similarity=%.4f, threshold=%.4f -> %s",
            score, self._threshold, verdict,
        )

        return self._make_result(
            image_1=image_path_1,
            image_2=image_path_2,
            face_1_detected=True,
            face_2_detected=True,
            similarity=score,
            threshold_used=self._threshold,
            is_match=is_match,
            verdict=verdict,
            error=None,
        )

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------
    def print_result(self, result: dict) -> None:
        """Print a human-readable summary of a comparison result.

        Designed for live demo console output -- clean, visual, and
        easy to read at a glance.

        Parameters
        ----------
        result : dict
            Output from :meth:`compare_images` or
            :meth:`compare_embeddings`.
        """
        w = 60
        print()
        print("+" + "-" * w + "+")
        print("|" + " FACE MATCH PIPELINE -- RESULT ".center(w) + "|")
        print("+" + "-" * w + "+")

        # Image paths (truncate if very long)
        img1 = result.get("image_1", "?")
        img2 = result.get("image_2", "?")
        max_path = w - 14
        if len(img1) > max_path:
            img1 = "..." + img1[-(max_path - 3):]
        if len(img2) > max_path:
            img2 = "..." + img2[-(max_path - 3):]

        print(f"|  Image 1 : {img1:<{w - 13}}|")
        print(f"|  Image 2 : {img2:<{w - 13}}|")
        print("+" + "-" * w + "+")

        # Detection status
        f1 = "Yes" if result.get("face_1_detected") else "No"
        f2 = "Yes" if result.get("face_2_detected") else "No"
        print(f"|  Face 1 detected : {f1:<{w - 22}}|")
        print(f"|  Face 2 detected : {f2:<{w - 22}}|")
        print("+" + "-" * w + "+")

        # Similarity and threshold
        sim = result.get("similarity")
        thr = result.get("threshold_used", self._threshold)
        if sim is not None:
            sim_str = f"{sim:.4f}"
            thr_str = f"{thr:.4f}"
            diff_str = f"{sim - thr:+.4f}"
            print(f"|  Similarity score : {sim_str:<{w - 23}}|")
            print(f"|  Match threshold  : {thr_str:<{w - 23}}|")
            print(f"|  Margin           : {diff_str:<{w - 23}}|")
        else:
            print(f"|  Similarity score : {'N/A':<{w - 23}}|")
            thr_str = f"{thr:.4f}"
            print(f"|  Match threshold  : {thr_str:<{w - 23}}|")

        print("+" + "-" * w + "+")

        # Verdict
        verdict = result.get("verdict", "Unknown")
        is_match = result.get("is_match")

        if is_match is True:
            tag = ">> MATCH <<"
        elif is_match is False:
            tag = ">> NO MATCH <<"
        else:
            tag = ">> ERROR <<"

        print("|" + "".center(w) + "|")
        print("|" + tag.center(w) + "|")
        print("|" + verdict.center(w) + "|")
        print("|" + "".center(w) + "|")
        print("+" + "-" * w + "+")

        # Error details (if any)
        error = result.get("error")
        if error:
            print(f"|  Error: {error:<{w - 11}}|")
            print("+" + "-" * w + "+")

        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description=(
            "Phase 1 Face Match Pipeline -- compare two face images "
            "and output a match/no-match verdict."
        ),
    )
    parser.add_argument(
        "image1",
        help="Path to the first face image.",
    )
    parser.add_argument(
        "image2",
        help="Path to the second face image.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MATCH_THRESHOLD,
        help=(
            f"Cosine-similarity match threshold (default: {MATCH_THRESHOLD}). "
            f"Higher = stricter matching."
        ),
    )
    return parser


def main() -> int:
    """CLI entry point.  Returns an exit code.

    Returns
    -------
    int
        0 = successful run (match or no-match verdict produced).
        2 = error (missing file, no face, etc.).
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging for CLI use -- keep it clean so the result box
    # stands out.  Internal diagnostics go to WARNING+ only.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = FaceMatchPipeline(match_threshold=args.threshold)
    result = pipeline.compare_images(args.image1, args.image2)
    pipeline.print_result(result)

    if result.get("error") is not None:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
