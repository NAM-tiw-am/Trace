"""
motion_detector.py — Frame-difference-based motion detector for Phase 2.

Compares consecutive grayscale+blurred frames via ``cv2.absdiff`` and
reports whether motion exceeds configurable thresholds.

Usage::

    from phase2.src.motion_detector import MotionDetector

    md = MotionDetector()
    has_motion = md.detect(frame)  # first call always returns False
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionDetector:
    """Detect inter-frame motion using frame differencing.

    Parameters
    ----------
    pixel_threshold : int
        Minimum absolute intensity difference (0–255) for a pixel to be
        counted as "changed".  Helps ignore sensor noise.  Default ``25``.
    area_threshold : float
        Fraction of total pixels that must be "changed" before the frame
        is declared as containing motion.  Default ``0.005`` (0.5 %).
    blur_kernel : tuple[int, int]
        Gaussian blur kernel size applied before differencing.  Larger
        kernels suppress more noise.  Default ``(21, 21)``.
    """

    def __init__(
        self,
        pixel_threshold: int = 25,
        area_threshold: float = 0.005,
        blur_kernel: tuple[int, int] = (21, 21),
    ) -> None:
        self._pixel_threshold = pixel_threshold
        self._area_threshold = area_threshold
        self._blur_kernel = blur_kernel

        self._prev_gray: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> tuple[bool, int]:
        """Check whether *frame* contains motion relative to the previous frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (OpenCV format).

        Returns
        -------
        has_motion : bool
            ``True`` if the changed-pixel ratio exceeds ``area_threshold``.
            Always ``False`` for the very first frame.
        changed_pixels : int
            Number of pixels that exceeded ``pixel_threshold``.
            Useful for tuning thresholds.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self._blur_kernel, 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False, 0

        # Absolute difference
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        # Threshold the difference image
        _, thresh = cv2.threshold(
            diff, self._pixel_threshold, 255, cv2.THRESH_BINARY
        )

        changed_pixels = int(cv2.countNonZero(thresh))
        total_pixels = thresh.shape[0] * thresh.shape[1]
        ratio = changed_pixels / total_pixels if total_pixels > 0 else 0.0

        has_motion = ratio >= self._area_threshold

        if has_motion:
            logger.debug(
                "Motion detected: %d changed pixels (%.2f%% of frame).",
                changed_pixels,
                ratio * 100,
            )

        return has_motion, changed_pixels

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the stored previous frame."""
        self._prev_gray = None
        logger.info("MotionDetector reset.")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = _THIS_DIR.parent.parent

    video_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(PROJECT_ROOT / "phase2" / "videos" / "input" / "test_cctv.mp4")
    )

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_file}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    md = MotionDetector()
    frame_idx = 0
    motion_count = 0

    print(f"\n  Scanning for motion in: {video_file}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        has_motion, changed = md.detect(frame)
        timestamp = frame_idx / fps
        if has_motion:
            motion_count += 1
            print(
                f"  Frame {frame_idx:>4d}  |  t={timestamp:>6.2f}s  |  "
                f"changed={changed:>6d}  |  MOTION"
            )
        frame_idx += 1

    cap.release()
    print(f"\n  Total frames: {frame_idx}  |  Motion frames: {motion_count}\n")
