from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class MotionDetector:
    """Frame differencing motion detector."""

    def __init__(
        self,
        pixel_threshold: int = 25,
        area_threshold: float = 0.005,
        blur_kernel: tuple[int, int] = (21, 21),
    ) -> None:
        self._pixel_thresh = pixel_threshold
        self._area_thresh = area_threshold
        self._blur_kernel = blur_kernel
        self._prev_gray: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray) -> bool:
        """Returns True if motion detected, False otherwise."""
        if frame is None or frame.size == 0:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self._blur_kernel, 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return True  # Process first frame

        diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(diff, self._pixel_thresh, 255, cv2.THRESH_BINARY)
        non_zero = cv2.countNonZero(thresh)
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_ratio = non_zero / total_pixels

        self._prev_gray = gray
        return motion_ratio >= self._area_thresh

    def reset(self) -> None:
        self._prev_gray = None


class VideoProcessor:
    """Read video, iterate frames with timestamps, and save snapshot evidence."""

    def __init__(self, video_path: str | Path) -> None:
        self._path = Path(video_path).resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {self._path}")

        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(raw_fps) if raw_fps and raw_fps > 0 else 25.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration = self._total_frames / self._fps if self._fps > 0 else 0.0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def iter_frames(self, step: int = 1) -> Iterator[Tuple[int, float, np.ndarray]]:
        """
        Yield (frame_number, timestamp_seconds, frame_bgr)
        step: sample every N frames (e.g. 1 = every frame, 2 = every 2nd frame)
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break

            if frame_idx % step == 0:
                timestamp = frame_idx / self._fps
                yield frame_idx, timestamp, frame

            frame_idx += 1

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @staticmethod
    def save_snapshot(
        frame: np.ndarray,
        output_path: str | Path,
        bbox: Optional[list[int]] = None,
        label: Optional[str] = None,
        score: Optional[float] = None,
    ) -> str:
        """Draw bounding box & label onto copy of frame and save to output_path."""
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        annotated = frame.copy()
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            color = (0, 220, 0) if (score and score >= 0.5) else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            text_items = []
            if label:
                text_items.append(label)
            if score is not None:
                text_items.append(f"{score:.1%}")
            
            if text_items:
                text_str = " | ".join(text_items)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (w, h), _ = cv2.getTextSize(text_str, font, font_scale, thickness)
                cv2.rectangle(annotated, (x1, max(0, y1 - h - 10)), (x1 + w + 10, y1), color, -1)
                cv2.putText(annotated, text_str, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), thickness)

        cv2.imwrite(str(p), annotated)
        return str(p)
