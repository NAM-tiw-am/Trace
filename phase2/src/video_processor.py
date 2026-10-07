"""
video_processor.py — Read and write MP4 videos with OpenCV for Phase 2.

Provides the ``VideoProcessor`` class to open an input MP4, iterate
frames, track timestamps, and create an annotated output MP4.

Usage::

    from phase2.src.video_processor import VideoProcessor

    vp = VideoProcessor("phase2/videos/input/test_cctv.mp4")
    print(vp.fps, vp.width, vp.height, vp.total_frames)

    frame = vp.get_frame()
    ts = vp.get_frame_timestamp()
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Read input video frames and create output video writers.

    Parameters
    ----------
    video_path : str or Path
        Path to the input MP4 video.

    Raises
    ------
    FileNotFoundError
        If the video file does not exist.
    RuntimeError
        If OpenCV cannot open the video.
    """

    def __init__(self, video_path: str | Path) -> None:
        self._path = Path(video_path).resolve()

        if not self._path.exists():
            raise FileNotFoundError(
                f"Input video not found: {self._path}"
            )

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(
                f"OpenCV cannot open video: {self._path}"
            )

        # Read video properties
        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps: float = raw_fps if raw_fps and raw_fps > 0 else 25.0
        self._width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames: int = int(
            self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self._frame_index: int = 0

        logger.info(
            "Opened video: %s  |  %dx%d @ %.1f FPS  |  %d frames",
            self._path.name,
            self._width,
            self._height,
            self._fps,
            self._total_frames,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def frame_index(self) -> int:
        return self._frame_index

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------
    def get_frame(self) -> cv2.typing.MatLike | None:
        """Return the next BGR frame, or ``None`` at end-of-video."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        self._frame_index += 1
        return frame

    def get_frame_timestamp(self) -> float:
        """Return the current frame's timestamp in seconds."""
        if self._fps <= 0:
            return 0.0
        return (self._frame_index - 1) / self._fps

    # ------------------------------------------------------------------
    # Output video writer
    # ------------------------------------------------------------------
    def create_output_video(
        self,
        output_path: str | Path,
    ) -> cv2.VideoWriter:
        """Create an mp4v video writer matching the input dimensions/FPS.

        Parameters
        ----------
        output_path : str or Path
            Destination MP4 file path.

        Returns
        -------
        cv2.VideoWriter
            An opened writer. Caller is responsible for releasing it.

        Raises
        ------
        RuntimeError
            If the writer fails to open.
        """
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out), fourcc, self._fps, (self._width, self._height)
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to create video writer: {out}"
            )

        logger.info("Created output writer: %s", out)
        return writer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Seek back to the first frame."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frame_index = 0
        logger.info("Video reset to frame 0.")

    def close(self) -> None:
        """Release the video capture resource."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Video capture released.")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = _THIS_DIR.parent.parent

    video_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(PROJECT_ROOT / "phase2" / "videos" / "input" / "test_cctv.mp4")
    )

    try:
        vp = VideoProcessor(video_file)
        print(f"\n  Video     : {video_file}")
        print(f"  FPS       : {vp.fps}")
        print(f"  Size      : {vp.width}x{vp.height}")
        print(f"  Frames    : {vp.total_frames}")
        print(f"  Duration  : {vp.total_frames / vp.fps:.1f}s\n")

        # Read a few frames to verify
        count = 0
        while True:
            frame = vp.get_frame()
            if frame is None:
                break
            count += 1
        print(f"  Read {count} frames successfully.\n")

        vp.close()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
