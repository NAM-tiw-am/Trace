"""
cctv_search.py — CCTV Missing-Person Search Pipeline for Phase 2.

Orchestrates the complete CCTV detection flow:
  1. Read video frames via ``VideoProcessor``.
  2. Skip static frames using ``MotionDetector``.
  3. Detect faces on motion frames via Phase 1 ``FaceDetector``.
  4. Generate embeddings via Phase 1 ``FaceRecognizer``.
  5. Compare against stored embeddings via Phase 1 ``FaceSimilarity``.
  6. Annotate frames (green box = match, red box = unknown).
  7. Write annotated MP4 and JSON match log.

Usage::

    from phase2.src.cctv_search import CCTVSearchPipeline
    from phase2.src.database import MissingPersonDB

    db = MissingPersonDB()
    pipeline = CCTVSearchPipeline(db, match_threshold=0.35)
    matches = pipeline.process_video("input.mp4", "output.mp4")
    pipeline.save_match_log(matches)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from phase2.src.database import MissingPersonDB
from phase2.src.motion_detector import MotionDetector
from phase2.src.phase1_bridge import FaceDetector, FaceRecognizer, FaceSimilarity
from phase2.src.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "phase2" / "results"


class CCTVSearchPipeline:
    """Full CCTV missing-person search pipeline.

    Parameters
    ----------
    db : MissingPersonDB
        The missing-person embedding database.
    match_threshold : float
        Cosine similarity above which a face is declared a match.
    alert_cooldown_seconds : float
        Minimum seconds between repeated alerts for the same person.
    """

    def __init__(
        self,
        db: MissingPersonDB,
        match_threshold: float = 0.35,
        alert_cooldown_seconds: float = 5.0,
    ) -> None:
        self._db = db
        self._threshold = match_threshold
        self._cooldown = alert_cooldown_seconds

        # Initialise Phase 1 components
        logger.info("Initialising Phase 1 face detection/recognition models …")
        self._detector = FaceDetector()
        self._recognizer = FaceRecognizer(detector=self._detector)
        self._similarity = FaceSimilarity()

        self._motion = MotionDetector()

        logger.info(
            "CCTVSearchPipeline ready  |  threshold=%.4f  |  "
            "cooldown=%.1fs  |  DB has %d person(s)",
            self._threshold,
            self._cooldown,
            self._db.count,
        )

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path,
    ) -> list[dict]:
        """Process a CCTV video end-to-end.

        Parameters
        ----------
        video_path : str or Path
            Input MP4 video.
        output_path : str or Path
            Output annotated MP4 video.

        Returns
        -------
        list[dict]
            De-duplicated match records.
        """
        vp = VideoProcessor(video_path)
        writer = vp.create_output_video(output_path)
        self._motion.reset()

        # Pre-fetch all DB embeddings once
        db_embeddings, db_person_ids = self._db.get_all_embeddings()
        if not db_embeddings:
            logger.warning("Database is empty — no persons to match against.")

        matches: list[dict] = []
        # Track last alert time per person_id for cooldown
        last_alert: dict[str, float] = {}

        total_frames = vp.total_frames
        motion_frames = 0
        faces_detected = 0

        try:
            while True:
                frame = vp.get_frame()
                if frame is None:
                    break

                frame_num = vp.frame_index
                timestamp = vp.get_frame_timestamp()

                # Progress logging every 50 frames
                if frame_num % 50 == 0:
                    logger.info(
                        "Processing frame %d/%d  (t=%.2fs)",
                        frame_num, total_frames, timestamp,
                    )

                # Step 1: Check for motion
                has_motion, _ = self._motion.detect(frame)

                if has_motion:
                    motion_frames += 1

                    # Step 2: Detect faces
                    detections = self._detector.detect(frame)

                    for det in detections:
                        faces_detected += 1
                        bbox = det["bbox"]
                        x1, y1, x2, y2 = bbox

                        # Step 3: Generate embedding for this face
                        embedding = self._recognizer.get_embedding(
                            frame, bbox=bbox
                        )

                        if embedding is None:
                            # Could not generate embedding — mark as Unknown
                            self._draw_unknown_box(frame, bbox)
                            continue

                        # Step 4: Compare against all DB embeddings
                        best_score = -1.0
                        best_pid = None

                        for db_emb, pid in zip(db_embeddings, db_person_ids):
                            score = self._similarity.cosine_similarity(
                                embedding, db_emb
                            )
                            if score > best_score:
                                best_score = score
                                best_pid = pid

                        # Step 5: Annotate
                        if best_score >= self._threshold and best_pid is not None:
                            person = self._db.get_person(best_pid)
                            name = person["name"] if person else best_pid

                            self._draw_match_box(
                                frame, bbox, name, best_score, timestamp
                            )

                            # Cooldown: only log if enough time has passed
                            last_t = last_alert.get(best_pid, -999.0)
                            if timestamp - last_t >= self._cooldown:
                                match_record = {
                                    "person_id": best_pid,
                                    "name": name,
                                    "timestamp_seconds": round(timestamp, 3),
                                    "frame_number": frame_num,
                                    "similarity_score": round(best_score, 4),
                                }
                                matches.append(match_record)
                                last_alert[best_pid] = timestamp
                                logger.info(
                                    "MATCH: %s (%.4f) at %.2fs, frame %d",
                                    name, best_score, timestamp, frame_num,
                                )
                        else:
                            self._draw_unknown_box(frame, bbox)

                # Draw timestamp on every frame
                self._draw_timestamp(frame, timestamp)

                # Write every frame (motion or not) to output
                writer.write(frame)

        finally:
            writer.release()
            vp.close()

        logger.info(
            "Pipeline complete  |  frames=%d  |  motion=%d  |  "
            "faces=%d  |  matches=%d",
            total_frames, motion_frames, faces_detected, len(matches),
        )
        return matches

    # ------------------------------------------------------------------
    # Match log
    # ------------------------------------------------------------------
    def save_match_log(
        self,
        matches: list[dict],
        output_path: str | Path | None = None,
    ) -> Path:
        """Write match records to a JSON file.

        Parameters
        ----------
        matches : list[dict]
            Match records from :meth:`process_video`.
        output_path : str or Path or None
            Destination JSON file.  Defaults to
            ``phase2/results/match_log.json``.

        Returns
        -------
        Path
            The path the log was written to.
        """
        if output_path is None:
            out = DEFAULT_RESULTS_DIR / "match_log.json"
        else:
            out = Path(output_path).resolve()

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(matches, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Match log saved: %s  (%d records)", out, len(matches))
        return out

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_match_box(
        frame: np.ndarray,
        bbox: list[int],
        name: str,
        score: float,
        timestamp: float,
    ) -> None:
        """Draw a green bounding box with person name and score."""
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)  # green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} ({score:.2f})"
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        # Background for label
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - baseline - 6),
            (x1 + label_size[0] + 4, y1),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label, (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_unknown_box(
        frame: np.ndarray,
        bbox: list[int],
    ) -> None:
        """Draw a red bounding box with 'Unknown' label."""
        x1, y1, x2, y2 = bbox
        color = (0, 0, 255)  # red
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = "Unknown"
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - baseline - 6),
            (x1 + label_size[0] + 4, y1),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label, (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_timestamp(
        frame: np.ndarray,
        timestamp: float,
    ) -> None:
        """Draw the timestamp in the top-right corner of the frame."""
        h, w = frame.shape[:2]
        ts_text = f"t={timestamp:.2f}s"
        text_size, _ = cv2.getTextSize(
            ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        tx = w - text_size[0] - 10
        ty = 25
        cv2.putText(
            frame, ts_text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
