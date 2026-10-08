from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ActiveTrack:
    person_id: int
    video_id: int
    job_id: Optional[int]
    start_frame: int
    start_time: float
    last_frame: int
    last_time: float
    best_frame: int
    best_time: float
    best_score: float
    best_bbox: list[int]
    best_snapshot_frame: Optional[np.ndarray] = None
    total_detections: int = 1

class SightingTracker:
    """
    Aggregates consecutive detections of the same person within a temporal window
    into coherent sighting events, preventing duplicate alerts across frames.
    """

    def __init__(self, time_window_seconds: float = 5.0) -> None:
        self._time_window = time_window_seconds
        # Map person_id -> ActiveTrack
        self._active_tracks: dict[int, ActiveTrack] = {}

    def update(
        self,
        person_id: int,
        video_id: int,
        job_id: Optional[int],
        frame_number: int,
        timestamp_seconds: float,
        score: float,
        bbox: list[int],
        frame_image: Optional[np.ndarray],
    ) -> Optional[ActiveTrack]:
        """
        Record a detection. If a track exists for person_id within time_window, update it.
        If a previous track for person_id has expired, finalizes and returns it, and starts a new track.
        """
        finalized_track = None

        if person_id in self._active_tracks:
            track = self._active_tracks[person_id]
            # Check if this detection is within the temporal window
            if timestamp_seconds - track.last_time <= self._time_window:
                # Update existing track
                track.last_frame = frame_number
                track.last_time = timestamp_seconds
                track.total_detections += 1
                if score > track.best_score:
                    track.best_score = score
                    track.best_frame = frame_number
                    track.best_time = timestamp_seconds
                    track.best_bbox = bbox
                    if frame_image is not None:
                        track.best_snapshot_frame = frame_image.copy()
                return None
            else:
                # Previous track timed out; finalize it
                finalized_track = track
                del self._active_tracks[person_id]

        # Start new track
        new_track = ActiveTrack(
            person_id=person_id,
            video_id=video_id,
            job_id=job_id,
            start_frame=frame_number,
            start_time=timestamp_seconds,
            last_frame=frame_number,
            last_time=timestamp_seconds,
            best_frame=frame_number,
            best_time=timestamp_seconds,
            best_score=score,
            best_bbox=bbox,
            best_snapshot_frame=frame_image.copy() if frame_image is not None else None,
            total_detections=1,
        )
        self._active_tracks[person_id] = new_track
        return finalized_track

    def finalize_expired(self, current_time: float) -> list[ActiveTrack]:
        """Finalize any tracks that have had no detections within the time window."""
        expired = []
        for person_id in list(self._active_tracks.keys()):
            track = self._active_tracks[person_id]
            if current_time - track.last_time > self._time_window:
                expired.append(track)
                del self._active_tracks[person_id]
        return expired

    def finalize_all(self) -> list[ActiveTrack]:
        """Flush and return all remaining tracks at video completion."""
        remaining = list(self._active_tracks.values())
        self._active_tracks.clear()
        return remaining
