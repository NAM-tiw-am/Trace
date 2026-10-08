import logging
import time
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.core.config import settings
from app.models.video import Video
from app.models.processing_job import ProcessingJob
from app.models.sighting import Sighting
from app.models.person import Person
from app.services.face_service import FaceService
from app.services.matching_service import MatchingService
from app.services.storage_service import StorageService
from app.ai.video_processor import VideoProcessor, MotionDetector
from app.ai.tracker import SightingTracker, ActiveTrack

logger = logging.getLogger(__name__)

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

class ProcessingService:
    @staticmethod
    def create_job(db: Session, video_id: int) -> ProcessingJob:
        """Create a new queued processing job for video."""
        job = ProcessingJob(
            video_id=video_id,
            status="QUEUED",
            progress=0.0,
            current_frame=0,
            total_frames=0,
            faces_detected=0,
            matches_found=0,
            started_at=None,
            completed_at=None,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @classmethod
    def execute_job(cls, job_id: int) -> None:
        """Execute video processing job in background thread/task."""
        db = SessionLocal()
        try:
            job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if not job:
                logger.error("Job %d not found for execution", job_id)
                return

            video = db.query(Video).filter(Video.id == job.video_id).first()
            if not video:
                job.status = "FAILED"
                job.error_message = f"Video {job.video_id} not found."
                db.commit()
                return

            job.status = "PROCESSING"
            job.started_at = datetime.utcnow()
            video.status = "PROCESSING"
            db.commit()

            detector = FaceService.get_detector()
            tracker = SightingTracker(time_window_seconds=settings.ALERT_COOLDOWN_SECONDS)
            motion_detector = MotionDetector(
                pixel_threshold=settings.MOTION_PIXEL_THRESHOLD,
                area_threshold=settings.MOTION_AREA_THRESHOLD,
            ) if settings.MOTION_DETECTION_ENABLED else None

            video_path = StorageService.get_abs_path(video.file_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found at: {video_path}")

            def save_sighting_from_track(t: ActiveTrack):
                # Fetch person for label
                person = db.query(Person).filter(Person.id == t.person_id).first()
                person_name = person.name if person else f"Person #{t.person_id}"

                # Snapshot paths
                abs_snapshot, rel_snapshot = StorageService.get_snapshot_paths(t.video_id, t.best_frame)
                if t.best_snapshot_frame is not None:
                    VideoProcessor.save_snapshot(
                        frame=t.best_snapshot_frame,
                        output_path=abs_snapshot,
                        bbox=t.best_bbox,
                        label=person_name,
                        score=t.best_score,
                    )
                    snapshot_file = rel_snapshot
                else:
                    snapshot_file = ""

                sighting = Sighting(
                    person_id=t.person_id,
                    video_id=t.video_id,
                    job_id=t.job_id,
                    timestamp_seconds=t.best_time,
                    timestamp_formatted=format_timestamp(t.best_time),
                    frame_number=t.best_frame,
                    similarity_score=float(t.best_score),
                    bbox_x1=t.best_bbox[0],
                    bbox_y1=t.best_bbox[1],
                    bbox_x2=t.best_bbox[2],
                    bbox_y2=t.best_bbox[3],
                    snapshot_path=snapshot_file,
                    match_status="POTENTIAL_MATCH",
                    notes=f"Track duration: {t.last_time - t.start_time:.1f}s, detections: {t.total_detections}",
                )
                db.add(sighting)
                job.matches_found += 1
                db.commit()

            with VideoProcessor(video_path) as vp:
                total_frames = vp.total_frames
                job.total_frames = total_frames
                db.commit()

                last_progress_update = time.time()

                for frame_idx, timestamp, frame in vp.iter_frames(step=1):
                    # Check motion if enabled
                    if motion_detector is not None:
                        has_motion = motion_detector.detect(frame)
                        if not has_motion:
                            continue

                    # Detect faces
                    faces = detector.detect(frame)
                    if faces:
                        job.faces_detected += len(faces)

                    for face_info in faces:
                        emb = face_info.get("embedding")
                        if emb is None:
                            continue

                        # Query database for matches
                        matches = MatchingService.search_similar_persons(
                            db=db,
                            query_embedding=emb,
                            threshold=settings.MATCH_THRESHOLD,
                            top_k=1,
                        )

                        for person_id, score in matches:
                            finalized = tracker.update(
                                person_id=person_id,
                                video_id=video.id,
                                job_id=job.id,
                                frame_number=frame_idx,
                                timestamp_seconds=timestamp,
                                score=score,
                                bbox=face_info["bbox"],
                                frame_image=frame,
                            )
                            if finalized:
                                save_sighting_from_track(finalized)

                    # Check for expired tracks during processing
                    expired_tracks = tracker.finalize_expired(current_time=timestamp)
                    for exp in expired_tracks:
                        save_sighting_from_track(exp)

                    # Update progress every 2 seconds
                    if time.time() - last_progress_update > 2.0:
                        job.current_frame = frame_idx
                        if total_frames > 0:
                            job.progress = min(99.0, (frame_idx / total_frames) * 100.0)
                        db.commit()
                        last_progress_update = time.time()

                # Finalize any remaining active tracks
                remaining_tracks = tracker.finalize_all()
                for rem in remaining_tracks:
                    save_sighting_from_track(rem)

            job.progress = 100.0
            job.current_frame = job.total_frames
            job.status = "COMPLETED"
            job.completed_at = datetime.utcnow()
            video.status = "COMPLETED"
            db.commit()
            logger.info("Job %d completed successfully. Matches: %d", job_id, job.matches_found)

        except Exception as e:
            logger.exception("Error processing job %d: %s", job_id, e)
            job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job:
                job.status = "FAILED"
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                db.commit()
            if 'video' in locals() and video:
                video.status = "FAILED"
                db.commit()
        finally:
            db.close()
