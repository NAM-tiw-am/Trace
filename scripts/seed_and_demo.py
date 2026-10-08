"""
seed_and_demo.py — End-to-end demonstration script for Trace.

Executes the complete MVP workflow:
1. Initialize Database
2. Create Case (CASE-2026-001)
3. Register Missing Person (Rahul Sharma)
4. Upload Reference Photo & Generate 512-D ArcFace Embedding in DB
5. Generate a sample CCTV video clip
6. Upload Video
7. Run AI Video Processing (Face Detection -> ArcFace -> pgvector / Cosine Match)
8. Output Sightings with Timestamps, Similarity Scores, and Snapshots!
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import SessionLocal, init_db
from app.models.case import Case
from app.models.person import Person
from app.models.photo import PersonPhoto
from app.models.video import Video
from app.models.sighting import Sighting
from app.services.face_service import FaceService
from app.services.video_service import VideoService
from app.services.processing_service import ProcessingService
from app.core.config import settings

def main():
    print("=" * 70)
    print("TRACE — AI MISSING PERSON IDENTIFICATION SYSTEM")
    print("Executing End-to-End MVP Demonstration...")
    print("=" * 70)

    # 1. Initialize Database
    init_db()
    db = SessionLocal()

    # 2. Create Case
    case_no = f"CASE-2026-DEMO"
    case = db.query(Case).filter(Case.case_number == case_no).first()
    if not case:
        case = Case(
            case_number=case_no,
            title="Missing Person Search - Exhibition Demo",
            description="Demonstration case for testing facial identification in CCTV feeds.",
            status="ACTIVE"
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        print(f"[+] Case created: {case.case_number} (ID: {case.id})")
    else:
        print(f"[*] Using existing case: {case.case_number} (ID: {case.id})")

    # 3. Register Missing Person
    person_name = "Rahul Sharma"
    person = db.query(Person).filter(Person.case_id == case.id, Person.name == person_name).first()
    if not person:
        person = Person(
            case_id=case.id,
            name=person_name,
            age=26,
            gender="Male",
            last_known_location="Central Transit Station Gate 2",
            status="MISSING"
        )
        db.add(person)
        db.commit()
        db.refresh(person)
        print(f"[+] Missing Person registered: {person.name} (ID: {person.id})")
    else:
        print(f"[*] Using existing person: {person.name} (ID: {person.id})")

    # 4. Upload Reference Photo & Generate Face Embedding
    ref_photo_src = PROJECT_ROOT / "Phase 1" / "data" / "test" / "person_01" / "image_01.jpg"
    if not ref_photo_src.exists():
        print(f"[!] Warning: Reference photo not found at {ref_photo_src}")
        return

    # Check if photo already registered
    photo = db.query(PersonPhoto).filter(PersonPhoto.person_id == person.id).first()
    if not photo:
        dest_photo = settings.PHOTOS_DIR / f"ref_{person.id}_image_01.jpg"
        dest_photo.write_bytes(ref_photo_src.read_bytes())

        photo = PersonPhoto(
            person_id=person.id,
            filename=dest_photo.name,
            file_path=str(dest_photo),
        )
        db.add(photo)
        db.commit()
        db.refresh(photo)

        print(f"[+] Reference photo stored: {photo.filename}")
        print("[*] Extracting 512-D ArcFace face embedding via InsightFace...")
        emb = FaceService.process_and_store_photo(db, person.id, photo.id, str(dest_photo))
        if emb:
            print(f"[+] Face detected! 512-D vector embedding stored in database.")
        else:
            print("[!] Failed to extract embedding from photo.")
            return
    else:
        print(f"[*] Person already has {len(person.photos)} photo(s) and {len(person.embeddings)} embedding(s).")

    # 5. Create a synthetic test CCTV video clip containing the person
    demo_video_path = settings.VIDEOS_DIR / "demo_cctv.mp4"
    if not demo_video_path.exists():
        print("[*] Creating synthetic CCTV footage containing test persons...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(demo_video_path), fourcc, 5.0, (640, 480))

        # Include person_01 (Rahul) and a couple other test persons
        sample_persons = ["person_01", "person_02", "person_03"]
        for p_dir_name in sample_persons:
            p_dir = PROJECT_ROOT / "Phase 1" / "data" / "test" / p_dir_name
            if p_dir.is_dir():
                for img_path in sorted(p_dir.glob("*.jpg")):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        resized = cv2.resize(img, (640, 480))
                        # Repeat each frame 5 times (1 second per image)
                        for _ in range(5):
                            out.write(resized)
        out.release()
        print(f"[+] Synthetic CCTV video generated at: {demo_video_path}")

    # 6. Register Video
    video = db.query(Video).filter(Video.filename == demo_video_path.name).first()
    if not video:
        video = VideoService.register_video(
            db=db,
            camera_name="Cam 04 - Terminal Concourse",
            filename=demo_video_path.name,
            file_path=str(demo_video_path),
            location="Terminal 1 Concourse East",
            case_id=case.id,
        )
        print(f"[+] Video registered: {video.camera_name} ({video.total_frames} frames, {video.duration_seconds:.1f}s)")
    else:
        print(f"[*] Using existing video record: {video.camera_name} (ID: {video.id})")

    # 7. Create & Run Processing Job
    print("[*] Launching AI Video Processing Job...")
    job = ProcessingService.create_job(db, video.id)
    print(f"[+] Job #{job.id} created. Running detection & matching pipeline...")

    # Execute synchronously for demo script
    ProcessingService.execute_job(job.id)

    # 8. Report Results
    db.refresh(job)
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE — Job #{job.id} Status: {job.status}")
    print(f"Total Frames Processed : {job.total_frames}")
    print(f"Faces Detected         : {job.faces_detected}")
    print(f"Matches Found          : {job.matches_found}")
    print(f"{'='*70}")

    sightings = db.query(Sighting).filter(Sighting.video_id == video.id).all()
    print(f"\nDETECTION RESULTS ({len(sightings)} Sighting Events):")
    for idx, s in enumerate(sightings, 1):
        print(f"  {idx}. Person: {s.person.name}")
        print(f"     Camera: {s.video.camera_name}")
        print(f"     Timestamp: {s.timestamp_formatted} (t={s.timestamp_seconds:.2f}s, frame={s.frame_number})")
        print(f"     Similarity: {s.similarity_score * 100:.2f}%")
        print(f"     Snapshot: {s.snapshot_path}")
        print(f"     Status: {s.match_status}")
        print(f"     Notes: {s.notes}")
        print("-" * 50)

    print("\n[OK] MVP Verification Completed Successfully!")
    db.close()

if __name__ == "__main__":
    main()
