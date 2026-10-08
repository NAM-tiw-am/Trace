import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import init_db
from app.ai.similarity import FaceSimilarity
from app.ai.tracker import SightingTracker

init_db()
client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data
    assert "thresholds" in data

def test_case_lifecycle():
    # 1. Create case
    case_num = f"CASE-TEST-001"
    create_payload = {
        "case_number": case_num,
        "title": "Test Investigation",
        "description": "Integration test case",
        "status": "ACTIVE"
    }
    res = client.post("/api/cases", json=create_payload)
    assert res.status_code == 201
    case_data = res.json()
    case_id = case_data["id"]
    assert case_data["case_number"] == case_num

    # 2. Get case
    res = client.get(f"/api/cases/{case_id}")
    assert res.status_code == 200
    assert res.json()["title"] == "Test Investigation"

    # 3. List cases
    res = client.get("/api/cases")
    assert res.status_code == 200
    assert any(c["id"] == case_id for c in res.json())

    # 4. Update case
    res = client.put(f"/api/cases/{case_id}", json={"title": "Updated Title"})
    assert res.status_code == 200
    assert res.json()["title"] == "Updated Title"

    # 5. Register Person under case
    person_payload = {
        "case_id": case_id,
        "name": "Jane Doe",
        "age": 24,
        "gender": "Female",
        "last_known_location": "Airport Terminal 2",
        "status": "MISSING"
    }
    res = client.post("/api/persons", json=person_payload)
    assert res.status_code == 201
    person_data = res.json()
    person_id = person_data["id"]
    assert person_data["name"] == "Jane Doe"

    # 6. Clean up
    del_res = client.delete(f"/api/cases/{case_id}")
    assert del_res.status_code == 204

def test_face_similarity_math():
    # Identical vectors should have similarity 1.0
    v1 = np.random.randn(512).astype(np.float32)
    v1 = v1 / np.linalg.norm(v1)
    sim = FaceSimilarity.cosine_similarity(v1, v1)
    assert abs(sim - 1.0) < 1e-4

    # Opposite vectors should have similarity -1.0
    v_opp = -v1
    sim_opp = FaceSimilarity.cosine_similarity(v1, v_opp)
    assert abs(sim_opp - (-1.0)) < 1e-4

def test_sighting_tracker_deduplication():
    tracker = SightingTracker(time_window_seconds=3.0)
    
    # Simulate consecutive detections of same person across 4 frames within 1 second
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Frame 1 at t=1.0s, score 0.75
    tracker.update(
        person_id=1, video_id=10, job_id=1,
        frame_number=1, timestamp_seconds=1.0, score=0.75,
        bbox=[10, 10, 50, 50], frame_image=dummy_frame
    )
    
    # Frame 2 at t=1.2s, higher score 0.88
    tracker.update(
        person_id=1, video_id=10, job_id=1,
        frame_number=2, timestamp_seconds=1.2, score=0.88,
        bbox=[12, 12, 52, 52], frame_image=dummy_frame
    )
    
    # Frame 3 at t=1.5s, score 0.82
    tracker.update(
        person_id=1, video_id=10, job_id=1,
        frame_number=3, timestamp_seconds=1.5, score=0.82,
        bbox=[14, 14, 54, 54], frame_image=dummy_frame
    )
    
    # Finalize all tracks
    all_tracks = tracker.finalize_all()
    assert len(all_tracks) == 1, "Consecutive detections should aggregate into a single Sighting Event"
    track = all_tracks[0]
    assert track.total_detections == 3
    assert track.best_score == 0.88
    assert track.best_frame == 2
    assert track.start_time == 1.0
    assert track.last_time == 1.5

def test_sighting_review_workflow():
    # 1. Create Case & Person
    c_res = client.post("/api/cases", json={"case_number": "CASE-SIGHTING-001", "title": "Sighting Test"})
    case_id = c_res.json()["id"]

    p_res = client.post("/api/persons", json={"case_id": case_id, "name": "Review Subject", "status": "MISSING"})
    person_id = p_res.json()["id"]

    # 2. Directly create a Sighting record
    from app.core.database import SessionLocal
    from app.models.sighting import Sighting
    from app.models.video import Video
    db = SessionLocal()
    video = Video(
        camera_name="Cam 01", filename="test.mp4", file_path="storage/videos/test.mp4",
        fps=25.0, total_frames=100, duration_seconds=4.0, status="COMPLETED"
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    sighting = Sighting(
        person_id=person_id,
        video_id=video.id,
        timestamp_seconds=2.5,
        timestamp_formatted="00:00:02",
        frame_number=62,
        similarity_score=0.86,
        bbox_x1=50, bbox_y1=50, bbox_x2=150, bbox_y2=150,
        snapshot_path="storage/snapshots/test.jpg",
        match_status="POTENTIAL_MATCH",
    )
    db.add(sighting)
    db.commit()
    db.refresh(sighting)
    sighting_id = sighting.id
    db.close()

    # 3. Retrieve sighting via API
    res = client.get(f"/api/sightings/{sighting_id}")
    assert res.status_code == 200
    assert res.json()["similarity_score"] == 0.86
    assert res.json()["match_status"] == "POTENTIAL_MATCH"

    # 4. Filter sightings by person
    res_person = client.get(f"/api/persons/{person_id}/sightings")
    assert res_person.status_code == 200
    assert len(res_person.json()) == 1

    # 5. Investigator reviews and confirms sighting
    res_update = client.put(f"/api/sightings/{sighting_id}/status", json={
        "match_status": "CONFIRMED",
        "notes": "Verified visually by lead investigator."
    })
    assert res_update.status_code == 200
    assert res_update.json()["match_status"] == "CONFIRMED"
    assert "lead investigator" in res_update.json()["notes"]

    # 6. Clean up
    client.delete(f"/api/cases/{case_id}")

def test_person_photo_embedding_upload():
    import os
    test_photo_path = "Phase 1/data/test/person_01/image_01.jpg"
    if not os.path.exists(test_photo_path):
        pytest.skip("Test image not found")

    # 1. Create Case & Person
    c_res = client.post("/api/cases", json={"case_number": "CASE-PHOTO-001", "title": "Photo Upload Test"})
    case_id = c_res.json()["id"]

    p_res = client.post("/api/persons", json={"case_id": case_id, "name": "Photo Test Person"})
    person_id = p_res.json()["id"]

    # 2. Upload photo file
    with open(test_photo_path, "rb") as f:
        res = client.post(
            f"/api/persons/{person_id}/photos",
            files={"file": ("image_01.jpg", f, "image/jpeg")}
        )
    assert res.status_code == 201
    photo_data = res.json()
    assert photo_data["person_id"] == person_id

    # 3. Verify embedding was created and attached to person
    res_person = client.get(f"/api/persons/{person_id}")
    assert res_person.status_code == 200
    person_details = res_person.json()
    assert len(person_details["photos"]) >= 1
    assert len(person_details["embeddings"]) >= 1

    # 4. Clean up
    client.delete(f"/api/cases/{case_id}")
