# TRACE — AI-Powered Missing Person Identification System 🔍

Trace is an end-to-end missing-person identification system that leverages deep learning face recognition to search CCTV footage for appearances of a person and aggregates potential sightings with visual evidence snapshots and timestamps.

---

## 🎯 Target Architecture & Workflow

```text
Missing Person Photo
        ↓
Face Detection & Alignment
        ↓
Face Embedding (ArcFace 512-D via InsightFace buffalo_l)
        ↓
Store Person + Embedding in PostgreSQL (+ pgvector)
        ↓
Upload CCTV Video
        ↓
Process Video (OpenCV + Frame Differencing Motion Detection)
        ↓
Detect Faces in Motion Frames
        ↓
Generate Query Embeddings
        ↓
pgvector Cosine Search (<=> Operator)
        ↓
Sighting Event Aggregator (Avoid Duplicate Detections across consecutive frames)
        ↓
Create Sighting Record + Evidence Snapshot
        ↓
Investigator Dashboard (Review, Confirm, Reject)
```

---

## 📂 Project Structure

```text
Trace/
│
├── app/
│   ├── main.py                  # FastAPI application entrypoint & lifespan
│   │
│   ├── api/                     # REST API Routers
│   │   ├── health.py            # Health check (/api/health)
│   │   ├── cases.py             # Cases CRUD (/api/cases)
│   │   ├── persons.py           # Missing persons & photo uploads (/api/persons)
│   │   ├── videos.py            # CCTV video upload & processing (/api/videos)
│   │   ├── sightings.py         # Sighting retrieval & investigator review (/api/sightings)
│   │   └── jobs.py              # Background processing jobs tracking (/api/jobs)
│   │
│   ├── core/                    # Core configuration & Database setup
│   │   ├── config.py            # Pydantic Settings & environment variables
│   │   └── database.py          # SQLAlchemy Session, pgvector & SQLite fallback
│   │
│   ├── models/                  # SQLAlchemy ORM Models
│   │   ├── case.py              # Case model
│   │   ├── person.py            # Person model
│   │   ├── photo.py             # PersonPhoto model
│   │   ├── embedding.py         # FaceEmbedding model (Vector 512)
│   │   ├── video.py             # Video model
│   │   ├── sighting.py          # Sighting model
│   │   └── processing_job.py    # ProcessingJob model
│   │
│   ├── schemas/                 # Pydantic Schemas for Validation & Serialization
│   │   ├── case.py
│   │   ├── person.py
│   │   ├── video.py
│   │   ├── sighting.py
│   │   └── processing_job.py
│   │
│   ├── services/                # Business & Orchestration Services
│   │   ├── face_service.py      # Face detection & embedding extraction
│   │   ├── video_service.py     # Video metadata parsing & registration
│   │   ├── matching_service.py  # pgvector cosine distance search
│   │   ├── storage_service.py   # File system storage for photos, videos, snapshots
│   │   └── processing_service.py# Video frame iteration & background processing
│   │
│   └── ai/                      # Computer Vision & Deep Learning Layer
│       ├── detector.py          # InsightFace RetinaFace face detector
│       ├── recognizer.py        # ArcFace w600k_r50 embedding extractor
│       ├── similarity.py        # Cosine similarity and Euclidean distance math
│       ├── tracker.py           # SightingTracker event aggregator & deduplicator
│       └── video_processor.py   # OpenCV video processor & motion detector
│
├── static/                      # Interactive Dashboard Frontend
│   ├── index.html               # Modern surveillance dashboard UI
│   ├── app.js                   # Client state, modals, video playback & polling
│   └── styles.css               # Dark theme surveillance styling
│
├── storage/                     # File and object storage
│   ├── photos/                  # Uploaded reference photos
│   ├── videos/                  # CCTV footage files
│   └── snapshots/               # Sighting evidence crops with bounding boxes
│
├── tests/
│   └── test_api.py              # Unit & integration tests
│
├── scripts/
│   └── seed_and_demo.py         # Complete end-to-end demonstration script
│
├── .env                         # Local environment configuration
├── .env.example                 # Environment template
├── docker-compose.yml           # PostgreSQL + pgvector + Backend container setup
├── Dockerfile                   # Backend Docker image specification
├── pytest.ini                   # Pytest configuration
└── requirements.txt             # Project dependencies
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Demo Script
Runs the complete MVP pipeline from creating a case, registering a missing person, extracting a 512-D ArcFace embedding, scanning CCTV footage, running vector matching, aggregating sightings, and saving snapshot evidence:
```bash
python scripts/seed_and_demo.py
```

### 3. Start the FastAPI Server & Web Dashboard
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- **Web Dashboard**: Open [http://localhost:8000](http://localhost:8000) in your browser.
- **Interactive API Docs (Swagger UI)**: Open [http://localhost:8000/docs](http://localhost:8000/docs).
- **Alternative API Docs (ReDoc)**: Open [http://localhost:8000/redoc](http://localhost:8000/redoc).

---

## 🧪 Running Tests

Run the test suite verifying API endpoints, vector mathematics, deduplication, photo uploads, and review workflows:
```bash
python -m pytest tests/ -v
```

---

## 📡 API Overview

### Cases
- `POST   /api/cases` — Create a case (`CASE-2026-001`)
- `GET    /api/cases` — List all cases
- `GET    /api/cases/{id}` — Get case details
- `PUT    /api/cases/{id}` — Update case details
- `DELETE /api/cases/{id}` — Delete case

### Missing Persons & Reference Photos
- `POST   /api/persons` — Register a missing person
- `GET    /api/persons` — List missing persons
- `GET    /api/persons/{id}` — Get person details with photos and embeddings
- `PUT    /api/persons/{id}` — Update person
- `DELETE /api/persons/{id}` — Delete person
- `POST   /api/persons/{id}/photos` — Upload reference photo (extracts 512-D ArcFace embedding)
- `GET    /api/persons/{id}/photos` — List person's photos
- `DELETE /api/persons/{id}/photos/{photo_id}` — Delete photo and embedding

### CCTV Videos & Video Processing
- `POST   /api/videos` — Upload CCTV video (MP4)
- `GET    /api/videos` — List CCTV videos
- `GET    /api/videos/{id}` — Get video metadata (FPS, frame count, duration)
- `POST   /api/videos/{id}/process` — Trigger background AI processing job
- `GET    /api/jobs/{id}` — Check status and progress of processing job

### Sightings & Verification Review
- `GET    /api/sightings` — List sightings (filter by `person_id`, `video_id`, `match_status`)
- `GET    /api/persons/{id}/sightings` — List all sightings for a specific person
- `GET    /api/videos/{id}/sightings` — List all sightings found in a video
- `GET    /api/sightings/{id}` — Get single sighting with snapshot & video links
- `PUT    /api/sightings/{id}/status` — Update verification status (`POTENTIAL_MATCH`, `REVIEWED`, `CONFIRMED`, `REJECTED`)

---

## 🐳 Running with Docker & PostgreSQL + pgvector

To run the complete production stack with PostgreSQL 16 and pgvector:

```bash
docker-compose up --build
```
This starts:
- **`db`**: PostgreSQL 16 with `pgvector/pgvector:pg16`
- **`app`**: FastAPI backend with InsightFace model caching and live dashboard on port 8000.
