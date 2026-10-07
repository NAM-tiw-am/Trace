# Missing Person Identification System 🔍

An AI-powered system that identifies missing persons in CCTV footage using face detection, recognition, and real-time video analysis.

## 🎯 What It Does

This project uses **InsightFace (ArcFace)** deep learning models to:
1. **Detect faces** in images and video frames
2. **Generate unique face embeddings** (512-dimensional numerical fingerprints)
3. **Search CCTV footage** for missing persons by comparing face embeddings
4. **Produce annotated video** with bounding boxes and match scores

## 📂 Project Structure

```
missing-person-identification/
├── Phase 1/                        # Core AI modules
│   ├── src/
│   │   ├── detector.py             # Face detection (InsightFace RetinaFace)
│   │   ├── recognizer.py           # Face embedding (ArcFace w600k_r50)
│   │   ├── similarity.py           # Embedding comparison (cosine similarity)
│   │   ├── pipeline.py             # End-to-end face matching pipeline
│   │   └── threshold_experiment.py # Threshold calibration
│   └── data/test/                  # Test images (25 persons × 3 images)
│
├── phase2/                         # CCTV Video Analysis Pipeline
│   ├── create_test_video.py        # Generate test CCTV video
│   ├── src/
│   │   ├── phase1_bridge.py        # Import bridge to Phase 1
│   │   ├── database.py             # Missing person embedding database
│   │   ├── video_processor.py      # MP4 video reader/writer
│   │   ├── motion_detector.py      # Frame-difference motion detection
│   │   ├── cctv_search.py          # Core CCTV search pipeline
│   │   └── demo_cctv.py            # Complete demo script
│   ├── videos/input/               # Input CCTV videos
│   ├── videos/output/              # Annotated output videos
│   ├── data/                       # Person embedding database (JSON)
│   └── results/                    # Match logs (JSON)
│
├── PROJECT_DOCUMENTATION.txt       # Detailed project documentation
├── PHASE2_CCTV_AI_GUIDE.md         # Phase 2 implementation guide
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/missing-person-identification.git
cd missing-person-identification

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run

### Step 1: Generate Test Video
```bash
python phase2/create_test_video.py
```
Creates a synthetic CCTV video from test images (75 images → 225 frames @ 5 FPS).

### Step 2: Run the CCTV Detection Demo
```bash
python -m phase2.src.demo_cctv
```
This will:
- Load 3 test persons into the database
- Scan the test video for their faces
- Output an annotated video with bounding boxes
- Save a JSON match log

### Expected Output
```
DEMO COMPLETE
People in DB   : 3
Matches found  : 3
Output video   : phase2/videos/output/annotated_cctv.mp4
Match log      : phase2/results/match_log.json
```

## 🖼️ Using Your Own Photos & Videos

### Add your own missing person
Edit `phase2/src/demo_cctv.py` — change the `DEMO_PEOPLE` list:

```python
DEMO_PEOPLE = [
    ("P001", "Person Name", 25, "path/to/clear_face_photo.jpg"),
]
```

### Use your own CCTV video
Place your MP4 file in `phase2/videos/input/` and rename it to `test_cctv.mp4`, or update the `INPUT_VIDEO` path in `demo_cctv.py`.

## ⚙️ How It Works

```
Missing Person Photo → Face Detection → Face Embedding (512-d vector)
                                              ↓
                                        Store in Database
                                              ↓
CCTV Video → Motion Detection → Face Detection → Face Embedding
                                                      ↓
                                                Compare (Cosine Similarity)
                                                      ↓
                                          Match ≥ 0.35 → Green Box (Name + Score)
                                          Match < 0.35 → Red Box (Unknown)
                                                      ↓
                                          Annotated Video + Match Log
```

### Key Concepts
| Concept | Description |
|---------|-------------|
| **Face Embedding** | 512 numbers that uniquely represent a face |
| **Cosine Similarity** | Measures how similar two embeddings are (0–1) |
| **Motion Detection** | Skips static frames to save processing time |
| **Match Threshold** | 0.35 — scores above this are flagged as matches |
| **Cooldown** | Prevents duplicate alerts for the same person (5s default) |

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Programming language |
| OpenCV | Image/video processing |
| NumPy | Numerical computations |
| InsightFace | Face detection & recognition AI |
| ArcFace (w600k_r50) | Face embedding model (512-dim) |
| ONNX Runtime | Neural network inference |

## 📊 Demo Results

| Person | Similarity Score | Detected At |
|--------|:---------------:|:-----------:|
| Person 01 | 75.32% | t=0.60s |
| Person 02 | 93.84% | t=1.80s |
| Person 03 | 96.40% | t=3.60s |

## ⏱️ Performance

- **CPU processing:** ~2–3 seconds per motion frame
- **45-second test video:** ~5–6 minutes total
- **GPU (CUDA):** 5–10× faster with `onnxruntime-gpu`

## 📝 License

This project is for educational purposes (VIT University — Semester 3 Project Exhibition).

## 👤 Author

Built as part of the Missing Person Identification project for VIT University.
