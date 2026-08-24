# Missing-Person Identification — Phase 1

Phase 1 validates the core of a missing-person face-recognition workflow: detect a face, produce an embedding, compare two embeddings, and make a conservative match decision. Its purpose is to establish that this foundation works reliably before expanding into a full missing-person search system.

This phase covers pairwise image comparison only. Phase 2 will add database search and evaluation at greater scale; Phase 3 will add deployment-oriented capabilities such as real-time processing and alerts.

> **Important:** This is an experimental prototype, not a production identification system. A similarity score is evidence to support review; it must not be the sole basis for consequential decisions.

## Quick start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Compare two images
python src/pipeline.py path/to/image1.jpg path/to/image2.jpg

# Override the calibrated threshold when required
python src/pipeline.py image1.jpg image2.jpg --threshold 0.40
```

## Project structure

```text
phase1/
├── src/                         # Detection, recognition, comparison, and calibration code
│   ├── detector.py               # InsightFace face detection
│   ├── recognizer.py             # ArcFace embedding extraction
│   ├── similarity.py             # Cosine-similarity calculation
│   ├── pipeline.py               # End-to-end comparison command / API
│   └── threshold_experiment.py   # Batch evaluation and threshold calibration
├── data/test/                    # Five-person test set and edge-case images
│   ├── person_01/ … person_05/   # Three images per person
│   └── edge_cases/               # multiple_faces.jpg and no_face.jpg
├── results/                      # Scores, histogram, and demonstration screenshots
│   ├── similarity_scores.csv
│   ├── threshold_histogram.png
│   └── demo_screenshots/
├── notebooks/                    # Exploratory and dataset-preparation notebooks/scripts
├── requirements.txt
├── README.md
└── PHASE_1_SUMMARY.md
```

`src/` contains reusable pipeline logic. `data/` holds evaluation inputs, while `results/` holds generated evidence and visualizations; it is not source code.

## Architecture

```text
Image A ──┐  Detect face → ArcFace embedding ──┐
          │                                    ├→ cosine similarity → threshold → match decision
Image B ──┘  Detect face → ArcFace embedding ──┘
```

InsightFace locates faces and supplies the pre-trained ArcFace recognition model. ArcFace represents each selected face as a 512-dimensional numeric embedding. The pipeline measures cosine similarity between the two embeddings and compares it with a calibrated threshold.

## Installation and setup

Requirements: Python 3.8 or later, OpenCV, InsightFace, ONNX Runtime, NumPy, and Matplotlib. Install them with `pip install -r requirements.txt` inside a virtual environment.

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/pipeline.py --help
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py --help
```

The first InsightFace run downloads the `buffalo_l` model bundle (approximately 300 MB). Internet access and sufficient disk space are therefore needed once during initial model setup.

## Usage

### Command line

```bash
python src/pipeline.py data/test/person_01/image_1.jpg data/test/person_01/image_2.jpg
```

The Phase 1 calibrated default threshold is **0.3468**. To use a stricter value for an experiment:

```bash
python src/pipeline.py image1.jpg image2.jpg --threshold 0.40
```

### Python

```python
from pipeline import FaceMatchPipeline

pipeline = FaceMatchPipeline(threshold=0.3468)
result = pipeline.compare("image1.jpg", "image2.jpg")

print(result)
```

Use the import path that matches your local package layout (for example, `from src.pipeline import FaceMatchPipeline` when running from the project root). The result should be treated as a comparison outcome plus supporting score, not an identity claim.

## Phase 1 validation results

The threshold experiment used 15 images: five people with three images each. It evaluated 75 genuine (same-person) pairs and 300 impostor (different-person) pairs.

| Measure | Genuine pairs | Impostor pairs |
|---|---:|---:|
| Pair count | 75 | 300 |
| Mean similarity | 0.6874 | 0.0061 |
| Median similarity | 0.7130 | 0.0013 |
| Minimum | 0.0263 | -0.1584 |
| Maximum | 0.8980 | 0.2179 |
| Standard deviation | 0.1348 | 0.0591 |

### Calibrated decision threshold: 0.3468

The threshold was derived as the midpoint between the mean genuine score (0.6874) and mean impostor score (0.0061): `(0.6874 + 0.0061) / 2 = 0.3468`. On this test set it preserved **zero false positives**, a deliberate priority for a missing-person context.

| Outcome at threshold 0.3468 | Result |
|---|---:|
| True positives | 73 / 75 (97.3%) |
| False negatives | 2 / 75 (2.7%) |
| True negatives | 300 / 300 (100.0%) |
| False positives | 0 / 300 (0.0%) |
| Overall accuracy | 99.5% |

The score tails overlap in the raw distributions: the lowest genuine score is 0.0263 and the highest impostor score is 0.2179. However, no impostor pair crossed the selected threshold. Two low-quality/degraded genuine pairs were missed. See [`results/threshold_histogram.png`](results/threshold_histogram.png) for the score distribution and [`results/similarity_scores.csv`](results/similarity_scores.csv) for the underlying results.

## Key findings

- The pipeline successfully separates same-person and different-person pairs at the calibrated threshold on the Phase 1 test set.
- The threshold is supported by measured scores rather than selected arbitrarily.
- Zero false positives were observed in 300 impostor comparisons; the genuine match rate was 97.3%.
- Two genuine pairs were missed, associated with low-quality or degraded images.
- The pipeline handles no-face and multiple-face inputs through explicit, graceful error handling.

## Limitations and future work

- The validation set is small: five people and three images per person. Results require replication on much larger, diverse datasets.
- Phase 1 uses CPU-based inference; GPU acceleration is not yet configured.
- It has not been evaluated for large-gallery database search, real-time CCTV streams, alerts, or operational workflows.
- Image quality can cause genuine matches to be missed.
- For missing-person use, favor stricter thresholds and human review to minimize false positives. Thresholds must be recalibrated for the deployment population, camera conditions, and operating risk.

## Module reference

| Module | Responsibility |
|---|---|
| `detector.py` | Detects faces and returns bounding boxes and landmarks. |
| `recognizer.py` | Generates a 512-dimensional ArcFace embedding for a face. |
| `similarity.py` | Computes cosine similarity between embeddings. |
| `pipeline.py` | Integrates detection, embedding, similarity, and the match decision. |
| `threshold_experiment.py` | Runs batch comparisons and calibrates/evaluates the threshold. |

## Reproducing the evaluation

Run the calibration experiment from the Phase 1 directory:

```bash
python src/threshold_experiment.py
```

Review the regenerated scores and histogram in `results/`. To exercise error handling, run the pipeline with `data/test/edge_cases/multiple_faces.jpg` and `data/test/edge_cases/no_face.jpg`; confirm the program reports a clear outcome rather than producing an unsupported match decision.

## Contributing

Phase 1 is considered stable. Keep fixes focused and documented, and develop Phase 2 features on a separate branch in the main project structure. Contributions that expand validation datasets, improve test coverage, or make error behavior clearer are especially valuable.

## License and attribution

This project is intended to be released under the MIT License, unless the repository specifies otherwise. Face detection and recognition rely on InsightFace and its ArcFace-based models; comply with the applicable licenses and model terms before redistribution or deployment.
