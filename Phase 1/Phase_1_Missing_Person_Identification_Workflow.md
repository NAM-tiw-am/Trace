# Missing Person Identification System — Phase 1 Workflow

## 1. Phase 1 Objective

The goal of Phase 1 is to build and validate the **core face recognition pipeline**.

By the end of Phase 1, the system should be able to:

1. Take an image as input.
2. Detect a face in the image.
3. Generate a facial embedding for the detected face.
4. Compare embeddings from two images.
5. Calculate a similarity/distance score.
6. Decide whether the two images are a **potential match** based on a validated threshold.
7. Demonstrate that the system can distinguish between the same and different people.

> **Phase 1 does not include the missing-person database, FAISS search, Streamlit dashboard, CCTV processing, alerts, or case management.** Those belong to later phases.

---

## 2. Phase 1 End-to-End Workflow

```text
                    INPUT IMAGE
                         |
                         v
                 Image Validation
                         |
                         v
                  Face Detection
                         |
                         v
                 Face Alignment
                         |
                         v
                Face Embedding
                         |
                         v
              512-D / Model Vector
                         |
              +----------+----------+
              |                     |
              v                     v
        Reference Image        Query Image
              |                     |
              v                     v
          Embedding             Embedding
              |                     |
              +----------+----------+
                         |
                         v
                 Similarity / Distance
                         |
                         v
                 Threshold Decision
                         |
                  +------+------+
                  |             |
                  v             v
             Potential      Unlikely
               Match          Match
```

---

# 3. Technology Stack

We will keep Phase 1 entirely in Python.

| Purpose | Technology |
|---|---|
| Programming language | Python |
| Image processing | OpenCV |
| Face detection / recognition | InsightFace |
| Recognition model | ArcFace through InsightFace |
| Numerical operations | NumPy |
| Testing / experimentation | Jupyter Notebook or Python scripts |
| Environment | Python virtual environment |
| Version control | Git + GitHub |

### Why InsightFace + ArcFace?

We do not want to train a face-recognition model from scratch.

Instead, we will use a **pre-trained face recognition model** to generate embeddings.

The important concept is:

```text
Face Image
    ↓
Pre-trained ArcFace model
    ↓
Facial Embedding
    ↓
Compare embeddings
```

The embedding represents facial features numerically, allowing us to compare two faces.

---

# 4. Team Workflow

If the team has multiple members, divide the work into parallel modules.

### Member / Group 1 — Face Detection

Responsible for:

- OpenCV basics
- Loading images
- Detecting faces
- Cropping faces
- Handling images with no detected face
- Testing detection on different image qualities

### Member / Group 2 — Face Recognition

Responsible for:

- Understanding embeddings
- Setting up InsightFace
- Loading ArcFace
- Generating embeddings
- Understanding cosine similarity / distance
- Comparing two embeddings

### Member / Group 3 — Dataset & Testing

Responsible for:

- Collecting a small legitimate test dataset
- Organizing images
- Creating same-person and different-person test pairs
- Running experiments
- Recording similarity scores
- Measuring false matches and missed matches

### Member / Group 4 — Integration & Documentation

Responsible for:

- Connecting the modules
- Creating the Phase 1 Python pipeline
- Maintaining GitHub
- Writing documentation
- Preparing screenshots/results
- Preparing the Phase 1 presentation/demo

If the team has fewer people, combine these responsibilities.

---

# 5. Step-by-Step Implementation Plan

## Step 1 — Set Up the Python Environment

Create a dedicated project directory.

```text
missing-person-identification/
│
├── phase1/
│
├── data/
│   ├── raw/
│   └── test/
│
├── notebooks/
│
├── src/
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

Windows:

```bash
venv\Scripts\activate
```

Then install the required packages.

The exact InsightFace installation may depend on the machine and ONNX Runtime setup, so the team should verify installation on the target machines before locking `requirements.txt`.

---

# 6. Step 2 — Understand Face Detection

Before using recognition, understand the first part of the pipeline:

```text
Image
 ↓
Face Detector
 ↓
Bounding Box
 ↓
Face Crop
```

For example:

```text
Original Image
+-----------------------------+
|                             |
|       ┌──────────┐          |
|       │   FACE   │          |
|       │          │          |
|       └──────────┘          |
|                             |
+-----------------------------+
```

The detector should return information such as:

- Bounding box
- Detection confidence
- Facial landmarks, if available

### Test cases

Test with:

- One clear face
- Multiple faces
- No face
- Side-facing face
- Low-light image
- Low-resolution image
- Different backgrounds

The goal is not to solve every difficult case in Phase 1.

The goal is to understand the limitations of the detection stage.

---

# 7. Step 3 — Generate Face Embeddings

Once a face is detected, pass it through the recognition model.

Conceptually:

```text
Image
  ↓
Face Detection
  ↓
Detected Face
  ↓
ArcFace
  ↓
Embedding Vector
```

The embedding is a numerical representation of the face.

For example:

```text
[0.12, -0.04, 0.31, ...]
```

The exact size depends on the model being used.

Do **not** treat the individual numbers as human-readable facial characteristics. They are learned feature representations.

---

# 8. Step 4 — Compare Two Faces

Suppose we have:

```text
Person A — Image 1
Person A — Image 2
```

Generate:

```text
Embedding A1
Embedding A2
```

Then compare them.

For example, cosine similarity can be used:

```text
similarity = cosine_similarity(embedding1, embedding2)
```

Conceptually:

```text
Same person:

Embedding 1 ───────────┐
                        ├──→ HIGH similarity
Embedding 2 ───────────┘


Different people:

Embedding 1 ───────────┐
                        ├──→ LOWER similarity
Embedding 2 ───────────┘
```

---

# 9. Step 5 — Do NOT Guess the Threshold

A very important part of Phase 1 is threshold selection.

Do not simply decide:

> "0.80 means match."

The threshold needs to be evaluated experimentally.

Create two types of comparisons:

### Genuine pairs

Two different photographs of the **same person**.

```text
Person A image 1
vs
Person A image 2
```

### Impostor pairs

Photographs of **different people**.

```text
Person A image
vs
Person B image
```

Collect the similarity scores.

Example:

| Pair | Same Person? | Similarity |
|---|---:|---:|
| A1 vs A2 | Yes | 0.89 |
| A1 vs A3 | Yes | 0.84 |
| B1 vs B2 | Yes | 0.91 |
| A1 vs B1 | No | 0.32 |
| A2 vs C1 | No | 0.41 |
| B1 vs C2 | No | 0.28 |

Then examine where the genuine and impostor scores overlap.

The final threshold should be based on your experimental results and the selected model, rather than being arbitrarily chosen.

---

# 10. Step 6 — Evaluate the Recognition System

We should measure the system instead of only showing a few successful examples.

Useful metrics include:

### True Positive

The system identifies a genuine pair as a potential match.

### True Negative

The system correctly rejects a different person.

### False Positive

The system incorrectly treats different people as a potential match.

This is particularly important for a missing-person system.

### False Negative

The system fails to identify two images of the same person as a potential match.

---

# 11. Step 7 — Create a Phase 1 Experiment

The team should prepare a controlled test set.

For example:

```text
test/
│
├── person_01/
│   ├── image_01.jpg
│   ├── image_02.jpg
│   └── image_03.jpg
│
├── person_02/
│   ├── image_01.jpg
│   ├── image_02.jpg
│   └── image_03.jpg
│
└── person_03/
    ├── image_01.jpg
    ├── image_02.jpg
    └── image_03.jpg
```

The dataset should contain multiple images per person so that we can test:

```text
Same-person comparisons
+
Different-person comparisons
```

Use images that the team is legally allowed to use for the project and avoid uploading sensitive personal photographs to public repositories.

---

# 12. Step 8 — Build the Phase 1 Prototype

The final Phase 1 program should have a simple workflow:

```text
START
  |
  v
Select Image 1
  |
  v
Detect Face
  |
  +---- No Face ----> Show Error
  |
  v
Generate Embedding
  |
  v
Select Image 2
  |
  v
Detect Face
  |
  +---- No Face ----> Show Error
  |
  v
Generate Embedding
  |
  v
Calculate Similarity
  |
  v
Apply Validated Threshold
  |
  +------ Match ------+
  |                   |
  |                   v
  |              Show Similarity
  |              + Potential Match
  |
  +------ No Match --+
                      |
                      v
                 Show Similarity
```

At this stage, a simple command-line program or notebook is sufficient.

A UI is **not necessary** for Phase 1.

---

# 13. Suggested Phase 1 Project Structure

Once the basic experiments work, organize the code like this:

```text
phase1/
│
├── src/
│   ├── detector.py
│   ├── recognizer.py
│   ├── similarity.py
│   └── pipeline.py
│
├── tests/
│   ├── test_detection.py
│   └── test_similarity.py
│
├── data/
│   └── test/
│
├── results/
│   ├── similarity_scores.csv
│   └── evaluation_results.csv
│
├── notebooks/
│   └── phase1_experiments.ipynb
│
├── requirements.txt
└── README.md
```

### Responsibility of each file

**`detector.py`**

Handles:

```text
Image → Detected faces
```

**`recognizer.py`**

Handles:

```text
Face → Embedding
```

**`similarity.py`**

Handles:

```text
Embedding 1 + Embedding 2
            ↓
       Similarity
```

**`pipeline.py`**

Connects everything:

```text
Image 1
  ↓
Detection
  ↓
Embedding
  ↓
          Comparison
  ↑
Embedding
  ↑
Detection
  ↑
Image 2
```

---

# 14. GitHub Workflow

Use Git from the beginning.

Recommended branches:

```text
main
│
├── face-detection
├── face-recognition
├── evaluation
└── integration
```

Each member works on their branch.

Example:

```bash
git checkout -b face-detection
```

After completing a feature:

```bash
git add .
git commit -m "Implement face detection"
git push
```

Then merge into the development/integration branch after testing.

Do not directly push unfinished code to `main`.

---

# 15. What We Should NOT Build in Phase 1

Avoid scope creep.

Do NOT spend time on:

- Streamlit UI
- FAISS
- Missing-person database
- CCTV video processing
- Alerts
- Email/SMS notifications
- Location tracking
- Age progression
- Authentication
- Cloud deployment
- Mobile application

These features can be considered later.

Phase 1 should answer one question:

> **Can our AI pipeline reliably detect and compare faces?**

---

# 16. Phase 1 Evaluation Demo

For the first evaluation, demonstrate the following.

### Demo 1 — Same Person

Input:

```text
Image A → Person X
Image B → Person X
```

Expected:

```text
High similarity
Potential Match
```

### Demo 2 — Different People

Input:

```text
Image A → Person X
Image B → Person Y
```

Expected:

```text
Lower similarity
Unlikely Match
```

### Demo 3 — Multiple Faces

Show an image containing multiple people.

Demonstrate that the detector can locate individual faces.

### Demo 4 — No Face

Upload an image without a detectable face.

The system should gracefully report:

```text
No suitable face detected.
```

### Demo 5 — Evaluation Results

Show a table/graph of your experimental similarity scores.

This is important because it demonstrates that the project is based on testing rather than a few hand-picked examples.

---

# 17. Phase 1 Final Deliverables

By the end of Phase 1, the team should have:

- [ ] Working Python environment
- [ ] Face detection module
- [ ] Face embedding module
- [ ] Face similarity module
- [ ] Same-person test results
- [ ] Different-person test results
- [ ] Threshold experiment
- [ ] Basic evaluation metrics
- [ ] Clean GitHub repository
- [ ] Phase 1 documentation
- [ ] Demo script/notebook
- [ ] Screenshots/results for presentation

---

# 18. Phase 1 Success Criteria

Phase 1 is considered successful when:

```text
✓ Images can be loaded
✓ Faces can be detected
✓ Face embeddings can be generated
✓ Two embeddings can be compared
✓ Same-person pairs generally produce higher similarity
✓ Different-person pairs generally produce lower similarity
✓ A threshold can be justified using experimental data
✓ The system handles invalid/no-face inputs
✓ The complete pipeline runs from input to result
```

---

# 19. What Comes After Phase 1

Once Phase 1 is stable, Phase 2 will build the actual search system:

```text
                  PHASE 1
             Face Recognition
                    ↓
             Face Embeddings
                    ↓
              Similarity
                    ↓
            ───────────────
                    ↓
                  PHASE 2
            Missing Person DB
                    ↓
              FAISS Index
                    ↓
             Vector Search
                    ↓
              Top-K Matches
                    ↓
                  PHASE 3
             Complete System
```

The key principle is:

> **Do not build Phase 2 until the Phase 1 recognition pipeline has been experimentally validated.**

That gives the project a clear progression and makes each evaluation demonstrate a meaningful new capability.
