# Phase 1: Face Recognition Pipeline — Experimental Validation

**Completed:** 24 August 2026

## Objective

Validate that a face-detection, embedding, and similarity-comparison pipeline can reliably match images of the same person while rejecting different-person pairs—without costly false positives.

## Approach

Using the LFW dataset, Phase 1 selected five people with three images each (15 images). Faces were detected with InsightFace, represented as ArcFace embeddings, and compared with cosine similarity. Scores from 75 genuine pairs and 300 impostor pairs were used to derive and evaluate an operating threshold.

## Results

| Metric | Result |
|---|---:|
| Validated threshold | **0.3468** |
| True positives | **73 / 75 (97.3%)** |
| True negatives | **300 / 300 (100.0%)** |
| False positives | **0 / 300 (0.0%)** |
| False negatives | **2 / 75 (2.7%)** |
| Overall accuracy | **99.5%** |

**Threshold selection:** 0.3468 is the midpoint between the mean genuine score (0.6874) and mean impostor score (0.0061). It achieved zero false positives in this experiment.

## What Phase 1 establishes

- Pairwise face matching works reliably on the Phase 1 test set.
- The selected threshold is experimentally derived, not guessed.
- No different-person pair was accepted at the calibrated threshold.
- The pipeline handles multiple-face and no-face inputs with graceful error handling.

## Caveats and next steps

This is promising but small-scale validation: five people and 15 total images cannot establish real-world accuracy or fairness. Two genuine pairs were missed due to low-quality/degraded images, and raw score tails overlap even though no impostor pair crossed the chosen threshold. Phase 2 will validate with larger and more diverse data and add database search; Phase 3 will address real-time and deployment features. Operational use should retain a conservative threshold and human review.

**Evidence:** `results/threshold_histogram.png` and `results/similarity_scores.csv`
