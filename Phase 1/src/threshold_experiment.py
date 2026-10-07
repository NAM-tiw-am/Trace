"""
threshold_experiment.py -- Threshold Calibration Experiment for Phase 1.

This script determines what cosine-similarity score should count as a
"match" by experimentally comparing genuine pairs (same person) with
impostor pairs (different people) and analysing the score distributions.

Pipeline position::

    Embeddings (recognizer.py)
         |
    Similarity scores (similarity.py)
         |
    >>> Threshold experiment (this script) <<<
         |
    Calibrated threshold for pipeline.py

Outputs:
    results/similarity_scores.csv      -- all pair scores
    results/threshold_histogram.png    -- score distribution plot (if matplotlib)

Usage::

    python threshold_experiment.py

Author:  Phase 1 -- Evaluation Team
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import statistics
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the src/ directory is importable (script lives in src/)
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from recognizer import FaceRecognizer  # noqa: E402
from similarity import FaceSimilarity  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration -- adjust these as needed
# ---------------------------------------------------------------------------

# Path to the test dataset (relative to the working directory).
DATASET_DIR: str = r"C:\Users\nt458\OneDrive\Documents\vit documents\semister 3\project exbhition\missing-person-identification\Phase 1\data\test"

# Directory where results are saved.
RESULTS_DIR: str = "results"

# Impostor pairing strategy:
#   "first_image_only"  -- pair each person's FIRST image against every
#                          other person's FIRST image.  Fast and sufficient
#                          for a small Phase 1 dataset.
#   "all_combinations"  -- pair every image of person A against every image
#                          of person B, for all A != B.  WARNING: this grows
#                          quadratically and is only practical for very small
#                          datasets (< ~20 persons).
IMPOSTOR_PAIR_STRATEGY: str = "first_image_only"

# Maximum number of person folders to process (0 = all).
MAX_PERSONS: int = 0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("threshold_experiment")


# ---------------------------------------------------------------------------
# Type alias for a pair record
# ---------------------------------------------------------------------------
PairRecord = dict  # keys: person1, image1, person2, image2, pair_type, similarity


# ---------------------------------------------------------------------------
# Step 1: Load embeddings
# ---------------------------------------------------------------------------
def load_embeddings(
    dataset_dir: str,
    recognizer: FaceRecognizer,
    max_persons: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    """Walk the dataset and generate an embedding for every image.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing ``person_XX/`` sub-folders.
    recognizer : FaceRecognizer
        Initialised recognizer instance.
    max_persons : int
        Maximum number of person folders to process (0 = all).

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``{person_id: {image_filename: embedding}}``.
        Images that fail to produce an embedding are excluded.
    """
    root = Path(dataset_dir)
    if not root.exists():
        logger.error("Dataset directory does not exist: %s", root.resolve())
        sys.exit(1)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    embeddings: dict[str, dict[str, np.ndarray]] = {}

    # Collect person folders
    person_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("person_")
    )
    if max_persons > 0:
        person_dirs = person_dirs[:max_persons]

    # Count total images for progress
    all_images: list[tuple[str, Path]] = []
    for pdir in person_dirs:
        for img in sorted(pdir.iterdir()):
            if img.suffix.lower() in extensions:
                all_images.append((pdir.name, img))

    total = len(all_images)
    logger.info(
        "Found %d person folders, %d total images.",
        len(person_dirs), total,
    )

    for idx, (person_id, img_path) in enumerate(all_images, start=1):
        logger.info("Processing image %d/%d: %s", idx, total, img_path.name)

        try:
            emb = recognizer.get_embedding_from_path(str(img_path))
        except Exception:
            logger.exception("Failed to process %s/%s", person_id, img_path.name)
            continue

        if emb is None:
            logger.warning(
                "No face detected in %s/%s -- excluding from experiment.",
                person_id, img_path.name,
            )
            continue

        embeddings.setdefault(person_id, {})[img_path.name] = emb

    # Summary
    total_embs = sum(len(v) for v in embeddings.values())
    logger.info(
        "Embedding generation complete: %d/%d images produced embeddings "
        "across %d persons.",
        total_embs, total, len(embeddings),
    )
    return embeddings


# ---------------------------------------------------------------------------
# Step 2: Generate genuine pairs
# ---------------------------------------------------------------------------
def generate_genuine_pairs(
    embeddings: dict[str, dict[str, np.ndarray]],
    sim: FaceSimilarity,
) -> list[PairRecord]:
    """Generate all unique same-person pairs and compute similarity.

    For a person with N images, this produces N*(N-1)/2 pairs.

    Parameters
    ----------
    embeddings : dict
        Output of :func:`load_embeddings`.
    sim : FaceSimilarity
        Similarity calculator instance.

    Returns
    -------
    list[PairRecord]
        List of pair records labelled ``"genuine"``.
    """
    pairs: list[PairRecord] = []

    for person_id, person_embs in sorted(embeddings.items()):
        names = sorted(person_embs.keys())
        if len(names) < 2:
            logger.warning(
                "%s has only %d image(s) -- skipping genuine pairing.",
                person_id, len(names),
            )
            continue

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                score = sim.cosine_similarity(
                    person_embs[names[i]], person_embs[names[j]]
                )
                pairs.append({
                    "person1": person_id,
                    "image1": names[i],
                    "person2": person_id,
                    "image2": names[j],
                    "pair_type": "genuine",
                    "similarity": score,
                })

    logger.info("Generated %d genuine pairs.", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Step 3: Generate impostor pairs
# ---------------------------------------------------------------------------
def generate_impostor_pairs(
    embeddings: dict[str, dict[str, np.ndarray]],
    sim: FaceSimilarity,
    strategy: str = "first_image_only",
) -> list[PairRecord]:
    """Generate cross-person pairs and compute similarity.

    Parameters
    ----------
    embeddings : dict
        Output of :func:`load_embeddings`.
    sim : FaceSimilarity
        Similarity calculator instance.
    strategy : str
        ``"first_image_only"`` or ``"all_combinations"``.

    Returns
    -------
    list[PairRecord]
        List of pair records labelled ``"impostor"``.
    """
    pairs: list[PairRecord] = []
    person_ids = sorted(embeddings.keys())

    if len(person_ids) < 2:
        logger.warning(
            "Only %d person(s) available -- skipping impostor pairing.",
            len(person_ids),
        )
        return pairs

    for i in range(len(person_ids)):
        for j in range(i + 1, len(person_ids)):
            p1 = person_ids[i]
            p2 = person_ids[j]
            p1_embs = embeddings[p1]
            p2_embs = embeddings[p2]

            if strategy == "first_image_only":
                # Only compare the first image of each person.
                name1 = sorted(p1_embs.keys())[0]
                name2 = sorted(p2_embs.keys())[0]
                score = sim.cosine_similarity(p1_embs[name1], p2_embs[name2])
                pairs.append({
                    "person1": p1,
                    "image1": name1,
                    "person2": p2,
                    "image2": name2,
                    "pair_type": "impostor",
                    "similarity": score,
                })
            elif strategy == "all_combinations":
                # WARNING: This grows as O(P^2 * I^2) where P = number of
                # persons and I = images per person.  Only for small datasets.
                for name1 in sorted(p1_embs.keys()):
                    for name2 in sorted(p2_embs.keys()):
                        score = sim.cosine_similarity(
                            p1_embs[name1], p2_embs[name2]
                        )
                        pairs.append({
                            "person1": p1,
                            "image1": name1,
                            "person2": p2,
                            "image2": name2,
                            "pair_type": "impostor",
                            "similarity": score,
                        })
            else:
                raise ValueError(
                    f"Unknown impostor pair strategy: '{strategy}'. "
                    f"Use 'first_image_only' or 'all_combinations'."
                )

    logger.info(
        "Generated %d impostor pairs (strategy: %s).",
        len(pairs), strategy,
    )
    return pairs


# ---------------------------------------------------------------------------
# Step 4: Save results to CSV
# ---------------------------------------------------------------------------
def save_results_csv(
    pairs: list[PairRecord],
    results_dir: str,
) -> Path:
    """Write all pair records to a CSV file.

    The file is sorted by pair_type (genuine first) then by descending
    similarity for easy visual scanning.

    Parameters
    ----------
    pairs : list[PairRecord]
        Combined genuine + impostor pairs.
    results_dir : str
        Directory to write the CSV into (created if it doesn't exist).

    Returns
    -------
    Path
        Absolute path to the written CSV file.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "similarity_scores.csv"

    # Sort: genuine first, then descending similarity within each group.
    sorted_pairs = sorted(
        pairs,
        key=lambda r: (r["pair_type"], -r["similarity"]),
    )

    fieldnames = ["person1", "image1", "person2", "image2", "pair_type", "similarity"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_pairs:
            # Round similarity for cleaner CSV output.
            out_row = dict(row)
            out_row["similarity"] = f"{row['similarity']:.6f}"
            writer.writerow(out_row)

    logger.info("Saved %d pair records to %s", len(sorted_pairs), csv_path)
    return csv_path.resolve()


# ---------------------------------------------------------------------------
# Step 5 & 6: Summary statistics and threshold analysis
# ---------------------------------------------------------------------------
def compute_statistics(scores: list[float]) -> dict[str, float]:
    """Compute descriptive statistics for a list of scores.

    Returns
    -------
    dict with keys: count, min, max, mean, median, stdev.
    """
    if not scores:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "stdev": 0}
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "stdev": statistics.stdev(scores) if len(scores) >= 2 else 0.0,
    }


def print_summary_and_threshold(
    genuine_scores: list[float],
    impostor_scores: list[float],
) -> float | None:
    """Print summary statistics, threshold analysis, and return the
    suggested threshold.

    Parameters
    ----------
    genuine_scores : list[float]
        Cosine similarity scores for genuine pairs.
    impostor_scores : list[float]
        Cosine similarity scores for impostor pairs.

    Returns
    -------
    float or None
        Suggested threshold, or None if insufficient data.
    """
    g_stats = compute_statistics(genuine_scores)
    i_stats = compute_statistics(impostor_scores)

    # ---- Summary statistics table ----
    print(f"\n{'=' * 70}")
    print("  SCORE DISTRIBUTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<12}  |  {'Genuine':>10}  |  {'Impostor':>10}")
    print(f"  {'-' * 12}--+--{'-' * 10}--+--{'-' * 10}")
    print(f"  {'Count':<12}  |  {g_stats['count']:>10}  |  {i_stats['count']:>10}")

    if g_stats["count"] > 0 and i_stats["count"] > 0:
        for label, key in [("Min", "min"), ("Max", "max"), ("Mean", "mean"),
                           ("Median", "median"), ("Std Dev", "stdev")]:
            print(f"  {label:<12}  |  {g_stats[key]:>10.4f}  |  {i_stats[key]:>10.4f}")
    elif g_stats["count"] > 0:
        for label, key in [("Min", "min"), ("Max", "max"), ("Mean", "mean"),
                           ("Median", "median"), ("Std Dev", "stdev")]:
            print(f"  {label:<12}  |  {g_stats[key]:>10.4f}  |  {'N/A':>10}")
    elif i_stats["count"] > 0:
        for label, key in [("Min", "min"), ("Max", "max"), ("Mean", "mean"),
                           ("Median", "median"), ("Std Dev", "stdev")]:
            print(f"  {label:<12}  |  {'N/A':>10}  |  {i_stats[key]:>10.4f}")
    else:
        print("\n  No pairs to analyse.")
        return None

    # ---- Overlap analysis ----
    if g_stats["count"] > 0 and i_stats["count"] > 0:
        print(f"\n{'=' * 70}")
        print("  OVERLAP ANALYSIS")
        print(f"{'=' * 70}")

        g_min = g_stats["min"]
        i_max = i_stats["max"]

        if i_max < g_min:
            gap = g_min - i_max
            print(f"  Clean separation! No overlap between distributions.")
            print(f"  Gap: {gap:.4f}  (impostor max={i_max:.4f}, genuine min={g_min:.4f})")
        else:
            overlap_low = max(g_min, i_stats["min"])
            overlap_high = min(g_stats["max"], i_max)
            print(f"  Distributions OVERLAP in range [{g_min:.4f}, {i_max:.4f}]")
            print(f"  Genuine min:  {g_min:.4f}")
            print(f"  Impostor max: {i_max:.4f}")

            # Count how many scores fall in the overlap zone
            g_in_overlap = sum(1 for s in genuine_scores if s <= i_max)
            i_in_overlap = sum(1 for s in impostor_scores if s >= g_min)
            print(f"  Genuine scores in overlap zone:  {g_in_overlap}/{g_stats['count']}")
            print(f"  Impostor scores in overlap zone: {i_in_overlap}/{i_stats['count']}")

    # ---- Suggested threshold ----
    suggested_threshold: float | None = None

    if g_stats["count"] > 0 and i_stats["count"] > 0:
        # Midpoint between mean genuine and mean impostor scores.
        suggested_threshold = (g_stats["mean"] + i_stats["mean"]) / 2.0

        print(f"\n{'=' * 70}")
        print("  SUGGESTED THRESHOLD")
        print(f"{'=' * 70}")
        print(f"  Method: midpoint of mean genuine ({g_stats['mean']:.4f}) "
              f"and mean impostor ({i_stats['mean']:.4f})")
        print(f"  Suggested threshold: {suggested_threshold:.4f}")
        print()
        print("  NOTE: This is ONE reasonable starting point derived from")
        print("  the data, not a definitive final answer. You should:")
        print("    - Visually inspect results/similarity_scores.csv")
        print("    - Adjust based on your risk tolerance")
        print("    - For missing-person systems, bias toward FEWER false")
        print("      positives (i.e. a HIGHER threshold) to avoid")
        print("      incorrectly identifying someone as a missing person.")

        # ---- Performance at suggested threshold ----
        tp = sum(1 for s in genuine_scores if s >= suggested_threshold)
        fn = sum(1 for s in genuine_scores if s < suggested_threshold)
        tn = sum(1 for s in impostor_scores if s < suggested_threshold)
        fp = sum(1 for s in impostor_scores if s >= suggested_threshold)

        total_genuine = tp + fn
        total_impostor = tn + fp

        print(f"\n{'=' * 70}")
        print(f"  PERFORMANCE AT THRESHOLD = {suggested_threshold:.4f}")
        print(f"{'=' * 70}")
        print(f"  {'Metric':<28}  |  {'Count':>7}  |  {'Rate':>8}")
        print(f"  {'-' * 28}--+--{'-' * 7}--+--{'-' * 8}")

        tp_rate = (tp / total_genuine * 100) if total_genuine > 0 else 0
        fn_rate = (fn / total_genuine * 100) if total_genuine > 0 else 0
        tn_rate = (tn / total_impostor * 100) if total_impostor > 0 else 0
        fp_rate = (fp / total_impostor * 100) if total_impostor > 0 else 0

        print(f"  {'True Positives (genuine OK)':<28}  |  {tp:>4}/{total_genuine:<2}  |  {tp_rate:>7.1f}%")
        print(f"  {'False Negatives (genuine miss)':<28}  |  {fn:>4}/{total_genuine:<2}  |  {fn_rate:>7.1f}%")
        print(f"  {'True Negatives (impostor OK)':<28}  |  {tn:>4}/{total_impostor:<2}  |  {tn_rate:>7.1f}%")
        print(f"  {'False Positives (impostor miss)':<28}  |  {fp:>4}/{total_impostor:<2}  |  {fp_rate:>7.1f}%")

        if total_genuine > 0 and total_impostor > 0:
            accuracy = (tp + tn) / (total_genuine + total_impostor) * 100
            print(f"\n  Overall accuracy: {accuracy:.1f}%")

    elif g_stats["count"] > 0:
        print(f"\n  Only genuine pairs available (no impostor data).")
        print(f"  Cannot compute a threshold without impostor comparisons.")
        print(f"  Genuine score range: [{g_stats['min']:.4f}, {g_stats['max']:.4f}]")
    else:
        print(f"\n  Only impostor pairs available (no genuine data).")
        print(f"  Cannot compute a threshold without genuine comparisons.")

    print()
    return suggested_threshold


# ---------------------------------------------------------------------------
# Step 7: Visualization
# ---------------------------------------------------------------------------
def save_histogram(
    genuine_scores: list[float],
    impostor_scores: list[float],
    threshold: float | None,
    results_dir: str,
) -> bool:
    """Save a histogram of score distributions to PNG.

    Parameters
    ----------
    genuine_scores, impostor_scores : list[float]
        Score lists for each class.
    threshold : float or None
        Suggested threshold to mark with a vertical line.
    results_dir : str
        Output directory.

    Returns
    -------
    bool
        True if the plot was saved, False if matplotlib was unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for file output.
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info(
            "matplotlib is not installed -- skipping histogram generation. "
            "Install it with: pip install matplotlib"
        )
        return False

    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine bin range from combined scores.
    all_scores = genuine_scores + impostor_scores
    if not all_scores:
        logger.warning("No scores to plot.")
        return False

    bins = np.linspace(
        max(0, min(all_scores) - 0.05),
        min(1, max(all_scores) + 0.05),
        40,
    )

    if genuine_scores:
        ax.hist(
            genuine_scores, bins=bins, alpha=0.6,
            color="#2196F3", edgecolor="white", linewidth=0.5,
            label=f"Genuine (n={len(genuine_scores)})",
        )
    if impostor_scores:
        ax.hist(
            impostor_scores, bins=bins, alpha=0.6,
            color="#f44336", edgecolor="white", linewidth=0.5,
            label=f"Impostor (n={len(impostor_scores)})",
        )

    if threshold is not None:
        ax.axvline(
            x=threshold, color="#4CAF50", linestyle="--", linewidth=2,
            label=f"Suggested threshold = {threshold:.4f}",
        )

    ax.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax.set_ylabel("Number of Pairs", fontsize=12)
    ax.set_title(
        "Genuine vs Impostor Score Distributions\n(Phase 1 Threshold Experiment)",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "threshold_histogram.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Histogram saved to %s", out_path)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full threshold experiment."""

    logger.info("=" * 60)
    logger.info("  PHASE 1 -- THRESHOLD EXPERIMENT")
    logger.info("=" * 60)
    logger.info("Dataset:  %s", DATASET_DIR)
    logger.info("Results:  %s", RESULTS_DIR)
    logger.info("Impostor strategy: %s", IMPOSTOR_PAIR_STRATEGY)
    logger.info("Max persons: %s", "all" if MAX_PERSONS == 0 else MAX_PERSONS)

    # ---- Initialise modules ----
    logger.info("Initialising FaceRecognizer ...")
    recognizer = FaceRecognizer()
    sim = FaceSimilarity()
    logger.info("Execution provider: %s", recognizer.provider)

    # ---- Step 1: Load embeddings ----
    embeddings = load_embeddings(DATASET_DIR, recognizer, max_persons=MAX_PERSONS)
    if not embeddings:
        logger.error("No embeddings were generated. Cannot proceed.")
        sys.exit(1)

    # ---- Step 2: Genuine pairs ----
    genuine_pairs = generate_genuine_pairs(embeddings, sim)

    # ---- Step 3: Impostor pairs ----
    impostor_pairs = generate_impostor_pairs(
        embeddings, sim, strategy=IMPOSTOR_PAIR_STRATEGY
    )

    # ---- Step 4: Save CSV ----
    all_pairs = genuine_pairs + impostor_pairs
    if all_pairs:
        csv_path = save_results_csv(all_pairs, RESULTS_DIR)
    else:
        logger.warning("No pairs to save.")
        csv_path = None

    # ---- Step 5 & 6: Summary + Threshold ----
    genuine_scores = [p["similarity"] for p in genuine_pairs]
    impostor_scores = [p["similarity"] for p in impostor_pairs]

    threshold = print_summary_and_threshold(genuine_scores, impostor_scores)

    # ---- Step 7: Histogram ----
    if genuine_scores or impostor_scores:
        save_histogram(genuine_scores, impostor_scores, threshold, RESULTS_DIR)

    # ---- Final summary ----
    logger.info("Experiment complete.")
    if csv_path:
        logger.info("CSV results: %s", csv_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
