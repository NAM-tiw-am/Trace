"""
test_similarity.py -- End-to-end similarity test using real face embeddings.

Loads test images from ``data/test/person_XX/image_XX.jpg``, generates
embeddings via ``FaceRecognizer``, then computes genuine-pair and
impostor-pair similarities using ``FaceSimilarity``.

Prints a clear summary showing the separation between genuine and impostor
score distributions -- this is a smoke test confirming the pipeline produces
sensible results, NOT the full threshold calibration experiment (that comes
in a later step).

Usage::

    python test_similarity.py                       # default paths
    python test_similarity.py path/to/test_folder   # custom test root
"""

from __future__ import annotations

import itertools
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the src/ directory is importable
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

import numpy as np  # noqa: E402
from recognizer import FaceRecognizer, ImageLoadError  # noqa: E402
from similarity import FaceSimilarity  # noqa: E402

# ---------------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _collect_person_images(
    test_root: Path,
    max_persons: int = 5,
) -> dict[str, list[Path]]:
    """Collect images grouped by person folder.

    Returns
    -------
    dict[str, list[Path]]
        Mapping of person folder name -> sorted list of image paths.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    persons: dict[str, list[Path]] = {}

    for entry in sorted(test_root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("person_"):
            continue
        if max_persons > 0 and len(persons) >= max_persons:
            break

        images = sorted(
            p for p in entry.iterdir()
            if p.suffix.lower() in extensions
        )
        if images:
            persons[entry.name] = images

    return persons


def _generate_embeddings(
    recognizer: FaceRecognizer,
    person_images: dict[str, list[Path]],
) -> dict[str, dict[str, np.ndarray]]:
    """Generate embeddings for all images, grouped by person.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        person_name -> { image_filename -> embedding }
        Images that fail to produce an embedding are silently skipped.
    """
    result: dict[str, dict[str, np.ndarray]] = {}
    total_images = sum(len(imgs) for imgs in person_images.values())
    processed = 0

    for person, images in person_images.items():
        person_embs: dict[str, np.ndarray] = {}
        for img_path in images:
            processed += 1
            try:
                emb = recognizer.get_embedding_from_path(str(img_path))
                if emb is not None:
                    person_embs[img_path.name] = emb
            except ImageLoadError as err:
                logger.warning("Failed to load %s: %s", img_path, err)
            except Exception:
                logger.exception("Error processing %s", img_path)

            if processed % 5 == 0 or processed == total_images:
                print(f"  [{processed}/{total_images}] embeddings generated ...", end="\r")

        if person_embs:
            result[person] = person_embs

    print()  # Clear progress line
    return result


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
def run_tests(test_root: Path, max_persons: int = 5) -> None:
    """Run genuine/impostor comparison tests.

    Parameters
    ----------
    test_root : Path
        Root directory containing person_XX/ sub-folders.
    max_persons : int
        Maximum number of person folders to use. 0 = all.
    """
    if not test_root.exists():
        print(f"ERROR: Test directory does not exist: {test_root}")
        sys.exit(1)

    # ---- Initialise modules ----
    print("Initialising FaceRecognizer ...")
    recognizer = FaceRecognizer()
    sim = FaceSimilarity()
    print(f"  -> Execution provider: {recognizer.provider}\n")

    # ---- Collect images ----
    person_images = _collect_person_images(test_root, max_persons)
    if not person_images:
        print(f"No person folders found under {test_root}")
        sys.exit(1)

    print(f"Found {len(person_images)} person folders:")
    for name, imgs in person_images.items():
        print(f"  {name}: {len(imgs)} images")
    print()

    # ---- Generate embeddings ----
    print("Generating embeddings ...")
    embeddings = _generate_embeddings(recognizer, person_images)

    total_embs = sum(len(v) for v in embeddings.values())
    print(f"Successfully generated {total_embs} embeddings.\n")

    if len(embeddings) < 2:
        print("Need at least 2 persons with embeddings for comparison.")
        sys.exit(1)

    # ---- Genuine pairs (same person, different images) ----
    print("=" * 65)
    print("  GENUINE PAIRS (same person, different images)")
    print("=" * 65)

    genuine_scores: list[float] = []

    for person, embs in sorted(embeddings.items()):
        names = sorted(embs.keys())
        if len(names) < 2:
            continue

        print(f"\n  {person}:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                score = sim.cosine_similarity(embs[names[i]], embs[names[j]])
                genuine_scores.append(score)
                print(f"    {names[i]} vs {names[j]}:  cosine={score:.4f}")

    # ---- Impostor pairs (different people) ----
    print(f"\n{'=' * 65}")
    print("  IMPOSTOR PAIRS (different people)")
    print("=" * 65)

    impostor_scores: list[float] = []
    person_names = sorted(embeddings.keys())

    # For efficiency, compare only the FIRST image of each person
    # against the first image of every other person.
    for i in range(len(person_names)):
        for j in range(i + 1, len(person_names)):
            p1 = person_names[i]
            p2 = person_names[j]
            # Use the first available image from each person.
            img1_name = sorted(embeddings[p1].keys())[0]
            img2_name = sorted(embeddings[p2].keys())[0]
            emb1 = embeddings[p1][img1_name]
            emb2 = embeddings[p2][img2_name]

            score = sim.cosine_similarity(emb1, emb2)
            impostor_scores.append(score)
            print(f"  {p1}/{img1_name} vs {p2}/{img2_name}:  cosine={score:.4f}")

    # ---- Summary ----
    print(f"\n{'=' * 65}")
    print("  SUMMARY")
    print("=" * 65)

    if genuine_scores:
        g_min = min(genuine_scores)
        g_max = max(genuine_scores)
        g_mean = sum(genuine_scores) / len(genuine_scores)
        print(f"\n  Genuine pairs  ({len(genuine_scores):>3} pairs):")
        print(f"    Min:  {g_min:.4f}")
        print(f"    Max:  {g_max:.4f}")
        print(f"    Mean: {g_mean:.4f}")
    else:
        print("\n  No genuine pairs computed.")
        g_mean = None

    if impostor_scores:
        i_min = min(impostor_scores)
        i_max = max(impostor_scores)
        i_mean = sum(impostor_scores) / len(impostor_scores)
        print(f"\n  Impostor pairs ({len(impostor_scores):>3} pairs):")
        print(f"    Min:  {i_min:.4f}")
        print(f"    Max:  {i_max:.4f}")
        print(f"    Mean: {i_mean:.4f}")
    else:
        print("\n  No impostor pairs computed.")
        i_mean = None

    # Sanity check
    if g_mean is not None and i_mean is not None:
        gap = g_mean - i_mean
        print(f"\n  Gap (genuine_mean - impostor_mean): {gap:.4f}")
        if gap > 0.05:
            print("  >> PASS: Genuine scores are noticeably higher than impostor scores.")
        elif gap > 0:
            print("  >> MARGINAL: Small positive gap -- model may need tuning or more data.")
        else:
            print("  >> FAIL: Genuine scores are NOT higher than impostor scores.")
            print("     Something may be wrong with the pipeline.")

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        root = Path(sys.argv[1])
    else:
        candidates = [
            Path("data") / "test",
            Path("..") / "data" / "test",
        ]
        root = next((c for c in candidates if c.exists()), candidates[0])

    max_p = 0 if "--all" in sys.argv else 5

    print(f"Test root: {root.resolve()}")
    print(f"Max person folders: {'all' if max_p == 0 else max_p}\n")
    run_tests(root, max_persons=max_p)
