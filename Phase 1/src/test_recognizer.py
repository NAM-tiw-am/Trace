"""
test_recognizer.py -- Integration test script for the FaceRecognizer module.

Scans a folder of test images organised as::

    data/test/
    +-- person_01/
    |   +-- image_01.jpg
    |   +-- image_02.jpg
    +-- person_02/
    |   +-- ...

For each image, generates an ArcFace embedding and prints a summary table.
Also performs a sanity check that embeddings for the SAME person across
different images are:
  1. The same vector length.
  2. NOT identical (i.e. different images produce different embeddings).

Usage::

    python test_recognizer.py                       # default paths
    python test_recognizer.py path/to/test_folder   # custom test root
    python test_recognizer.py path/to/test_folder --all  # all person folders
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Configure logging -- keep it clean so the table is readable
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
def _collect_images(root: Path) -> list[Path]:
    """Recursively collect image files under *root*.

    Supported extensions: .jpg, .jpeg, .png, .bmp, .webp.
    Results are sorted for deterministic output.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() in extensions:
                images.append(Path(dirpath) / fname)
    images.sort()
    return images


def _relative(path: Path, root: Path) -> str:
    """Return *path* relative to *root*, falling back to the full path."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _bbox_area(bbox: list[int]) -> int:
    """Compute area of a bounding box [x1, y1, x2, y2]."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
def run_tests(test_root: Path, max_persons: int = 10) -> None:
    """Run the recognizer against all images under *test_root*.

    Parameters
    ----------
    test_root : Path
        Root directory containing ``person_XX/`` sub-folders.
    max_persons : int
        Maximum number of ``person_XX`` folders to test.
        Set to ``0`` to test all.
    """
    if not test_root.exists():
        print(f"ERROR: Test directory does not exist: {test_root}")
        sys.exit(1)

    # ---- Initialise recognizer ----
    print("Initialising FaceRecognizer ...")
    recognizer = FaceRecognizer()
    print(f"  -> Execution provider: {recognizer.provider}\n")

    # ---- Collect test images ----
    all_images = _collect_images(test_root)
    if not all_images:
        print(f"No images found under {test_root}")
        sys.exit(1)

    # Optionally limit person folders
    if max_persons > 0:
        selected: list[Path] = []
        seen_persons: set[str] = set()
        for img in all_images:
            rel = img.relative_to(test_root)
            folder = rel.parts[0] if len(rel.parts) > 1 else ""
            if folder.startswith("person_"):
                if len(seen_persons) >= max_persons and folder not in seen_persons:
                    continue
                seen_persons.add(folder)
            selected.append(img)
        all_images = selected

    # ---- Generate embeddings ----
    # Store results grouped by person folder for the sanity check.
    results: list[dict] = []
    person_embeddings: dict[str, list[tuple[str, np.ndarray]]] = {}

    total = len(all_images)
    for idx, img_path in enumerate(all_images, start=1):
        rel_path = _relative(img_path, test_root)

        # Determine person folder name
        try:
            person_folder = img_path.relative_to(test_root).parts[0]
        except (ValueError, IndexError):
            person_folder = "unknown"

        try:
            embedding = recognizer.get_embedding_from_path(str(img_path))
            if embedding is not None:
                shape_str = str(embedding.shape)
                norm_str = f"{float(np.linalg.norm(embedding)):.4f}"
                status = "OK"
                # Record for sanity check
                person_embeddings.setdefault(person_folder, []).append(
                    (rel_path, embedding)
                )
            else:
                shape_str = "-"
                norm_str = "-"
                status = "NO FACE"
        except ImageLoadError as err:
            shape_str = "-"
            norm_str = "-"
            status = f"LOAD ERROR"
        except Exception as err:
            shape_str = "-"
            norm_str = "-"
            status = f"ERROR"

        results.append({
            "path": rel_path,
            "shape": shape_str,
            "norm": norm_str,
            "status": status,
        })

        # Progress indicator (every 10 images)
        if idx % 10 == 0 or idx == total:
            print(f"  [{idx}/{total}] processed ...", end="\r")

    print()  # Clear the progress line

    # ---- Print summary table ----
    col_path = max(len(r["path"]) for r in results)
    col_path = max(col_path, len("Image Path"))

    header = (
        f"  {'Image Path':<{col_path}}  |  Shape       |  L2 Norm  |  Status"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    ok_count = 0
    fail_count = 0

    for r in results:
        print(
            f"  {r['path']:<{col_path}}  |  {r['shape']:<11}  |  "
            f"{r['norm']:<8}  |  {r['status']}"
        )
        if r["status"] == "OK":
            ok_count += 1
        else:
            fail_count += 1

    print(separator)
    print(
        f"\n  Total: {len(results)}  |  "
        f"OK: {ok_count}  |  "
        f"Failed/No face: {fail_count}\n"
    )

    # ---- Sanity checks ----
    print("=" * 60)
    print("  SANITY CHECKS")
    print("=" * 60)

    all_passed = True

    for person, entries in sorted(person_embeddings.items()):
        if len(entries) < 2:
            continue

        print(f"\n  {person} ({len(entries)} embeddings):")

        # Check 1: All embeddings have the same shape.
        shapes = {e.shape for _, e in entries}
        if len(shapes) == 1:
            print(f"    [PASS] All embeddings have shape {shapes.pop()}")
        else:
            print(f"    [FAIL] Inconsistent shapes: {shapes}")
            all_passed = False

        # Check 2: Embeddings are NOT all identical.
        # Compare each pair; if ALL pairs are identical, that's a bug.
        all_identical = True
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                name_i, emb_i = entries[i]
                name_j, emb_j = entries[j]
                if not np.array_equal(emb_i, emb_j):
                    all_identical = False
                    break
            if not all_identical:
                break

        if not all_identical:
            print(
                "    [PASS] Different images produce different embeddings "
                "(not duplicated)"
            )
        else:
            print(
                "    [FAIL] All embeddings are identical -- possible bug "
                "(same array returned every time?)"
            )
            all_passed = False

        # Bonus: show pairwise cosine similarities to give a feel for
        # same-person consistency.
        if len(entries) >= 2:
            sims = []
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    sim = float(np.dot(entries[i][1], entries[j][1]))
                    sims.append(sim)
            avg_sim = sum(sims) / len(sims)
            min_sim = min(sims)
            max_sim = max(sims)
            print(
                f"    [INFO] Pairwise cosine similarity: "
                f"avg={avg_sim:.4f}  min={min_sim:.4f}  max={max_sim:.4f}"
            )

    print()
    if all_passed:
        print("  >> All sanity checks PASSED.")
    else:
        print("  >> Some sanity checks FAILED.  Review output above.")
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
