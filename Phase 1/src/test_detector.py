"""
test_detector.py — Integration test script for the FaceDetector module.

Scans a folder of test images organised as::

    data/test/
    ├── person_01/
    │   ├── image_01.jpg
    │   └── image_02.jpg
    ├── person_02/
    │   └── ...
    └── edge_cases/
        ├── multi_face.jpg    ← expect ≥ 2 faces
        └── no_face.jpg       ← expect 0 faces

Prints a clean summary table showing detection results and pass/fail
status for each image.

Usage::

    python test_detector.py                       # use default paths
    python test_detector.py path/to/test_folder   # custom test root
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the ``src/`` directory is importable when running from the repo root
# or from the ``src/`` directory itself.
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
_src_dir = _this_dir  # test_detector.py lives alongside detector.py in src/
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from detector import FaceDetector, ImageLoadError  # noqa: E402

# ---------------------------------------------------------------------------
# Configure logging — keep it clean so the table is readable
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expected outcomes for edge-case images
# ---------------------------------------------------------------------------
# Keys are **filenames** (not full paths).  Values are callables that receive
# the detection count and return True if the result is considered a "pass".
EDGE_CASE_EXPECTATIONS: dict[str, dict] = {
    "no_face.jpg": {
        "description": "Image with no face -> should detect 0",
        "check": lambda count: count == 0,
    },
    "no_face.png": {
        "description": "Image with no face -> should detect 0",
        "check": lambda count: count == 0,
    },
    "multi_face.jpg": {
        "description": "Image with multiple faces -> should detect >= 2",
        "check": lambda count: count >= 2,
    },
    "multi_face.png": {
        "description": "Image with multiple faces -> should detect >= 2",
        "check": lambda count: count >= 2,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _collect_images(root: Path) -> list[Path]:
    """Recursively collect image files under *root*.

    Supported extensions: ``.jpg``, ``.jpeg``, ``.png``, ``.bmp``, ``.webp``.
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


def _pass_fail(image_path: Path, num_faces: int) -> str:
    """Determine pass/fail status for an image based on expectations.

    For images inside an ``edge_cases`` folder, check against
    :data:`EDGE_CASE_EXPECTATIONS`.  For normal ``person_XX`` images,
    a pass means at least one face was detected.
    """
    fname = image_path.name.lower()

    # Is this an edge-case image?
    if "edge_cases" in image_path.parts or "edge_case" in image_path.parts:
        for pattern, spec in EDGE_CASE_EXPECTATIONS.items():
            if fname == pattern.lower():
                return "PASS" if spec["check"](num_faces) else "FAIL"
        # Unknown edge-case file — no expectation defined
        return "—"

    # Standard person image: expect ≥ 1 face
    return "PASS" if num_faces >= 1 else "FAIL"


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
def run_tests(test_root: Path, max_persons: int = 10) -> None:
    """Run the detector against all images under *test_root* and print a
    summary table.

    Parameters
    ----------
    test_root : Path
        Root directory containing ``person_XX/`` sub-folders and optionally
        an ``edge_cases/`` folder.
    max_persons : int
        Maximum number of ``person_XX`` folders to test (to keep runtime
        reasonable on large datasets).  Set to ``0`` to test all.
    """
    if not test_root.exists():
        print(f"ERROR: Test directory does not exist: {test_root}")
        sys.exit(1)

    # ---- Initialise detector ----
    print("Initialising FaceDetector ...")
    detector = FaceDetector()
    print(f"  -> Execution provider: {detector.provider}\n")

    # ---- Collect test images ----
    all_images = _collect_images(test_root)
    if not all_images:
        print(f"No images found under {test_root}")
        sys.exit(1)

    # Optionally limit person folders (but always include edge_cases)
    if max_persons > 0:
        selected: list[Path] = []
        seen_persons: set[str] = set()
        for img in all_images:
            # Find the immediate sub-folder name under test_root
            rel = img.relative_to(test_root)
            folder = rel.parts[0] if len(rel.parts) > 1 else ""

            if folder.startswith("person_"):
                if len(seen_persons) >= max_persons and folder not in seen_persons:
                    continue
                seen_persons.add(folder)
            selected.append(img)
        all_images = selected

    # ---- Run detection on every image ----
    results: list[dict] = []

    for img_path in all_images:
        rel_path = _relative(img_path, test_root)
        try:
            detections = detector.detect_from_path(str(img_path))
            num_faces = len(detections)
            confidences = [f"{d['confidence']:.4f}" for d in detections]
            error = ""
        except ImageLoadError as err:
            num_faces = -1
            confidences = []
            error = str(err)
        except Exception as err:  # noqa: BLE001
            num_faces = -1
            confidences = []
            error = f"Unexpected: {err}"

        status = _pass_fail(img_path, num_faces) if not error else "ERROR"

        results.append(
            {
                "path": rel_path,
                "num_faces": num_faces,
                "confidences": ", ".join(confidences) if confidences else "-",
                "status": status,
                "error": error,
            }
        )

    # ---- Print summary table ----
    col_path = max(len(r["path"]) for r in results)
    col_path = max(col_path, len("Image Path"))
    col_conf = max(len(r["confidences"]) for r in results)
    col_conf = max(col_conf, len("Confidence(s)"))

    header = (
        f"  {'Image Path':<{col_path}}  |  Faces  |  "
        f"{'Confidence(s)':<{col_conf}}  |  Status"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    pass_count = fail_count = error_count = skip_count = 0

    for r in results:
        faces_str = (
            f"{r['num_faces']:>5}" if r["num_faces"] >= 0 else "  ERR"
        )
        status = r["status"]
        line = (
            f"  {r['path']:<{col_path}}  | {faces_str}   |  "
            f"{r['confidences']:<{col_conf}}  |  {status}"
        )
        if r["error"]:
            line += f"  ({r['error']})"
        print(line)

        if status == "PASS":
            pass_count += 1
        elif status == "FAIL":
            fail_count += 1
        elif status == "ERROR":
            error_count += 1
        else:
            skip_count += 1

    print(separator)
    total = len(results)
    print(
        f"\n  Total: {total}  |  "
        f"PASS: {pass_count}  |  "
        f"FAIL: {fail_count}  |  "
        f"ERROR: {error_count}  |  "
        f"SKIPPED/UNKNOWN: {skip_count}\n"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Default test root is relative to the project structure
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        # Assume script is run from Phase 1/src/ or Phase 1/
        candidates = [
            Path("data") / "test",
            Path("..") / "data" / "test",
        ]
        root = next((c for c in candidates if c.exists()), candidates[0])

    # Optional: pass --all to test every person folder
    max_p = 0 if "--all" in sys.argv else 10

    print(f"Test root: {root.resolve()}")
    print(f"Max person folders: {'all' if max_p == 0 else max_p}\n")
    run_tests(root, max_persons=max_p)
