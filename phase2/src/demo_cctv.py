"""
demo_cctv.py — Full Phase 2 CCTV Detection Demo.

End-to-end demo that:
  1. Creates/loads the MissingPersonDB.
  2. Registers demo people using Phase 1 test images.
  3. Runs the CCTVSearchPipeline on the test video.
  4. Produces annotated_cctv.mp4 and match_log.json.
  5. Prints a clear summary.

Usage (from project root)::

    python -m phase2.src.demo_cctv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent  # missing-person-identification/


_project_root_str = str(PROJECT_ROOT)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

from phase2.src.database import MissingPersonDB  # noqa: E402
from phase2.src.cctv_search import CCTVSearchPipeline  # noqa: E402
from phase2.src.phase1_bridge import FaceDetector, FaceRecognizer  # noqa: E402

people = [
    ("P001", "Person 01", 30, "Phase 1/data/test/person_01/image_01.jpg"),
    ("P002","Person 02",30,"Phase 1/data/test/person_03/image_02.jpg")
]

# Paths
INPUT_VIDEO = PROJECT_ROOT / "phase2" / "videos" / "input" / "test_cctv.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "phase2" / "videos" / "output" / "annotated_cctv.mp4"
MATCH_LOG = PROJECT_ROOT / "phase2" / "results" / "match_log.json"
MATCH_THRESHOLD = 0.35


def main() -> int:
    """Run the full Phase 2 CCTV detection demo.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("demo_cctv")

    print(f"\n{'=' * 60}")
    print("  PHASE 2 — CCTV MISSING-PERSON DETECTION DEMO")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # 1. Check input video exists
    # ------------------------------------------------------------------
    if not INPUT_VIDEO.exists():
        print(f"  ERROR: Input video not found: {INPUT_VIDEO}")
        print("  Run 'python phase2/create_test_video.py' first.\n")
        return 1

    # ------------------------------------------------------------------
    # 2. Load / create the database
    # ------------------------------------------------------------------
    db = MissingPersonDB()
    db.clear()  # Clear old demo records

    # ------------------------------------------------------------------
    # 3. Create embeddings for demo people
    # ------------------------------------------------------------------
    logger.info("Creating embeddings for demo people …")
    detector = FaceDetector()
    recognizer = FaceRecognizer(detector=detector)

    loaded = 0
    for pid, name, age, rel_path in people:
        img_path = PROJECT_ROOT / rel_path
        if not img_path.exists():
            logger.warning("Image not found for %s: %s — skipping.", name, img_path)
            continue

        embedding = recognizer.get_embedding_from_path(str(img_path))
        if embedding is None:
            logger.warning("No face detected in %s — skipping.", img_path)
            continue

        db.add_person(pid, name, age, embedding, str(img_path))
        loaded += 1
        logger.info("  ✓ %s (%s) — embedding generated.", name, pid)

    db.save()

    if loaded == 0:
        print("  ERROR: No demo people could be loaded. Aborting.\n")
        return 1

    print(f"  People loaded  : {loaded}/{len(people)}")
    print(f"  Database path  : {db.db_path}")

    print(f"\n  Input video    : {INPUT_VIDEO}")
    print(f"  Output video   : {OUTPUT_VIDEO}")
    print(f"  Match threshold: {MATCH_THRESHOLD}")
    print()

    pipeline = CCTVSearchPipeline(
        db=db,
        match_threshold=MATCH_THRESHOLD,
        alert_cooldown_seconds=5.0,
    )

    matches = pipeline.process_video(
        video_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
    )

    # ------------------------------------------------------------------
    # 5. Save match log
    # ------------------------------------------------------------------
    log_path = pipeline.save_match_log(matches, output_path=MATCH_LOG)

    # ------------------------------------------------------------------
    # 6. Print summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  DEMO COMPLETE")
    print(f"{'=' * 60}")
    print(f"  People in DB   : {db.count}")
    print(f"  Input video    : {INPUT_VIDEO}")
    print(f"  Matches found  : {len(matches)}")
    print(f"  Output video   : {OUTPUT_VIDEO}")
    print(f"  Match log      : {log_path}")
    print(f"{'=' * 60}")

    if matches:
        print("\n  Match details:")
        for m in matches:
            print(
                f"    • {m['name']} (ID: {m['person_id']})  "
                f"score={m['similarity_score']:.4f}  "
                f"t={m['timestamp_seconds']:.2f}s  "
                f"frame={m['frame_number']}"
            )
    else:
        print("\n  No matches found.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
