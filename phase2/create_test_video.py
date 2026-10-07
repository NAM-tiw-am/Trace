"""
create_test_video.py — Generate a synthetic CCTV test video from Phase 1 test images.

Reads valid images recursively from Phase 1/data/test, resizes each to
640×480, writes each frame three times (to simulate a static CCTV scene
per person), and outputs phase2/videos/input/test_cctv.mp4.

Usage (from project root):
    python phase2/create_test_video.py
"""

from __future__ import annotations

from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Paths — resolved from this file's location
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent  # missing-person-identification/

PHASE1_TEST_DIR = PROJECT_ROOT / "Phase 1" / "data" / "test"
OUTPUT_VIDEO = PROJECT_ROOT / "phase2" / "videos" / "input" / "test_cctv.mp4"

# Video settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 5
CODEC = "mp4v"
REPEATS_PER_IMAGE = 3

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(root: Path) -> list[Path]:
    """Recursively collect image files under *root*, sorted for determinism."""
    images = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return images


def main() -> None:
    # Validate source directory
    if not PHASE1_TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Phase 1 test directory not found: {PHASE1_TEST_DIR}"
        )

    # Collect images
    image_paths = collect_images(PHASE1_TEST_DIR)
    if not image_paths:
        raise RuntimeError(
            f"No images found under {PHASE1_TEST_DIR}. "
            "Cannot create test video."
        )

    # Ensure output directory exists
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to create video writer for {OUTPUT_VIDEO}. "
            "Check codec availability."
        )

    total_frames = 0
    source_count = 0

    try:
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  WARNING: skipping unreadable image: {img_path}")
                continue

            # Resize to target frame dimensions
            resized = cv2.resize(
                frame, (FRAME_WIDTH, FRAME_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            # Write each image multiple times to simulate static scene
            for _ in range(REPEATS_PER_IMAGE):
                writer.write(resized)
                total_frames += 1

            source_count += 1
    finally:
        writer.release()

    print(f"\n{'=' * 60}")
    print(f"  Test CCTV video created successfully")
    print(f"{'=' * 60}")
    print(f"  Output path    : {OUTPUT_VIDEO}")
    print(f"  Source images   : {source_count}")
    print(f"  Total frames   : {total_frames}")
    print(f"  FPS            : {FPS}")
    print(f"  Resolution     : {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Duration       : {total_frames / FPS:.1f} seconds")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
