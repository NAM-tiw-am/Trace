"""
phase1_bridge.py — Safely import Phase 1 classes from the ``Phase 1/src`` directory.

The ``Phase 1`` folder name contains a space, which makes it invalid as
a Python package identifier.  Instead of fragile ``sys.path.insert``
hacks, this bridge uses ``importlib.util`` to load modules by absolute
file path, providing clean, relocatable access to Phase 1's
``FaceDetector``, ``FaceRecognizer``, and ``FaceSimilarity`` classes.

Usage::

    from phase2.src.phase1_bridge import FaceDetector, FaceRecognizer, FaceSimilarity
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate Phase 1 source directory
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent  # missing-person-identification/
PHASE1_SRC = PROJECT_ROOT / "Phase 1" / "src"


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    """Load a Python module from an absolute file path.

    Parameters
    ----------
    module_name : str
        Name to register the module under in ``sys.modules``.
    file_path : Path
        Absolute path to the ``.py`` file.

    Returns
    -------
    ModuleType
        The loaded module.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ImportError
        If the module cannot be loaded.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Phase 1 module not found: {file_path}"
        )

    # If already loaded, return cached version
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot create module spec for {file_path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Temporarily add Phase 1 src to sys.path so that Phase 1's own
    # internal imports (e.g. recognizer.py importing detector.py) work.
    phase1_src_str = str(PHASE1_SRC)
    added = False
    if phase1_src_str not in sys.path:
        sys.path.insert(0, phase1_src_str)
        added = True

    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        # Clean up: remove from sys.path only if WE added it
        if added and phase1_src_str in sys.path:
            sys.path.remove(phase1_src_str)

    logger.debug("Loaded Phase 1 module '%s' from %s", module_name, file_path)
    return module


# ---------------------------------------------------------------------------
# Load Phase 1 modules and re-export their public classes
# ---------------------------------------------------------------------------
_detector_mod = _load_module(
    "phase1_detector", PHASE1_SRC / "detector.py"
)
_recognizer_mod = _load_module(
    "phase1_recognizer", PHASE1_SRC / "recognizer.py"
)
_similarity_mod = _load_module(
    "phase1_similarity", PHASE1_SRC / "similarity.py"
)

# Public re-exports
FaceDetector = _detector_mod.FaceDetector
FaceRecognizer = _recognizer_mod.FaceRecognizer
FaceSimilarity = _similarity_mod.FaceSimilarity
ImageLoadError = _detector_mod.ImageLoadError

__all__ = [
    "FaceDetector",
    "FaceRecognizer",
    "FaceSimilarity",
    "ImageLoadError",
    "PROJECT_ROOT",
    "PHASE1_SRC",
]
