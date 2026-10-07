"""
database.py — Missing-Person Embedding Database for Phase 2.

Provides the ``MissingPersonDB`` class that persists person records
(ID, name, age, photo path, and face embedding) to a JSON file at
``phase2/data/missing_persons.json``.

Usage::

    from phase2.src.database import MissingPersonDB

    db = MissingPersonDB()
    db.add_person("P001", "Alice", 28, embedding_array, "photo.jpg")
    db.save()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent  # missing-person-identification/
DEFAULT_DB_PATH = PROJECT_ROOT / "phase2" / "data" / "missing_persons.json"


class MissingPersonDB:
    """Manage a JSON-persisted database of missing-person embeddings.

    Parameters
    ----------
    db_path : Path or str or None
        Path to the JSON database file.  Defaults to
        ``phase2/data/missing_persons.json`` relative to the project root.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._records: dict[str, dict] = {}
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load records from the JSON database file.

        Gracefully handles missing, empty, or invalid JSON files by
        starting with an empty database.
        """
        if not self._db_path.exists():
            logger.info(
                "Database file not found (%s). Starting empty.", self._db_path
            )
            self._records = {}
            return

        try:
            text = self._db_path.read_text(encoding="utf-8").strip()
            if not text:
                logger.info("Database file is empty. Starting empty.")
                self._records = {}
                return

            data = json.loads(text)
            if not isinstance(data, dict):
                logger.warning(
                    "Database JSON is not a dict (got %s). Starting empty.",
                    type(data).__name__,
                )
                self._records = {}
                return

            self._records = data
            logger.info(
                "Loaded %d person record(s) from %s.",
                len(self._records),
                self._db_path,
            )
        except (json.JSONDecodeError, OSError) as err:
            logger.warning(
                "Failed to load database (%s). Starting empty.", err
            )
            self._records = {}

    def save(self) -> None:
        """Persist current records to the JSON database file."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path.write_text(
            json.dumps(self._records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Saved %d record(s) to %s.", len(self._records), self._db_path
        )

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def add_person(
        self,
        person_id: str,
        name: str,
        age: int,
        embedding: np.ndarray,
        photo_path: str,
    ) -> None:
        """Add or update a person record in the database.

        Parameters
        ----------
        person_id : str
            Unique identifier for the person (must not be blank).
        name : str
            Person's name.
        age : int
            Person's age.
        embedding : np.ndarray
            Face embedding vector (1-D NumPy array).
        photo_path : str
            Path to the person's reference photo.

        Raises
        ------
        ValueError
            If ``person_id`` is blank.
        """
        if not person_id or not person_id.strip():
            raise ValueError("person_id must not be blank.")

        # Convert embedding to JSON-safe list of floats
        embedding_list = embedding.flatten().astype(float).tolist()

        self._records[person_id] = {
            "person_id": person_id,
            "name": name,
            "age": age,
            "photo_path": str(photo_path),
            "embedding": embedding_list,
        }
        logger.info("Added/updated person '%s' (ID: %s).", name, person_id)

    def get_person(self, person_id: str) -> Optional[dict]:
        """Retrieve a person record by ID.

        Returns
        -------
        dict or None
            The record dict with the ``embedding`` converted back to a
            NumPy array, or ``None`` if the person_id is not found.
        """
        record = self._records.get(person_id)
        if record is None:
            return None

        # Return a copy with embedding as NumPy array
        result = dict(record)
        result["embedding"] = np.array(
            record["embedding"], dtype=np.float32
        )
        return result

    def get_all_embeddings(
        self,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Return all stored embeddings and their corresponding person IDs.

        Returns
        -------
        embeddings : list[np.ndarray]
            List of 1-D NumPy embedding arrays.
        person_ids : list[str]
            Matching list of person IDs.
        """
        embeddings: list[np.ndarray] = []
        person_ids: list[str] = []

        for pid, record in self._records.items():
            emb = np.array(record["embedding"], dtype=np.float32)
            embeddings.append(emb)
            person_ids.append(pid)

        return embeddings, person_ids

    def clear(self) -> None:
        """Clear all records from the in-memory database."""
        count = len(self._records)
        self._records.clear()
        logger.info("Cleared %d record(s) from database.", count)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        """Return the number of records in the database."""
        return len(self._records)

    @property
    def db_path(self) -> Path:
        """Return the database file path."""
        return self._db_path

    def __repr__(self) -> str:
        return (
            f"MissingPersonDB(path={self._db_path}, "
            f"records={len(self._records)})"
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db = MissingPersonDB()
    db.clear()

    # Add a fake person with a random embedding
    fake_emb = np.random.randn(512).astype(np.float32)
    fake_emb = fake_emb / np.linalg.norm(fake_emb)  # L2-normalise
    db.add_person("TEST_001", "Test Person", 30, fake_emb, "test_photo.jpg")
    db.save()

    # Reload and verify
    db2 = MissingPersonDB()
    person = db2.get_person("TEST_001")
    assert person is not None, "Failed to retrieve test person"
    assert person["name"] == "Test Person"
    assert isinstance(person["embedding"], np.ndarray)
    assert person["embedding"].shape == (512,)
    print(f"\n  Self-test PASSED — saved and reloaded embedding "
          f"(shape={person['embedding'].shape})")

    # Clean up
    db2.clear()
    db2.save()
    print("  Cleaned up test data.\n")
