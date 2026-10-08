from __future__ import annotations
import logging
from typing import Sequence
import numpy as np

logger = logging.getLogger(__name__)

class FaceSimilarity:
    """Compare face embeddings using cosine similarity or Euclidean distance."""

    @staticmethod
    def cosine_similarity(u: np.ndarray | Sequence[float], v: np.ndarray | Sequence[float]) -> float:
        """Compute cosine similarity between two 1-D vectors."""
        a = np.asarray(u, dtype=np.float32).ravel()
        b = np.asarray(v, dtype=np.float32).ravel()

        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-7 or norm_b < 1e-7:
            return 0.0

        dot = np.dot(a, b)
        similarity = float(dot / (norm_a * norm_b))
        # Clip numerical inaccuracies
        return max(-1.0, min(1.0, similarity))

    @staticmethod
    def euclidean_distance(u: np.ndarray | Sequence[float], v: np.ndarray | Sequence[float]) -> float:
        """Compute Euclidean distance between two vectors."""
        a = np.asarray(u, dtype=np.float32).ravel()
        b = np.asarray(v, dtype=np.float32).ravel()
        return float(np.linalg.norm(a - b))

    @staticmethod
    def find_best_match(
        query_embedding: np.ndarray,
        candidate_embeddings: list[tuple[int, np.ndarray]],
        threshold: float = 0.35,
    ) -> list[tuple[int, float]]:
        """
        Compare query against a list of (candidate_id, embedding) tuples.
        Returns sorted list of (candidate_id, score) where score >= threshold.
        """
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        norm_q = np.linalg.norm(q)
        if norm_q > 1e-6:
            q = q / norm_q

        results = []
        for cand_id, emb in candidate_embeddings:
            c = np.asarray(emb, dtype=np.float32).ravel()
            norm_c = np.linalg.norm(c)
            if norm_c > 1e-6:
                c = c / norm_c
            score = float(np.dot(q, c))
            if score >= threshold:
                results.append((cand_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
