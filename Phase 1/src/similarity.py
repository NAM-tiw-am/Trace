"""
similarity.py -- Face Embedding Comparison Module for the Missing-Person
Identification Pipeline.

This module provides the ``FaceSimilarity`` class for comparing face
embeddings produced by the ``FaceRecognizer`` module.  It supports
cosine similarity and Euclidean distance, plus batch/pairwise operations
and full N x N similarity matrices.

Pipeline position::

    Embedding 1 + Embedding 2 -> FaceSimilarity -> Score

This module does NOT generate embeddings (that's ``recognizer.py``) and
does NOT make match/no-match threshold decisions (that belongs in
``pipeline.py`` after experimental threshold calibration).

All embeddings are expected to be **L2-normalised** (unit length) 1-D
numpy arrays.  The class validates this defensively.

Relationship between metrics for unit-normalised vectors
--------------------------------------------------------
For two unit vectors u and v::

    euclidean_distance(u, v)^2 = 2 - 2 * cosine_similarity(u, v)

So the two metrics are NOT independent -- they encode the same
geometric information.  Use whichever is more natural for your
downstream logic.

Typical usage::

    from similarity import FaceSimilarity

    sim = FaceSimilarity()
    score = sim.cosine_similarity(embedding_a, embedding_b)

Author:  Phase 1 -- Face Similarity Team
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FaceSimilarity
# ---------------------------------------------------------------------------
class FaceSimilarity:
    """Compare face embeddings using cosine similarity or Euclidean distance.

    All public methods accept 1-D numpy arrays (or array-like inputs that
    can be converted via ``np.asarray``).  Inputs are validated for shape,
    dimension consistency, and numeric sanity (no NaN/Inf).

    This class is intentionally stateless -- it carries no configuration.
    A class is used (rather than bare module functions) so that it can be
    cleanly dependency-injected and mocked in tests, and to keep the
    public API grouped under a single importable name.

    Examples
    --------
    >>> sim = FaceSimilarity()
    >>> score = sim.cosine_similarity(emb_a, emb_b)
    >>> dist  = sim.euclidean_distance(emb_a, emb_b)
    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_embedding(embedding: np.ndarray, name: str = "embedding") -> np.ndarray:
        """Validate and coerce a single embedding vector.

        Parameters
        ----------
        embedding : array-like
            Input embedding (1-D).
        name : str
            Human-readable label for error messages (e.g. ``"embedding1"``).

        Returns
        -------
        np.ndarray
            Validated 1-D float array.

        Raises
        ------
        ValueError
            If the input is empty, not 1-D, or contains NaN/Inf values.
        TypeError
            If the input cannot be converted to a numpy array.
        """
        try:
            arr = np.asarray(embedding, dtype=np.float64)
        except (ValueError, TypeError) as err:
            raise TypeError(
                f"{name}: cannot convert to numpy array -- {err}"
            ) from err

        if arr.ndim != 1:
            raise ValueError(
                f"{name}: expected a 1-D array, got shape {arr.shape}"
            )
        if arr.size == 0:
            raise ValueError(f"{name}: embedding is empty (length 0)")

        if np.any(np.isnan(arr)):
            raise ValueError(
                f"{name}: embedding contains NaN values"
            )
        if np.any(np.isinf(arr)):
            raise ValueError(
                f"{name}: embedding contains Inf values"
            )

        return arr

    @staticmethod
    def _validate_pair(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate a pair of embeddings for pairwise comparison.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Validated pair.

        Raises
        ------
        ValueError
            If dimensions don't match or either embedding is invalid.
        """
        e1 = FaceSimilarity._validate_embedding(embedding1, "embedding1")
        e2 = FaceSimilarity._validate_embedding(embedding2, "embedding2")

        if e1.shape[0] != e2.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch: embedding1 has {e1.shape[0]} "
                f"dimensions, embedding2 has {e2.shape[0]} dimensions. "
                f"This usually means embeddings were generated by different "
                f"models or model versions."
            )

        return e1, e2

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------
    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two embedding vectors.

        For L2-normalised embeddings (unit vectors), this is equivalent
        to the dot product.

        Parameters
        ----------
        embedding1 : array-like
            First embedding vector (1-D).
        embedding2 : array-like
            Second embedding vector (1-D).

        Returns
        -------
        float
            Similarity score in ``[-1.0, 1.0]``.  In practice, scores
            for L2-normalised face embeddings from the same model fall
            in ``[0.0, 1.0]``.  Higher means more similar.

        Raises
        ------
        ValueError
            If inputs are invalid or have mismatched dimensions.
        ValueError
            If either embedding is a zero vector (all zeros), since
            cosine similarity is undefined for zero vectors.
        """
        e1, e2 = self._validate_pair(embedding1, embedding2)

        norm1 = float(np.linalg.norm(e1))
        norm2 = float(np.linalg.norm(e2))

        if norm1 < 1e-10:
            raise ValueError(
                "embedding1 is a zero vector (L2 norm ~ 0). "
                "Cosine similarity is undefined for zero vectors. "
                "This likely indicates a degenerate upstream embedding."
            )
        if norm2 < 1e-10:
            raise ValueError(
                "embedding2 is a zero vector (L2 norm ~ 0). "
                "Cosine similarity is undefined for zero vectors. "
                "This likely indicates a degenerate upstream embedding."
            )

        # For unit vectors: cos_sim = dot(e1, e2).
        # We still divide by norms for correctness in case the caller
        # passes non-normalised vectors.
        similarity = float(np.dot(e1, e2) / (norm1 * norm2))

        # Clamp to [-1.0, 1.0] to guard against floating-point drift.
        return max(-1.0, min(1.0, similarity))

    def euclidean_distance(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute the Euclidean (L2) distance between two embedding vectors.

        Parameters
        ----------
        embedding1 : array-like
            First embedding vector (1-D).
        embedding2 : array-like
            Second embedding vector (1-D).

        Returns
        -------
        float
            Non-negative distance.  ``0.0`` means identical vectors.
            For L2-normalised vectors the maximum possible distance is
            ``2.0`` (diametrically opposite unit vectors).

        Raises
        ------
        ValueError
            If inputs are invalid or have mismatched dimensions.

        Notes
        -----
        For unit-normalised vectors, Euclidean distance and cosine
        similarity are related by::

            distance^2 = 2 - 2 * cosine_similarity

        They encode the same geometric information and are monotonically
        inversely related, so using one or the other is a matter of
        convention.
        """
        e1, e2 = self._validate_pair(embedding1, embedding2)
        return float(np.linalg.norm(e1 - e2))

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    _VALID_METRICS = frozenset({"cosine", "euclidean"})

    def compare(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine",
    ) -> float:
        """Compare two embeddings using the specified metric.

        This is a convenience dispatcher that calls either
        :meth:`cosine_similarity` or :meth:`euclidean_distance`.

        Parameters
        ----------
        embedding1 : array-like
            First embedding vector (1-D).
        embedding2 : array-like
            Second embedding vector (1-D).
        metric : str
            ``"cosine"`` (default) or ``"euclidean"``.

        Returns
        -------
        float
            The similarity score (cosine) or distance (euclidean).

        Raises
        ------
        ValueError
            If *metric* is not one of the supported values, or if
            inputs are invalid.
        """
        metric = metric.lower().strip()
        if metric not in self._VALID_METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Supported metrics: {sorted(self._VALID_METRICS)}"
            )

        if metric == "cosine":
            return self.cosine_similarity(embedding1, embedding2)
        else:
            return self.euclidean_distance(embedding1, embedding2)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------
    def compare_pairs(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
        metric: str = "cosine",
    ) -> list[float]:
        """Compare a list of embedding pairs.

        Each pair is processed independently.  If a pair fails validation,
        a warning is logged, that pair is **skipped**, and processing
        continues with the remaining pairs.

        .. warning::
            The output list may be **shorter** than the input list if any
            pairs were skipped due to validation errors.  Do NOT assume a
            1-to-1 index correspondence.  If you need to track which pairs
            succeeded, iterate manually with :meth:`compare` and handle
            exceptions per-pair.

        Parameters
        ----------
        pairs : list[tuple[array-like, array-like]]
            List of ``(embedding1, embedding2)`` tuples.
        metric : str
            ``"cosine"`` or ``"euclidean"``.

        Returns
        -------
        list[float]
            Scores for successfully compared pairs.
        """
        # Validate metric once up front.
        metric_lower = metric.lower().strip()
        if metric_lower not in self._VALID_METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Supported metrics: {sorted(self._VALID_METRICS)}"
            )

        results: list[float] = []
        for idx, pair in enumerate(pairs):
            try:
                score = self.compare(pair[0], pair[1], metric=metric_lower)
                results.append(score)
            except (ValueError, TypeError) as err:
                logger.warning(
                    "Skipping pair %d/%d due to validation error: %s",
                    idx + 1,
                    len(pairs),
                    err,
                )
        logger.info(
            "compare_pairs: %d/%d pairs compared successfully (metric=%s).",
            len(results),
            len(pairs),
            metric_lower,
        )
        return results

    def build_similarity_matrix(
        self,
        embeddings: list[np.ndarray],
        metric: str = "cosine",
    ) -> np.ndarray:
        """Build an N x N pairwise similarity/distance matrix.

        Parameters
        ----------
        embeddings : list[array-like]
            List of N embedding vectors.
        metric : str
            ``"cosine"`` or ``"euclidean"``.

        Returns
        -------
        np.ndarray
            Shape ``(N, N)`` matrix.  Entry ``[i][j]`` is the
            similarity (cosine) or distance (euclidean) between
            ``embeddings[i]`` and ``embeddings[j]``.

            - Cosine: diagonal is ``1.0`` (self-similarity), matrix is
              symmetric.
            - Euclidean: diagonal is ``0.0`` (self-distance), matrix is
              symmetric.

        Raises
        ------
        ValueError
            If fewer than 1 embedding is provided, if any embedding
            fails validation, or if embeddings have inconsistent
            dimensions.
        """
        metric_lower = metric.lower().strip()
        if metric_lower not in self._VALID_METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Supported metrics: {sorted(self._VALID_METRICS)}"
            )

        n = len(embeddings)
        if n == 0:
            raise ValueError("Cannot build similarity matrix from 0 embeddings.")

        # Validate all embeddings and stack into a matrix for vectorised ops.
        validated = []
        for i, emb in enumerate(embeddings):
            validated.append(
                self._validate_embedding(emb, name=f"embeddings[{i}]")
            )

        dim = validated[0].shape[0]
        for i, v in enumerate(validated):
            if v.shape[0] != dim:
                raise ValueError(
                    f"Dimension mismatch: embeddings[0] has {dim} dims, "
                    f"embeddings[{i}] has {v.shape[0]} dims."
                )

        # Stack into (N, D) matrix for efficient computation.
        mat = np.stack(validated)  # shape (N, D)

        if metric_lower == "cosine":
            # Cosine similarity for unit vectors: S = mat @ mat.T
            # For non-unit vectors, normalise row-wise first.
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            # Guard against zero vectors.
            zero_mask = norms.flatten() < 1e-10
            if np.any(zero_mask):
                zero_indices = np.where(zero_mask)[0].tolist()
                raise ValueError(
                    f"Zero-vector embedding(s) at index(es) {zero_indices}. "
                    f"Cosine similarity is undefined for zero vectors."
                )
            mat_normed = mat / norms
            result = mat_normed @ mat_normed.T
            # Clamp to [-1, 1] for floating-point safety.
            np.clip(result, -1.0, 1.0, out=result)
        else:
            # Euclidean distance matrix.
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b
            sq_norms = np.sum(mat ** 2, axis=1)
            dot_products = mat @ mat.T
            # Broadcast: dist_sq[i,j] = sq_norms[i] + sq_norms[j] - 2*dot[i,j]
            dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * dot_products
            # Guard against tiny negative values from floating-point drift.
            np.maximum(dist_sq, 0.0, out=dist_sq)
            result = np.sqrt(dist_sq)

        return result


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- SYNTHETIC embeddings for API demonstration only ----
    # These are NOT real face embeddings -- they are random vectors
    # used purely to show the API working end-to-end.
    np.random.seed(42)
    DIM = 512

    def _make_unit_vector(dim: int) -> np.ndarray:
        """Create a random L2-normalised vector."""
        v = np.random.randn(dim).astype(np.float64)
        return v / np.linalg.norm(v)

    emb_a = _make_unit_vector(DIM)
    emb_b = _make_unit_vector(DIM)
    emb_c = _make_unit_vector(DIM)
    emb_d = _make_unit_vector(DIM)

    sim = FaceSimilarity()

    # ---- Single-pair comparisons ----
    print(f"\n{'=' * 60}")
    print("  SINGLE-PAIR COMPARISON (synthetic embeddings)")
    print(f"{'=' * 60}")

    cos = sim.cosine_similarity(emb_a, emb_b)
    euc = sim.euclidean_distance(emb_a, emb_b)
    comp = sim.compare(emb_a, emb_b, metric="cosine")

    print(f"  cosine_similarity(A, B)  = {cos:.6f}")
    print(f"  euclidean_distance(A, B) = {euc:.6f}")
    print(f"  compare(A, B, 'cosine')  = {comp:.6f}")
    print(f"  Verify: euc^2 ~= 2 - 2*cos  ->  {euc**2:.6f} ~= {2 - 2*cos:.6f}")

    # ---- Similarity matrix ----
    labels = ["A", "B", "C", "D"]
    all_embs = [emb_a, emb_b, emb_c, emb_d]

    print(f"\n{'=' * 60}")
    print("  COSINE SIMILARITY MATRIX (4 synthetic embeddings)")
    print(f"{'=' * 60}")

    matrix = sim.build_similarity_matrix(all_embs, metric="cosine")
    # Pretty-print
    header = "       " + "  ".join(f"  {lbl:>5}" for lbl in labels)
    print(header)
    for i, row in enumerate(matrix):
        row_str = "  ".join(f"{v:>7.4f}" for v in row)
        print(f"  {labels[i]}    {row_str}")

    print(f"\n{'=' * 60}")
    print("  EUCLIDEAN DISTANCE MATRIX")
    print(f"{'=' * 60}")

    matrix_euc = sim.build_similarity_matrix(all_embs, metric="euclidean")
    print(header)
    for i, row in enumerate(matrix_euc):
        row_str = "  ".join(f"{v:>7.4f}" for v in row)
        print(f"  {labels[i]}    {row_str}")

    print()
