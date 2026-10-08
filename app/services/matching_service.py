import logging
import json
from typing import List, Tuple
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.embedding import FaceEmbedding
from app.ai.similarity import FaceSimilarity
from app.core.database import is_sqlite
from app.core.config import settings

logger = logging.getLogger(__name__)

class MatchingService:
    @staticmethod
    def search_similar_persons(
        db: Session,
        query_embedding: np.ndarray,
        threshold: float = settings.MATCH_THRESHOLD,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Search for persons matching the query embedding.
        Returns a list of tuples: [(person_id, similarity_score), ...]
        where similarity_score >= threshold, sorted descending.
        """
        # Ensure query vector is unit normalized
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q = q / norm

        # If PostgreSQL with pgvector is active
        if not is_sqlite:
            try:
                # In pgvector: <=> operator is cosine distance: distance = 1 - cosine_similarity
                max_cosine_dist = 1.0 - threshold
                query_list = q.tolist()
                sql = text("""
                    SELECT person_id, (1 - (embedding <=> :query_vector::vector)) AS similarity
                    FROM face_embeddings
                    WHERE (embedding <=> :query_vector::vector) <= :max_dist
                    ORDER BY embedding <=> :query_vector::vector ASC
                    LIMIT :limit
                """)
                res = db.execute(sql, {
                    "query_vector": str(query_list),
                    "max_dist": max_cosine_dist,
                    "limit": top_k
                }).fetchall()

                return [(row[0], float(row[1])) for row in res]
            except Exception as e:
                logger.warning("PostgreSQL pgvector query error, falling back to python calculation: %s", e)

        # Fallback for SQLite or in-memory search
        all_embeddings = db.query(FaceEmbedding).all()
        candidates = []
        for emb_record in all_embeddings:
            raw_emb = emb_record.embedding
            if isinstance(raw_emb, str):
                try:
                    raw_emb = json.loads(raw_emb)
                except Exception:
                    continue
            if raw_emb is not None:
                candidates.append((emb_record.person_id, np.array(raw_emb, dtype=np.float32)))

        return FaceSimilarity.find_best_match(q, candidates, threshold=threshold)[:top_k]
