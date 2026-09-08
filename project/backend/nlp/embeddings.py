"""
Day 7: Embedding system.
Wraps a local sentence-transformers model so the rest of the app doesn't
need to know encoding details. Model loads once (module-level singleton)
since loading it per-request would be extremely slow.

IMPORTANT: EMBEDDING_DIM must match the `vector(384)` column in schema.sql
(Day 1). If you ever swap to a different embedding model with a different
output dimension, you MUST update schema.sql's `vector(384)` too, or
inserts will fail.
"""

import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # must match schema.sql vector(384)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Loads the model once and caches it (first call pays the load cost)."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns a plain Python list (pgvector-friendly)."""
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Embed many strings at once — much faster than calling embed_text() in a
    loop, since the model batches internally. Used during ingestion when we
    embed every chunk of a paper.
    """
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
    return vectors.tolist()
