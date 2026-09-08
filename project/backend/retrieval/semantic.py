"""
Day 7: Semantic retrieval.
Embeds the user's query and finds chunks with the closest embedding
vectors using pgvector's cosine distance operator (<=>).

Since embeddings are normalized (normalize_embeddings=True in embeddings.py),
cosine distance and dot product rank identically, but we use cosine (<=>)
since it's the operator class our ivfflat index in schema.sql was built
with (vector_cosine_ops).
"""

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.nlp.embeddings import embed_text


@dataclass
class SemanticResult:
    chunk_id: int
    paper_id: int
    section_id: int | None
    page: int
    chunk_index: int
    text: str
    similarity: float  # 1 - cosine_distance; higher = more similar


def semantic_search(
    query: str,
    db: Session,
    limit: int = 20,
    paper_ids: list[int] | None = None,
) -> list[SemanticResult]:
    query_embedding = embed_text(query)

    base_sql = """
        SELECT
            id, paper_id, section_id, page, chunk_index, text,
            1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
        FROM chunks
        WHERE embedding IS NOT NULL
    """
    params = {"query_embedding": str(query_embedding), "limit": limit}

    if paper_ids:
        base_sql += " AND paper_id = ANY(:paper_ids)"
        params["paper_ids"] = paper_ids

    base_sql += " ORDER BY embedding <=> CAST(:query_embedding AS vector) LIMIT :limit"

    rows = db.execute(text(base_sql), params).fetchall()

    return [
        SemanticResult(
            chunk_id=row.id,
            paper_id=row.paper_id,
            section_id=row.section_id,
            page=row.page,
            chunk_index=row.chunk_index,
            text=row.text,
            similarity=float(row.similarity),
        )
        for row in rows
    ]
