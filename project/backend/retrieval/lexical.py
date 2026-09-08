"""
Day 6: Lexical retrieval.
Uses Postgres full-text search (the `search_vector` tsvector column, kept
in sync by a DB trigger since Day 1's schema.sql) to find chunks matching
a query's exact terms — critical for technical terms like "BLEU-4" or
"RoBERTa" that semantic/embedding search alone can under-rank.

We use `plainto_tsquery` (handles arbitrary user text safely, no query
syntax the user has to know) and `ts_rank` for relevance scoring.
"""

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass
class LexicalResult:
    chunk_id: int
    paper_id: int
    section_id: int | None
    page: int
    chunk_index: int
    text: str
    rank: float


def lexical_search(
    query: str,
    db: Session,
    limit: int = 20,
    paper_ids: list[int] | None = None,
) -> list[LexicalResult]:
    """
    Full-text search chunks.text against `query`, ranked by ts_rank.
    Optionally restrict to a set of paper_ids (used later by the retrieval
    planner for "compare Paper A and Paper B"-style filtering).
    """
    base_sql = """
        SELECT
            id, paper_id, section_id, page, chunk_index, text,
            ts_rank(search_vector, plainto_tsquery('english', :query)) AS rank
        FROM chunks
        WHERE search_vector @@ plainto_tsquery('english', :query)
    """
    params = {"query": query, "limit": limit}

    if paper_ids:
        base_sql += " AND paper_id = ANY(:paper_ids)"
        params["paper_ids"] = paper_ids

    base_sql += " ORDER BY rank DESC LIMIT :limit"

    rows = db.execute(text(base_sql), params).fetchall()

    return [
        LexicalResult(
            chunk_id=row.id,
            paper_id=row.paper_id,
            section_id=row.section_id,
            page=row.page,
            chunk_index=row.chunk_index,
            text=row.text,
            rank=float(row.rank),
        )
        for row in rows
    ]
