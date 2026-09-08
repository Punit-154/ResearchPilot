"""
Day 8: Hybrid retrieval.
Combines lexical (Day 6) and semantic (Day 7) results using Reciprocal
Rank Fusion (RRF) — a simple, well-established way to merge two ranked
lists without needing to normalize or compare their raw scores directly
(ts_rank and cosine similarity are on completely different scales, so
averaging them directly would be meaningless; RRF sidesteps that by only
using each result's RANK POSITION, not its raw score).

RRF formula per result:
    score = sum over each list it appears in of  1 / (k + rank_position)

where k is a small constant (60 is the standard default from the original
RRF paper) that dampens the influence of very top-ranked results slightly
and keeps the formula well-behaved.

A chunk that ranks well in BOTH lexical and semantic search will end up
with a higher fused score than one that only ranks well in one — which is
exactly the "lexical handles exact terms, semantic handles meaning,
together more robust" behavior described in the architecture doc.
"""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from backend.retrieval.lexical import lexical_search
from backend.retrieval.semantic import semantic_search

RRF_K = 60


@dataclass
class HybridResult:
    chunk_id: int
    paper_id: int
    section_id: int | None
    page: int
    chunk_index: int
    text: str
    rrf_score: float
    lexical_rank: int | None  # 1-indexed position in lexical list, or None if absent
    semantic_rank: int | None  # 1-indexed position in semantic list, or None if absent


def hybrid_search(
    query: str,
    db: Session,
    limit: int = 20,
    candidate_pool_size: int = 50,
    paper_ids: list[int] | None = None,
) -> list[HybridResult]:
    """
    Runs both lexical and semantic search (each returning up to
    `candidate_pool_size` results), fuses them with RRF, and returns the
    top `limit` fused results.

    candidate_pool_size is deliberately larger than `limit`: RRF benefits
    from having a decent-sized pool from each method to properly reward
    chunks that appear in both, rather than just the top few from each.
    """
    lexical_results = lexical_search(query, db, limit=candidate_pool_size, paper_ids=paper_ids)
    semantic_results = semantic_search(query, db, limit=candidate_pool_size, paper_ids=paper_ids)

    # chunk_id -> accumulated RRF score
    scores: dict[int, float] = {}
    # chunk_id -> (lexical_rank, semantic_rank), each possibly None
    ranks: dict[int, list[int | None]] = {}
    # chunk_id -> full result data (for building final output)
    chunk_data: dict[int, dict] = {}

    for i, r in enumerate(lexical_results):
        rank_pos = i + 1
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (RRF_K + rank_pos)
        ranks.setdefault(r.chunk_id, [None, None])[0] = rank_pos
        chunk_data[r.chunk_id] = {
            "paper_id": r.paper_id,
            "section_id": r.section_id,
            "page": r.page,
            "chunk_index": r.chunk_index,
            "text": r.text,
        }

    for i, r in enumerate(semantic_results):
        rank_pos = i + 1
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (RRF_K + rank_pos)
        ranks.setdefault(r.chunk_id, [None, None])[1] = rank_pos
        if r.chunk_id not in chunk_data:
            chunk_data[r.chunk_id] = {
                "paper_id": r.paper_id,
                "section_id": r.section_id,
                "page": r.page,
                "chunk_index": r.chunk_index,
                "text": r.text,
            }

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    results = []
    for chunk_id, score in fused:
        data = chunk_data[chunk_id]
        lex_rank, sem_rank = ranks[chunk_id]
        results.append(
            HybridResult(
                chunk_id=chunk_id,
                paper_id=data["paper_id"],
                section_id=data["section_id"],
                page=data["page"],
                chunk_index=data["chunk_index"],
                text=data["text"],
                rrf_score=score,
                lexical_rank=lex_rank,
                semantic_rank=sem_rank,
            )
        )

    return results
