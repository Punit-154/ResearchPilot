"""
Day 6: Query routes.
This is the first stage where you can actually ASK the system something,
rather than just inspecting ingested data.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.retrieval.lexical import lexical_search
from backend.retrieval.semantic import semantic_search
from backend.retrieval.hybrid import hybrid_search

router = APIRouter(prefix="/query", tags=["query"])


@router.get("/lexical")
def query_lexical(
    q: str = Query(..., description="Search query text"),
    limit: int = Query(default=10, le=50),
    paper_id: int | None = Query(default=None, description="Restrict to one paper"),
    db: Session = Depends(get_db),
):
    paper_ids = [paper_id] if paper_id is not None else None
    results = lexical_search(q, db, limit=limit, paper_ids=paper_ids)

    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "chunk_id": r.chunk_id,
                "paper_id": r.paper_id,
                "page": r.page,
                "rank": round(r.rank, 5),
                "text_preview": r.text[:250] + ("..." if len(r.text) > 250 else ""),
            }
            for r in results
        ],
    }


@router.get("/semantic")
def query_semantic(
    q: str = Query(..., description="Search query text"),
    limit: int = Query(default=10, le=50),
    paper_id: int | None = Query(default=None, description="Restrict to one paper"),
    db: Session = Depends(get_db),
):
    paper_ids = [paper_id] if paper_id is not None else None
    results = semantic_search(q, db, limit=limit, paper_ids=paper_ids)

    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "chunk_id": r.chunk_id,
                "paper_id": r.paper_id,
                "page": r.page,
                "similarity": round(r.similarity, 5),
                "text_preview": r.text[:250] + ("..." if len(r.text) > 250 else ""),
            }
            for r in results
        ],
    }


@router.get("/hybrid")
def query_hybrid(
    q: str = Query(..., description="Search query text"),
    limit: int = Query(default=10, le=50),
    paper_id: int | None = Query(default=None, description="Restrict to one paper"),
    db: Session = Depends(get_db),
):
    paper_ids = [paper_id] if paper_id is not None else None
    results = hybrid_search(q, db, limit=limit, paper_ids=paper_ids)

    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "chunk_id": r.chunk_id,
                "paper_id": r.paper_id,
                "page": r.page,
                "rrf_score": round(r.rrf_score, 6),
                "lexical_rank": r.lexical_rank,
                "semantic_rank": r.semantic_rank,
                "text_preview": r.text[:250] + ("..." if len(r.text) > 250 else ""),
            }
            for r in results
        ],
    }
