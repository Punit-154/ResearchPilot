"""
Day 11: Baseline endpoint.
Mirrors /answer's interface as closely as possible so the two can be
compared side-by-side on the same query.
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import Paper, QueryHistory
from backend.llm.baseline import run_baseline
from backend.llm.grok_client import GrokAPIError

router = APIRouter(prefix="/baseline", tags=["baseline"])


@router.get("/answer")
def get_baseline_answer(
    q: str = Query(..., description="Research question to answer"),
    top_k: int = Query(default=8, le=30),
    paper_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
):
    paper_ids = [paper_id] if paper_id is not None else None

    try:
        result = run_baseline(q, db, top_k=top_k, paper_ids=paper_ids)
    except GrokAPIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    response_data = {
        "query": q,
        "answer": result.answer,
        "retrieved_chunk_ids": result.retrieved_chunk_ids,
        "retrieved_pages": result.retrieved_pages,
    }

    paper_titles = []
    if paper_id is not None:
        paper = db.query(Paper).filter(Paper.id == paper_id).first()
        paper_titles.append(paper.title if paper else f"Unknown ({paper_id})")

    history_entry = QueryHistory(
        question=q,
        paper_ids=[paper_id] if paper_id is not None else [],
        paper_titles=paper_titles,
        full_response=response_data,
        is_baseline=True
    )
    db.add(history_entry)
    db.commit()

    return response_data
