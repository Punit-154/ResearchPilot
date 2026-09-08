"""
Day 3: Paper inspection routes — lets us verify structure detection worked.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import Paper, Section, Chunk

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("/{paper_id}/sections")
def get_sections(paper_id: int, db: Session = Depends(get_db)):
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    sections = (
        db.query(Section)
        .filter(Section.paper_id == paper_id)
        .order_by(Section.section_order)
        .all()
    )

    return {
        "paper_id": paper.id,
        "title": paper.title,
        "num_pages": paper.num_pages,
        "sections": [
            {
                "name": s.name,
                "order": s.section_order,
                "start_page": s.start_page,
                "end_page": s.end_page,
            }
            for s in sections
        ],
    }


@router.get("/{paper_id}/chunks/count")
def get_chunk_count(paper_id: int, db: Session = Depends(get_db)):
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    count = db.query(Chunk).filter(Chunk.paper_id == paper_id).count()
    return {"paper_id": paper_id, "chunk_count": count}


@router.get("/{paper_id}/chunks")
def get_chunks(
    paper_id: int,
    limit: int = Query(default=10, le=100),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    chunks = (
        db.query(Chunk)
        .filter(Chunk.paper_id == paper_id)
        .order_by(Chunk.chunk_index)
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "paper_id": paper_id,
        "returned": len(chunks),
        "chunks": [
            {
                "chunk_index": c.chunk_index,
                "section_id": c.section_id,
                "page": c.page,
                "text_preview": c.text[:200] + ("..." if len(c.text) > 200 else ""),
            }
            for c in chunks
        ],
    }
