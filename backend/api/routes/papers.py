"""
Day 3: Paper inspection routes — lets us verify structure detection worked.
"""

import os
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import Paper, Section, Chunk
from backend.llm.paper_analysis import extract_subject_name
from backend.llm.grok_client import GrokAPIError

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("")
def list_papers(db: Session = Depends(get_db)):
    """Day 13: list all ingested papers with their auto-detected subject_name."""
    papers = db.query(Paper).order_by(Paper.id).all()
    return {
        "papers": [
            {
                "paper_id": p.id,
                "title": p.title,
                "subject_name": p.subject_name,
                "num_pages": p.num_pages,
                "filename": p.filename,
            }
            for p in papers
        ]
    }


@router.delete("")
def clear_all_papers(
    confirm: bool = Query(
        default=False,
        description="Must be explicitly set to true — this permanently deletes ALL "
        "papers, sections, chunks, and their uploaded PDF files.",
    ),
    db: Session = Depends(get_db),
):
    """
    Day 14: delete ALL papers (and their sections/chunks via cascading FK
    delete, per schema.sql) plus their uploaded PDF files on disk. Requires
    confirm=true to prevent accidental calls (e.g. from browser tab restore
    triggering a GET on a bookmarked DELETE URL, or a client bug).
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="This deletes ALL papers permanently. Call with ?confirm=true to proceed.",
        )

    papers = db.query(Paper).all()
    deleted_count = len(papers)

    # Remove uploaded PDF files from disk (best-effort — a missing file
    # shouldn't block clearing the database records)
    for p in papers:
        try:
            if p.filepath and os.path.exists(p.filepath):
                os.remove(p.filepath)
        except OSError:
            pass  # non-fatal — file may already be gone or locked

    # Cascading delete: schema.sql defines sections/chunks with
    # ON DELETE CASCADE on paper_id, so deleting Paper rows removes their
    # sections and chunks automatically at the database level.
    db.query(Paper).delete()

    # Reset the auto-increment counter — DELETE alone doesn't do this in
    # Postgres (by design, to avoid ID reuse conflicts in normal operation),
    # but for a clean demo it's nicer if the next upload starts at id=1
    # again instead of continuing from wherever the counter was.
    db.execute(text("ALTER SEQUENCE papers_id_seq RESTART WITH 1"))

    db.commit()

    return {"deleted_papers": deleted_count, "message": "All papers, sections, and chunks cleared. Next upload will start at id=1."}


@router.post("/{paper_id}/extract-subject-name")
def backfill_subject_name(paper_id: int, db: Session = Depends(get_db)):
    """
    Day 13: manually (re-)run subject_name extraction for a paper that's
    already ingested — much faster than re-uploading the whole PDF when
    you just want to retry after a rate-limit failure, or backfill it for
    papers ingested before this feature existed.
    """
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    if not paper.abstract:
        raise HTTPException(status_code=400, detail="Paper has no stored abstract to extract from")

    try:
        subject_name = extract_subject_name(paper.title, paper.abstract)
    except GrokAPIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    paper.subject_name = subject_name
    db.commit()
    db.refresh(paper)

    return {"paper_id": paper.id, "title": paper.title, "subject_name": paper.subject_name}


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
