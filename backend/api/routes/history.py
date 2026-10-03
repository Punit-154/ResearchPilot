from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.database.models import QueryHistory

router = APIRouter(prefix="/history", tags=["history"])

@router.get("")
def get_history(
    paper_id: int | None = Query(default=None, description="Filter history by a specific paper ID"),
    db: Session = Depends(get_db)
):
    query = db.query(QueryHistory).order_by(QueryHistory.created_at.desc())
    if paper_id is not None:
        query = query.filter(QueryHistory.paper_ids.any(paper_id))
    
    records = query.all()
    
    return {
        "history": [
            {
                "id": record.id,
                "created_at": record.created_at,
                "question": record.question,
                "paper_ids": record.paper_ids,
                "paper_titles": record.paper_titles,
                "is_baseline": record.is_baseline
            }
            for record in records
        ]
    }

@router.get("/{history_id}")
def get_history_detail(history_id: int, db: Session = Depends(get_db)):
    record = db.query(QueryHistory).filter(QueryHistory.id == history_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="History record not found")

    # Return the full response exactly as it was generated
    return record.full_response

@router.delete("")
def clear_history(db: Session = Depends(get_db)):
    deleted = db.query(QueryHistory).delete()
    db.commit()
    return {"deleted_entries": deleted}
