"""
Day 2: Ingestion API routes.
Separated from main.py so future stages can each own their route file
without merge conflicts / clutter in one giant main.py.
"""

import os
import shutil
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.ingestion.ingest import ingest_pdf

router = APIRouter(prefix="/ingest", tags=["ingestion"])

UPLOAD_DIR = os.path.join("data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/pdf")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    try:
        paper = ingest_pdf(save_path, db)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}")

    return {
        "paper_id": paper.id,
        "title": paper.title,
        "filename": paper.filename,
        "num_pages": paper.num_pages,
    }
