"""
Day 1: FastAPI skeleton.
Goal: prove the API layer is alive AND can talk to Postgres/pgvector.
Run with:  uvicorn backend.api.main:app --reload
"""

from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database.db import get_db, engine, Base
from backend.database import models  # noqa: F401  (ensures models are registered)
from backend.api.routes import ingestion, papers, query, verify

app = FastAPI(title="Research QA System")

app.include_router(ingestion.router)
app.include_router(papers.router)
app.include_router(query.router)
app.include_router(verify.router)


@app.on_event("startup")
def on_startup():
    # Creates tables if they don't exist yet (schema.sql already does this via
    # docker-entrypoint-initdb.d, but this makes local dev without Docker easier too)
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    """Confirms the API can reach Postgres AND that pgvector is installed."""
    result = db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
    has_pgvector = result.first() is not None
    return {"database": "connected", "pgvector_installed": has_pgvector}


@app.get("/papers/count")
def papers_count(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT COUNT(*) FROM papers;"))
    count = result.scalar()
    return {"papers_count": count}
