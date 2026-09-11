"""
Day 10-11: The main end-to-end answer endpoint.
This is the culmination of the whole pipeline: hybrid retrieval -> evidence
pack -> LLM synthesis -> citation verification -> final answer with
per-claim validation status.
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import Paper
from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.entity_resolution import resolve_paper_names
from backend.llm.synthesis import synthesize_answer
from backend.llm.grok_client import GrokAPIError
from backend.verification.citation_verification import verify_claims

router = APIRouter(prefix="/answer", tags=["answer"])


@router.get("")
def get_answer(
    q: str = Query(..., description="Research question to answer"),
    top_k: int = Query(default=8, le=30, description="Number of evidence chunks to retrieve"),
    paper_id: list[int] | None = Query(
        default=None,
        description="Restrict to one or more papers by numeric ID. Repeat for multiple: "
        "?paper_id=1&paper_id=2",
    ),
    paper_name: list[str] | None = Query(
        default=None,
        description="Restrict to one or more papers by name instead of ID (e.g. "
        "?paper_name=ZJIT&paper_name=lazy%20basic%20block). Matches against the "
        "paper's title or auto-detected subject name. Can be combined with paper_id.",
    ),
    subject_name: str | None = Query(
        default=None,
        description="Paper's system/subject name (e.g. 'ZJIT') — improves citation "
        "verification accuracy. Usually NOT needed anymore: if omitted and exactly "
        "one paper is being queried, its auto-detected subject_name (Day 13) is used "
        "automatically. Only needed to override, or when querying multiple papers "
        "with different subjects (a known simplification — see docs).",
    ),
    db: Session = Depends(get_db),
):
    paper_ids = list(paper_id) if paper_id else []
    resolution_warnings = []

    if paper_name:
        resolved_ids, warnings = resolve_paper_names(paper_name, db)
        paper_ids.extend(resolved_ids)
        resolution_warnings.extend(warnings)

    paper_ids = list(dict.fromkeys(paper_ids)) or None  # dedupe, preserve order, empty->None

    # Auto subject_name: if the caller didn't specify one AND we're querying
    # exactly one paper, use that paper's auto-extracted subject_name
    # (Day 13) — removes the need to manually pass it for the common case.
    effective_subject_name = subject_name
    if effective_subject_name is None and paper_ids and len(paper_ids) == 1:
        paper_row = db.query(Paper).filter(Paper.id == paper_ids[0]).first()
        if paper_row and paper_row.subject_name:
            effective_subject_name = paper_row.subject_name

    evidence = hybrid_search(q, db, limit=top_k, paper_ids=paper_ids)

    # Fetch paper titles for evidence labeling (needed when evidence spans
    # multiple papers, so the LLM/human can tell sources apart)
    distinct_paper_ids = {e.paper_id for e in evidence}
    paper_titles = {}
    if distinct_paper_ids:
        papers = db.query(Paper).filter(Paper.id.in_(distinct_paper_ids)).all()
        paper_titles = {p.id: p.title for p in papers}

    try:
        synthesis = synthesize_answer(q, evidence, paper_titles=paper_titles)
    except GrokAPIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        # Grok returned non-JSON — surface it clearly rather than a generic 500
        raise HTTPException(status_code=502, detail=str(e))

    claim_verifications = verify_claims(
        synthesis.claims, synthesis.evidence_by_label, subject_name=effective_subject_name
    )

    # Overall citation health: what fraction of claims are actually backed
    # by their cited evidence, per Day 9's NLI verification.
    total_claims = len(claim_verifications)
    valid_claims = sum(1 for c in claim_verifications if c.status == "VALID")

    return {
        "query": q,
        "answer": synthesis.answer,
        "llm_reported_consensus": synthesis.consensus,
        "uncertainties": synthesis.uncertainties,
        "resolution_warnings": resolution_warnings,
        "subject_name_used": effective_subject_name,
        "claims": [
            {
                "id": cv.claim_id,
                "text": cv.claim_text,
                "citations": cv.citation_labels,
                "verification_status": cv.status,
                "citation_details": cv.per_citation_labels,
                "joint_citation_check": cv.joint_check,
            }
            for cv in claim_verifications
        ],
        "citation_health": {
            "total_claims": total_claims,
            "valid_claims": valid_claims,
            "flagged_claims": total_claims - valid_claims,
        },
        "evidence_used": [
            {
                "label": label,
                "chunk_id": item.chunk_id,
                "paper_id": item.paper_id,
                "paper_title": item.paper_title,
                "page": item.page,
            }
            for label, item in synthesis.evidence_by_label.items()
        ],
    }
