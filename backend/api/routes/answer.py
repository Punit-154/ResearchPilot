"""
Day 10-11: The main end-to-end answer endpoint.
This is the culmination of the whole pipeline: hybrid retrieval -> evidence
pack -> LLM synthesis -> citation verification -> final answer with
per-claim validation status.
"""

from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import uuid
import json

from backend.database.db import get_db, SessionLocal
from backend.database.models import Paper, QueryHistory
from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.entity_resolution import resolve_paper_names
from backend.llm.synthesis import synthesize_answer, stream_synthesize_answer
from backend.llm.decomposition import decompose_claims
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
    use_claim_decomposition: bool = Query(
        default=False,
        description="When True, a post-generation decomposition step splits any "
        "compound LLM claims into atomic sub-facts before NLI verification. Adds "
        "~2s and one extra LLM call per answer. Defaults OFF for A/B testing — "
        "do not set as default until evaluation confirms improvement.",
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
    # Also fetch titles for the specifically queried paper_ids even if no evidence was found for them
    query_paper_ids = set(paper_ids) if paper_ids else set()
    all_needed_titles = distinct_paper_ids | query_paper_ids

    paper_titles = {}
    paper_subject_names = {}
    if all_needed_titles:
        papers = db.query(Paper).filter(Paper.id.in_(all_needed_titles)).all()
        paper_titles = {p.id: p.title for p in papers}
        paper_subject_names = {p.id: p.subject_name for p in papers}

    try:
        synthesis = synthesize_answer(q, evidence, paper_titles=paper_titles)
    except GrokAPIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        # Grok returned non-JSON — surface it clearly rather than a generic 500
        raise HTTPException(status_code=502, detail=str(e))

    # ── Optional post-generation claim decomposition ─────────────────────
    # Runs BETWEEN synthesis and verification so verify_claims only ever
    # sees the post-decomposition (now-atomic) claim list. Fix 1's joint
    # citation check therefore operates per sub-claim, not per original
    # compound claim — which is the correct behaviour.
    if use_claim_decomposition:
        try:
            claims_to_verify = decompose_claims(synthesis.claims)
        except (GrokAPIError, ValueError) as e:
            # Decomposition failure is non-fatal: fall back to original claims
            # rather than returning a 502 — the answer is still usable.
            print(f"[answer] Decomposition failed, falling back to original claims: {e}")
            claims_to_verify = synthesis.claims
    else:
        claims_to_verify = synthesis.claims

    claim_verifications = verify_claims(
        claims_to_verify,
        synthesis.evidence_by_label,
        subject_name=effective_subject_name,
        subject_names_by_paper_id=paper_subject_names,
    )

    # Overall citation health: what fraction of claims are actually backed
    # by their cited evidence, per Day 9's NLI verification.
    total_claims = len(claim_verifications)
    valid_claims = sum(1 for c in claim_verifications if c.status == "VALID")

    response_data = {
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
                "parent_claim_id": cv.parent_claim_id,  # None for atomic/top-level claims
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
                "text": item.text,
            }
            for label, item in synthesis.evidence_by_label.items()
        ],
    }

    history_entry = QueryHistory(
        question=q,
        paper_ids=paper_ids if paper_ids else [],
        paper_titles=[paper_titles.get(pid, f"Unknown ({pid})") for pid in (paper_ids or [])],
        full_response=response_data,
        is_baseline=False
    )
    db.add(history_entry)
    db.commit()

    return response_data


# ── Streaming / Async Verification ──────────────────────────────────────────

VERIFICATION_TASKS = {}


def async_verify_and_save(
    query_id: str,
    q: str,
    synthesis,
    use_claim_decomposition: bool,
    effective_subject_name: str | None,
    paper_subject_names: dict,
    paper_ids: list[int],
    paper_titles: dict,
):
    try:
        # Run decomposition if requested
        if use_claim_decomposition:
            try:
                claims_to_verify = decompose_claims(synthesis.claims)
            except Exception as e:
                print(f"[async_verify] Decomposition failed: {e}")
                claims_to_verify = synthesis.claims
        else:
            claims_to_verify = synthesis.claims

        claim_verifications = verify_claims(
            claims_to_verify,
            synthesis.evidence_by_label,
            subject_name=effective_subject_name,
            subject_names_by_paper_id=paper_subject_names,
        )

        total_claims = len(claim_verifications)
        valid_claims = sum(1 for c in claim_verifications if c.status == "VALID")
        resolution_warnings = [] # Simplified for async

        response_data = {
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
                    "parent_claim_id": cv.parent_claim_id,
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
                    "text": item.text,
                }
                for label, item in synthesis.evidence_by_label.items()
            ],
        }

        # Save to DB
        with SessionLocal() as db:
            history_entry = QueryHistory(
                question=q,
                paper_ids=paper_ids if paper_ids else [],
                paper_titles=[paper_titles.get(pid, f"Unknown ({pid})") for pid in (paper_ids or [])],
                full_response=response_data,
                is_baseline=False
            )
            db.add(history_entry)
            db.commit()
            
        VERIFICATION_TASKS[query_id] = {"status": "done", "result": response_data}
    except Exception as e:
        import traceback
        traceback.print_exc()
        VERIFICATION_TASKS[query_id] = {"status": "error", "error": str(e)}


@router.get("/stream")
def stream_answer(
    background_tasks: BackgroundTasks,
    q: str = Query(...),
    top_k: int = Query(default=8, le=30),
    paper_id: list[int] | None = Query(default=None),
    use_claim_decomposition: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    """
    Phase 1: Stream the answer paragraph text as SSE.
    Once the stream ends, kicks off background verification.
    """
    query_id = str(uuid.uuid4())
    VERIFICATION_TASKS[query_id] = {"status": "processing", "result": None}

    paper_ids = list(dict.fromkeys(paper_id)) if paper_id else None

    # Auto subject_name
    effective_subject_name = None
    if paper_ids and len(paper_ids) == 1:
        paper_row = db.query(Paper).filter(Paper.id == paper_ids[0]).first()
        if paper_row and paper_row.subject_name:
            effective_subject_name = paper_row.subject_name

    evidence = hybrid_search(q, db, limit=top_k, paper_ids=paper_ids)

    distinct_paper_ids = {e.paper_id for e in evidence}
    query_paper_ids = set(paper_ids) if paper_ids else set()
    all_needed_titles = distinct_paper_ids | query_paper_ids
    
    paper_titles = {}
    paper_subject_names = {}
    if all_needed_titles:
        papers = db.query(Paper).filter(Paper.id.in_(all_needed_titles)).all()
        paper_titles = {p.id: p.title for p in papers}
        paper_subject_names = {p.id: p.subject_name for p in papers}

    def generate():
        # First send the query ID so frontend knows where to poll
        yield f"event: metadata\ndata: {json.dumps({'query_id': query_id})}\n\n"
        
        try:
            stream_gen = stream_synthesize_answer(q, evidence, paper_titles=paper_titles)
            final_result = None
            for event_type, data in stream_gen:
                if event_type == 'answer_chunk':
                    # Send text chunk
                    yield f"event: text\ndata: {json.dumps({'chunk': data})}\n\n"
                elif event_type == 'result':
                    final_result = data
            
            if final_result:
                background_tasks.add_task(
                    async_verify_and_save,
                    query_id, q, final_result, use_claim_decomposition,
                    effective_subject_name, paper_subject_names, paper_ids, paper_titles
                )
            else:
                VERIFICATION_TASKS[query_id] = {"status": "error", "error": "No final result generated."}
        except Exception as e:
            VERIFICATION_TASKS[query_id] = {"status": "error", "error": str(e)}
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/verify/{query_id}")
def poll_verification(query_id: str):
    """
    Phase 2: Poll for the results of the async verification.
    """
    if query_id not in VERIFICATION_TASKS:
        raise HTTPException(status_code=404, detail="Query ID not found.")
    
    task = VERIFICATION_TASKS[query_id]
    if task["status"] == "done":
        # Clear it from memory to prevent leaks
        del VERIFICATION_TASKS[query_id]
        return {"status": "done", "result": task["result"]}
    elif task["status"] == "error":
        err = task["error"]
        del VERIFICATION_TASKS[query_id]
        raise HTTPException(status_code=500, detail=err)
    else:
        return {"status": "processing"}
