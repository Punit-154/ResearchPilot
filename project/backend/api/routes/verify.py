"""
Day 9: Verification test route.
Lets us verify NLI + consensus work correctly BEFORE wiring in the LLM
(Day 10-11). Takes a query (to retrieve evidence) and an explicit claim
(normally this would come from LLM output later) and checks how well the
retrieved evidence supports that claim.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.retrieval.hybrid import hybrid_search
from backend.nlp.nli import check_entailment_batch
from backend.verification.consensus import compute_consensus

router = APIRouter(prefix="/verify", tags=["verification"])


@router.get("/claim")
def verify_claim(
    q: str = Query(..., description="Query used to retrieve evidence"),
    claim: str = Query(..., description="Claim to check evidence against"),
    top_k: int = Query(default=5, le=20),
    paper_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
):
    paper_ids = [paper_id] if paper_id is not None else None
    evidence = hybrid_search(q, db, limit=top_k, paper_ids=paper_ids)

    if not evidence:
        return {"claim": claim, "verdict": "INSUFFICIENT_EVIDENCE", "evidence": []}

    pairs = [(e.text, claim) for e in evidence]
    nli_results = check_entailment_batch(pairs)

    chunk_ids = [e.chunk_id for e in evidence]
    consensus = compute_consensus(chunk_ids, nli_results)

    # merge evidence text back in for a readable response
    evidence_by_id = {e.chunk_id: e for e in evidence}
    evidence_output = []
    for v in consensus.evidence_verdicts:
        e = evidence_by_id[v.chunk_id]
        evidence_output.append(
            {
                "chunk_id": v.chunk_id,
                "page": e.page,
                "nli_label": v.label,
                "confidence": round(v.confidence, 4),
                "text_preview": e.text[:200] + ("..." if len(e.text) > 200 else ""),
            }
        )

    return {
        "query": q,
        "claim": claim,
        "consensus_verdict": consensus.verdict,
        "support_count": consensus.support_count,
        "contradict_count": consensus.contradict_count,
        "neutral_count": consensus.neutral_count,
        "evidence": evidence_output,
    }
