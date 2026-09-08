"""
Day 10 (Stage 22): Citation verification.
The LLM might cite evidence that's topically relevant but doesn't actually
support what it claimed (a common LLM failure mode: plausible-sounding
citation, wrong actual support). This module catches that by re-running
NLI between each claim and its cited evidence — using the same sentence
selection + pronoun resolution fixes validated in Day 9.
"""

from dataclasses import dataclass

from backend.llm.synthesis import Claim, EvidenceItem
from backend.nlp.nli import check_entailment_batch


@dataclass
class ClaimVerification:
    claim_id: str
    claim_text: str
    citation_labels: list[str]
    status: str  # "VALID" | "FLAGGED" | "NO_CITATIONS"
    per_citation_labels: list[dict]  # [{"label": "E01", "nli_label": "entailment", "confidence": 0.9}]


def verify_claims(
    claims: list[Claim],
    evidence_by_label: dict[str, EvidenceItem],
    subject_name: str | None = None,
) -> list[ClaimVerification]:
    """
    For each claim, checks whether ITS OWN cited evidence actually entails
    the claim text. A claim is:
      - "NO_CITATIONS" if the LLM cited nothing (should not happen given
        our system prompt, but defensively handled)
      - "FLAGGED" if none of its citations entail the claim (possible
        hallucinated support, or the LLM cited the wrong passage)
      - "VALID" if at least one citation entails the claim
    """
    results = []

    # Collect all (premise, hypothesis) pairs across all claims for one
    # efficient batched NLI call, tracking which claim/citation each pair
    # belongs to so we can regroup results afterward.
    pairs = []
    pair_owners = []  # (claim_index, label)

    for claim_idx, claim in enumerate(claims):
        for label in claim.citation_labels:
            evidence_item = evidence_by_label.get(label)
            if evidence_item is None:
                continue  # LLM cited a label that doesn't exist — skip, handled below
            pairs.append((evidence_item.text, claim.text))
            pair_owners.append((claim_idx, label))

    nli_results = check_entailment_batch(pairs, subject_name=subject_name, top_n_sentences=3) if pairs else []

    # regroup by claim
    per_claim_citation_results: dict[int, list[dict]] = {i: [] for i in range(len(claims))}
    for (claim_idx, label), nli in zip(pair_owners, nli_results):
        per_claim_citation_results[claim_idx].append(
            {"label": label, "nli_label": nli.label, "confidence": round(nli.confidence, 4)}
        )

    for claim_idx, claim in enumerate(claims):
        citation_results = per_claim_citation_results[claim_idx]

        if not claim.citation_labels:
            status = "NO_CITATIONS"
        elif any(c["nli_label"] == "entailment" for c in citation_results):
            status = "VALID"
        else:
            status = "FLAGGED"

        results.append(
            ClaimVerification(
                claim_id=claim.id,
                claim_text=claim.text,
                citation_labels=claim.citation_labels,
                status=status,
                per_citation_labels=citation_results,
            )
        )

    return results
