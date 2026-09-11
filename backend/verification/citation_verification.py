"""
Day 10 (Stage 22): Citation verification.
The LLM might cite evidence that's topically relevant but doesn't actually
support what it claimed (a common LLM failure mode: plausible-sounding
citation, wrong actual support). This module catches that by re-running
NLI between each claim and its cited evidence — using the same sentence
selection + pronoun resolution fixes validated in Day 9.

Day 15 (accuracy improvement, Fix 1): many claims combine facts from
MULTIPLE citations (e.g. citing both E04 and E08), where neither citation
ALONE fully supports the claim, even though the claim is genuinely true
given both together. The original design only checked each citation
individually. This adds a second check: for claims with 2+ citations, the
cited evidence is also combined into one joint premise and checked as a
whole — a claim is VALID if EITHER an individual citation entails it OR
the combined evidence does. This directly targets compound claims without
loosening what counts as "supported" — we still only use evidence the LLM
itself cited, just combined correctly.
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
    joint_check: dict | None = None  # result of checking all citations combined, if 2+ citations


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
      - "FLAGGED" if neither any individual citation NOR the combined
        citations entail the claim (possible hallucinated support, or the
        LLM cited the wrong passage)
      - "VALID" if at least one citation entails the claim ON ITS OWN, OR
        the citations TOGETHER entail the claim (Day 15 Fix 1 — see module
        docstring)
    """
    results = []

    # ── Pass 1: individual citation checks (original behavior) ──────────
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

    per_claim_citation_results: dict[int, list[dict]] = {i: [] for i in range(len(claims))}
    for (claim_idx, label), nli in zip(pair_owners, nli_results):
        per_claim_citation_results[claim_idx].append(
            {"label": label, "nli_label": nli.label, "confidence": round(nli.confidence, 4)}
        )

    # ── Pass 2 (Day 15 Fix 1): joint check for claims with 2+ citations ──
    # Combine all cited evidence texts into one premise and check as a
    # whole — this surfaces compound facts that no single citation alone
    # contains, without introducing any evidence the LLM didn't itself cite.
    joint_pairs = []
    joint_owners = []  # claim_index

    for claim_idx, claim in enumerate(claims):
        valid_labels = [l for l in claim.citation_labels if l in evidence_by_label]
        if len(valid_labels) >= 2:
            combined_text = " ".join(evidence_by_label[l].text for l in valid_labels)
            joint_pairs.append((combined_text, claim.text))
            joint_owners.append(claim_idx)

    # Use a slightly larger sentence pool for joint checks since the
    # combined premise pulls from more source material than a single chunk.
    joint_nli_results = (
        check_entailment_batch(joint_pairs, subject_name=subject_name, top_n_sentences=4)
        if joint_pairs
        else []
    )

    joint_result_by_claim: dict[int, dict] = {}
    for claim_idx, nli in zip(joint_owners, joint_nli_results):
        joint_result_by_claim[claim_idx] = {
            "nli_label": nli.label,
            "confidence": round(nli.confidence, 4),
        }

    # ── Combine both passes into final verdicts ──────────────────────────
    for claim_idx, claim in enumerate(claims):
        citation_results = per_claim_citation_results[claim_idx]
        joint_result = joint_result_by_claim.get(claim_idx)

        individual_entails = any(c["nli_label"] == "entailment" for c in citation_results)
        joint_entails = joint_result is not None and joint_result["nli_label"] == "entailment"

        if not claim.citation_labels:
            status = "NO_CITATIONS"
        elif individual_entails or joint_entails:
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
                joint_check=joint_result,
            )
        )

    return results
