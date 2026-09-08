"""
Day 9: Consensus / contradiction analysis.
Aggregates several (evidence, claim) NLI verdicts into one overall picture.
Used two ways in this system:
  1. Pre-LLM (Stage 18 in the architecture doc): sanity-check whether the
     retrieved evidence pool agrees with itself on a topic, useful signal
     for multi-paper comparison questions.
  2. Post-LLM / citation verification (Stage 22): check whether evidence
     cited for a specific generated claim actually supports it.

We deliberately do NOT majority-vote blindly — a single strong CONTRADICTION
among several ENTAILMENTs is flagged as MIXED, not smoothed over, per the
architecture doc's philosophy ("the system doesn't blindly majority-vote").
"""

from dataclasses import dataclass

from backend.nlp.nli import NLIResult

# Only trust a label if the model is at least this confident; otherwise
# treat it as effectively NEUTRAL (too uncertain to count as support/contradiction).
CONFIDENCE_THRESHOLD = 0.55


@dataclass
class EvidenceVerdict:
    chunk_id: int
    label: str  # "entailment" | "contradiction" | "neutral"
    confidence: float


@dataclass
class ConsensusResult:
    verdict: str  # "SUPPORTED" | "CONTRADICTED" | "MIXED" | "INSUFFICIENT_EVIDENCE"
    support_count: int
    contradict_count: int
    neutral_count: int
    evidence_verdicts: list[EvidenceVerdict]


def compute_consensus(
    evidence_chunk_ids: list[int], nli_results: list[NLIResult]
) -> ConsensusResult:
    """
    evidence_chunk_ids and nli_results must be the same length and
    correspond positionally (i.e. nli_results[i] is the NLI verdict for
    evidence_chunk_ids[i] against whatever claim was checked).
    """
    assert len(evidence_chunk_ids) == len(nli_results), "chunk_ids and nli_results length mismatch"

    verdicts = []
    support = 0
    contradict = 0
    neutral = 0

    for chunk_id, nli in zip(evidence_chunk_ids, nli_results):
        # Low-confidence predictions are downgraded to neutral — we'd rather
        # under-claim support/contradiction than over-claim on a weak signal.
        effective_label = nli.label if nli.confidence >= CONFIDENCE_THRESHOLD else "neutral"

        verdicts.append(
            EvidenceVerdict(chunk_id=chunk_id, label=effective_label, confidence=nli.confidence)
        )

        if effective_label == "entailment":
            support += 1
        elif effective_label == "contradiction":
            contradict += 1
        else:
            neutral += 1

    if support == 0 and contradict == 0:
        overall = "INSUFFICIENT_EVIDENCE"
    elif support > 0 and contradict > 0:
        overall = "MIXED"
    elif contradict > 0:
        overall = "CONTRADICTED"
    else:
        overall = "SUPPORTED"

    return ConsensusResult(
        verdict=overall,
        support_count=support,
        contradict_count=contradict,
        neutral_count=neutral,
        evidence_verdicts=verdicts,
    )
