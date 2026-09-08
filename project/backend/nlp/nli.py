"""
Day 9: NLI (Natural Language Inference) verification.
Given (evidence_text, claim_text), determines whether the evidence
ENTAILS, CONTRADICTS, or is NEUTRAL toward the claim.

This is the core "verification" layer of the architecture: relevance
(from retrieval) is not the same as support. A chunk can be topically
about the right subject while not actually confirming the claim.

Model: cross-encoder/nli-deberta-v3-base via sentence-transformers'
CrossEncoder — takes (premise, hypothesis) pairs directly, outputs
3-class logits [contradiction, entailment, neutral] (this is the standard
label order for this model family, per its model card).
"""

import os
from functools import lru_cache
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

NLI_MODEL_NAME = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")

# Standard label order for cross-encoder/nli-deberta-v3-base and most
# sentence-transformers NLI cross-encoders.
_LABELS = ["contradiction", "entailment", "neutral"]


@dataclass
class NLIResult:
    label: str  # "entailment" | "contradiction" | "neutral"
    confidence: float  # softmax probability of the winning label


@lru_cache(maxsize=1)
def get_nli_model() -> CrossEncoder:
    return CrossEncoder(NLI_MODEL_NAME)


def check_entailment(evidence_text: str, claim_text: str) -> NLIResult:
    """
    Single (evidence, claim) pair -> NLIResult.
    evidence_text is the PREMISE (what we know to be true, from the paper).
    claim_text is the HYPOTHESIS (what the LLM/user asserted).
    """
    model = get_nli_model()
    scores = model.predict([(evidence_text, claim_text)])[0]  # raw logits, shape (3,)

    import numpy as np
    probs = np.exp(scores) / np.sum(np.exp(scores))  # softmax
    winner_idx = int(np.argmax(probs))

    return NLIResult(label=_LABELS[winner_idx], confidence=float(probs[winner_idx]))


def check_entailment_batch(pairs: list[tuple[str, str]]) -> list[NLIResult]:
    """
    Batched version — much faster than looping check_entailment() when
    verifying many (evidence, claim) pairs at once (e.g. all Top-K evidence
    against one claim, or one evidence chunk against many claims).
    """
    if not pairs:
        return []

    import numpy as np

    model = get_nli_model()
    all_scores = model.predict(pairs)  # shape (N, 3)

    results = []
    for scores in all_scores:
        probs = np.exp(scores) / np.sum(np.exp(scores))
        winner_idx = int(np.argmax(probs))
        results.append(NLIResult(label=_LABELS[winner_idx], confidence=float(probs[winner_idx])))
    return results
