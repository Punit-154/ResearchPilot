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
import re

from sentence_transformers import CrossEncoder

NLI_MODEL_NAME = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")

# Standard label order for cross-encoder/nli-deberta-v3-base and most
# sentence-transformers NLI cross-encoders.
_LABELS = ["contradiction", "entailment", "neutral"]

# ─────────────────────────────────────────────────────────────────────
# KNOWN LIMITATION + WORKAROUND: general-purpose NLI models (trained on
# generic sentence pairs like SNLI/MultiNLI) cannot resolve that a paper's
# self-referential "we"/"our system" refers to the specific named system
# being asked about. Confirmed empirically: swapping "We lift locals..."
# to "ZJIT lifts locals..." against the claim "ZJIT lifts locals into SSA
# values" moved the entailment logit from -0.59 (neutral wins) to +4.02
# (entailment clearly wins) — a decisive difference, not noise.
#
# Workaround: before running NLI, replace common first-person self-
# reference patterns in the EVIDENCE (premise) with the paper's subject
# name, when we know it (e.g. the paper's title or main system name).
# This is a heuristic, not true coreference resolution — documented as a
# known limitation / future-work item (a domain-adapted or fine-tuned NLI
# model, e.g. on SciFact, would resolve this properly).
# ─────────────────────────────────────────────────────────────────────

_SELF_REFERENCE_PATTERNS = [
    (re.compile(r"\bWe\b"), None),
    (re.compile(r"\bwe\b"), None),
    (re.compile(r"\bOur\b"), None),
    (re.compile(r"\bour\b"), None),
]


def resolve_self_references(text: str, subject_name: str | None) -> str:
    """
    Replace self-referential pronouns ("We", "we", "Our", "our") with the
    paper's subject name (e.g. "ZJIT"), if provided. No-op if subject_name
    is None — callers that don't know the paper's subject can skip this.

    Also fixes subject-verb agreement: "we lift" (plural-form verb) must
    become "ZJIT lifts" (third-person singular), not the ungrammatical
    "ZJIT lift". Confirmed empirically (debug_chunk_nli.py) that feeding
    ungrammatical text measurably hurts NLI model confidence — this isn't
    just cosmetic.
    """
    if not subject_name:
        return text

    def _replace_we_plus_verb(match: re.Match) -> str:
        we_word = match.group(1)
        verb = match.group(2)
        name = subject_name if we_word[0].isupper() else subject_name
        conjugated = _conjugate_third_person_singular(verb)
        return f"{name} {conjugated}"

    # "We <verb>" / "we <verb>" -> "ZJIT <verb+s>"
    result = re.sub(r"\b(We|we)\s+(\w+)", _replace_we_plus_verb, text)
    # "Our" / "our" -> possessive form (no verb agreement issue here)
    result = re.sub(r"\bOur\b", f"{subject_name}'s", result)
    result = re.sub(r"\bour\b", f"{subject_name}'s", result)
    return result


# Common irregular verbs that don't just take a plain "+s" for third-person
# singular — covers the handful most likely to appear in this kind of prose.
_IRREGULAR_THIRD_PERSON = {
    "have": "has",
    "do": "does",
    "go": "goes",
    "use": "uses",
    "are": "is",
    "am": "is",
}

# Modal/auxiliary verbs are identical across all grammatical persons in
# English ("we can" / "it can", never "it cans") — must NOT be conjugated.
_MODAL_VERBS = {
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
}


def _conjugate_third_person_singular(verb: str) -> str:
    lower = verb.lower()
    if lower in _MODAL_VERBS:
        return verb  # unchanged — modals don't conjugate
    if lower in _IRREGULAR_THIRD_PERSON:
        conjugated = _IRREGULAR_THIRD_PERSON[lower]
    elif lower.endswith(("s", "sh", "ch", "x", "z")):
        conjugated = verb + "es"
    elif lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        conjugated = verb[:-1] + "ies"
    else:
        conjugated = verb + "s"
    return conjugated


@dataclass
class NLIResult:
    label: str  # "entailment" | "contradiction" | "neutral"
    confidence: float  # softmax probability of the winning label


@lru_cache(maxsize=1)
def get_nli_model() -> CrossEncoder:
    return CrossEncoder(NLI_MODEL_NAME)


def check_entailment(evidence_text: str, claim_text: str, subject_name: str | None = None) -> NLIResult:
    """
    Single (evidence, claim) pair -> NLIResult.
    evidence_text is the PREMISE (what we know to be true, from the paper).
    claim_text is the HYPOTHESIS (what the LLM/user asserted).

    subject_name: if provided, self-referential pronouns in evidence_text
    ("we", "our") are resolved to this name first — see module docstring
    above for why this matters and the empirical evidence behind it.
    """
    if subject_name:
        evidence_text = resolve_self_references(evidence_text, subject_name)

    model = get_nli_model()
    scores = model.predict([(evidence_text, claim_text)])[0]  # raw logits, shape (3,)

    import numpy as np
    probs = np.exp(scores) / np.sum(np.exp(scores))  # softmax
    winner_idx = int(np.argmax(probs))

    return NLIResult(label=_LABELS[winner_idx], confidence=float(probs[winner_idx]))


def check_entailment_batch(
    pairs: list[tuple[str, str]],
    subject_name: str | None = None,
    use_sentence_selection: bool = True,
    top_n_sentences: int = 2,
) -> list[NLIResult]:
    """
    Batched version — much faster than looping check_entailment() when
    verifying many (evidence, claim) pairs at once (e.g. all Top-K evidence
    against one claim, or one evidence chunk against many claims).

    subject_name: see check_entailment() docstring.
    use_sentence_selection: if True (default), each premise is first reduced
    to its most claim-relevant sentence(s) — see sentence_selection.py for
    why this matters.
    top_n_sentences: how many sentences to select per premise. Default 2
    works well for simple, single-fact claims (validated Day 9). Compound
    claims that synthesize multiple source facts (common in LLM-generated
    claims — see Day 10 citation verification) may need a higher value
    (e.g. 3) to have a chance of capturing all the relevant source sentences,
    at some risk of including a topically-similar-but-irrelevant sentence.
    """
    if not pairs:
        return []

    if use_sentence_selection:
        from backend.nlp.sentence_selection import select_relevant_sentences

        pairs = [
            (select_relevant_sentences(premise, hyp, top_n=top_n_sentences), hyp)
            for premise, hyp in pairs
        ]

    if subject_name:
        pairs = [(resolve_self_references(premise, subject_name), hyp) for premise, hyp in pairs]

    import numpy as np

    model = get_nli_model()
    all_scores = model.predict(pairs)  # shape (N, 3)

    results = []
    for scores in all_scores:
        probs = np.exp(scores) / np.sum(np.exp(scores))
        winner_idx = int(np.argmax(probs))
        results.append(NLIResult(label=_LABELS[winner_idx], confidence=float(probs[winner_idx])))
    return results
