"""
Day 9: NLI (Natural Language Inference) verification.
Given (evidence_text, claim_text), determines whether the evidence
ENTAILS, CONTRADICTS, or is NEUTRAL toward the claim.

This is the core "verification" layer of the architecture: relevance
(from retrieval) is not the same as support. A chunk can be topically
about the right subject while not actually confirming the claim.

Day 16 (Fix 3): switched from cross-encoder/nli-deberta-v3-base to
MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli — a larger model
additionally trained on FEVER (fact verification) and ANLI (adversarial
NLI), both more relevant to "does this evidence support this claim"-style
verification than plain SNLI/MultiNLI. This is a well-established, widely
cited model in NLP research specifically for this kind of task, not an
arbitrary substitution.

IMPORTANT (lesson from Day 9): we were previously burned by assuming a
model's output label order without checking. This implementation reads
label order directly from the model's own config.id2label at load time,
so it is correct regardless of what the actual order is — never hardcoded.
"""

import os
from functools import lru_cache
from dataclasses import dataclass
import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLI_MODEL_NAME = os.getenv(
    "NLI_MODEL", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)

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


@dataclass
class _LoadedNLIModel:
    tokenizer: any
    model: any
    label_map: dict[int, str]  # index -> normalized label ("entailment"/"contradiction"/"neutral")


@lru_cache(maxsize=1)
def get_nli_model() -> _LoadedNLIModel:
    """
    Loads the NLI model + tokenizer once and caches it. Label order is
    read from the model's own config.id2label rather than assumed —
    see module docstring for why this matters (Day 9 label-order bug).
    """
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    model.eval()

    raw_id2label = model.config.id2label  # e.g. {0: "entailment", 1: "neutral", 2: "contradiction"}
    label_map = {}
    for idx, raw_label in raw_id2label.items():
        normalized = raw_label.strip().lower()
        # Defensive normalization in case the model card uses slightly
        # different casing/spelling than our internal 3 labels.
        if "entail" in normalized:
            label_map[idx] = "entailment"
        elif "contradict" in normalized:
            label_map[idx] = "contradiction"
        else:
            label_map[idx] = "neutral"

    print(f"[nli] Loaded {NLI_MODEL_NAME} — label map: {label_map}")
    return _LoadedNLIModel(tokenizer=tokenizer, model=model, label_map=label_map)


def _predict_batch(pairs: list[tuple[str, str]]) -> list[NLIResult]:
    """Runs the loaded model on a batch of (premise, hypothesis) pairs."""
    loaded = get_nli_model()

    inputs = loaded.tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = loaded.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    results = []
    for row in probs:
        winner_idx = int(torch.argmax(row).item())
        confidence = float(row[winner_idx].item())
        label = loaded.label_map[winner_idx]
        results.append(NLIResult(label=label, confidence=confidence))
    return results


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

    return _predict_batch([(evidence_text, claim_text)])[0]


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

    return _predict_batch(pairs)
