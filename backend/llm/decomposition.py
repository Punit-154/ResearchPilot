"""
Post-generation claim decomposition.

Inserts between synthesize_answer() and verify_claims() when
use_claim_decomposition=True is passed to the /answer route.

A single batched LLM call asks the model to split any compound claim
(one sentence containing 2+ facts) into atomic sub-facts, each inheriting
the parent's citation labels unless the model can confidently narrow them.
Atomic claims are returned unchanged.

Design decisions documented in project report (Day 16 / Feature 4):
- One batched call per answer (not one call per claim) — saves ~10x tokens
  and latency at the cost of a slightly more complex prompt.
- temperature=0.0 — decomposition is deterministic parsing, not creative.
- parent_claim_id: sub-claims carry the original claim id so the frontend
  can group them and aggregate their verification statuses.
- Aggregation rule (in citation_verification.py and the frontend):
  parent is VALID iff ALL sub-facts are VALID (including NO_CITATIONS
  treated as FLAGGED). See project report for justification.
"""

import json
import re
from dataclasses import dataclass

from backend.llm.synthesis import Claim
from backend.llm.grok_client import call_grok


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

DECOMPOSITION_SYSTEM_PROMPT = """\
You are a claim decomposition assistant. You will receive a list of factual claims, each with an ID and citation labels. Your job is to break any compound claim into atomic sub-facts — where "atomic" means the claim states exactly ONE verifiable fact.

RULES:
1. If a claim is already atomic (states exactly one fact), return it unchanged as a single item in its output list.
2. If a claim is compound (states 2+ facts, often joined by "and", "while", "instead of", "which means", "whereas", "unlike", etc.), split it into separate atomic sub-claims.
3. Each sub-claim inherits ALL citation labels from its parent unless you can confidently assign a more specific subset. When in doubt, keep all parent citations on each sub-claim — do not remove a citation speculatively.
4. Do NOT invent new facts. Every sub-claim must be directly derivable from the original claim text — do not add interpretation or outside knowledge.
5. Do NOT split a claim that is genuinely atomic just to produce more output. One fact = one claim.

Respond with ONLY valid JSON, no markdown fences. Schema:

{
  "decomposed": [
    {
      "parent_id": "C01",
      "sub_claims": [
        {"text": "<atomic sub-fact>", "citations": ["E01", "E02"]},
        {"text": "<another atomic sub-fact>", "citations": ["E01"]}
      ]
    }
  ]
}

If a claim is already atomic, sub_claims will contain exactly one item (the claim itself, unmodified).
Output ONLY the JSON object above. Nothing else.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_SUFFIX_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _extract_json(raw_text: str) -> dict:
    """
    Strip optional ```json … ``` fences before parsing — LLMs sometimes add
    them despite instructions not to (same defensive pattern as synthesis.py).
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _build_user_prompt(claims: list[Claim]) -> str:
    """
    Serialise the claim list into the format the decomposition prompt expects.
    """
    items = [
        {"id": c.id, "text": c.text, "citations": c.citation_labels}
        for c in claims
    ]
    return json.dumps({"claims": items}, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def decompose_claims(claims: list[Claim]) -> list[Claim]:
    """
    Decompose a list of claims into atomic sub-facts using a single batched
    LLM call.  Returns a flat list of Claim objects:

    - Claims the LLM judges already atomic are returned unchanged
      (same id, same text, same citations, parent_claim_id=None).
    - Compound claims are replaced by N sub-claims with:
        id             = "<parent_id>_a", "<parent_id>_b", ...
        text           = the atomic sub-fact text
        citation_labels = as returned by the LLM (subset or full parent set)
        parent_claim_id = the original parent claim id

    Robustness:
    - If the LLM response omits a claim entirely (e.g. truncation), that
      claim falls back to being returned unchanged rather than being dropped.
    - If the LLM returns an unrecognised parent_id, that entry is silently
      skipped (does not crash; we only trust ids that exist in our input).
    """
    if not claims:
        return []

    # Map original claims by id for O(1) lookup and fallback
    original_by_id: dict[str, Claim] = {c.id: c for c in claims}

    user_prompt = _build_user_prompt(claims)

    raw_response = call_grok(
        DECOMPOSITION_SYSTEM_PROMPT,
        user_prompt,
        max_tokens=1000,
        temperature=0.0,
    )

    try:
        parsed = _extract_json(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Decomposition LLM did not return valid JSON. Raw response was:\n{raw_response}"
        ) from e

    # Build a map: parent_id -> list of sub_claim dicts from the LLM response
    llm_map: dict[str, list[dict]] = {}
    for entry in parsed.get("decomposed", []):
        parent_id = entry.get("parent_id")
        # Silently skip hallucinated parent ids not in our input
        if parent_id not in original_by_id:
            continue
        llm_map[parent_id] = entry.get("sub_claims", [])

    # Reconstruct the flat output list, preserving original claim order
    result: list[Claim] = []
    for original in claims:
        sub_claims_raw = llm_map.get(original.id)

        if sub_claims_raw is None:
            # LLM omitted this claim entirely — fall back to original unchanged
            result.append(original)
            continue

        if len(sub_claims_raw) <= 1:
            # Atomic claim (or degenerate 0-item list) — return original unchanged
            # so id and parent_claim_id stay exactly as they were
            result.append(original)
            continue

        # Compound claim → produce one new Claim per sub-fact
        for i, sub in enumerate(sub_claims_raw):
            suffix = _SUFFIX_LETTERS[i] if i < len(_SUFFIX_LETTERS) else str(i)
            new_claim = Claim(
                id=f"{original.id}_{suffix}",
                text=sub.get("text", ""),
                citation_labels=sub.get("citations", original.citation_labels),
            )
            new_claim.parent_claim_id = original.id
            result.append(new_claim)

    return result
