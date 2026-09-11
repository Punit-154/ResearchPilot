"""
Day 10-11: Evidence pack + structured LLM synthesis.

Builds a controlled context (Stage 19: Evidence Pack) from retrieved
evidence, sends it to Grok with strict instructions to answer ONLY from
that evidence and cite it by label (Stage 20-21), and parses the
structured JSON response back into Python objects.

The evidence pack is the ONLY scientific context given to the LLM — no
pretrained-knowledge answers, per the architecture doc's explicit boundary
("Not: Determine facts from its pretrained knowledge").
"""

import json
import re
from dataclasses import dataclass, field

from backend.retrieval.hybrid import HybridResult
from backend.llm.grok_client import call_grok, GrokAPIError


@dataclass
class EvidenceItem:
    label: str  # "E01", "E02", ...
    chunk_id: int
    paper_id: int
    paper_title: str
    page: int
    text: str


@dataclass
class Claim:
    id: str
    text: str
    citation_labels: list[str]  # e.g. ["E01", "E02"]


@dataclass
class SynthesisResult:
    answer: str
    claims: list[Claim]
    consensus: str
    uncertainties: list[str]
    evidence_by_label: dict[str, EvidenceItem]


def build_evidence_pack(
    evidence: list[HybridResult], paper_titles: dict[int, str] | None = None
) -> tuple[str, dict[str, EvidenceItem]]:
    """
    Converts retrieved evidence into labeled text blocks ([E01], [E02], ...)
    for the LLM prompt, and returns a label -> EvidenceItem map so we can
    resolve citations back to real chunk data afterward.

    paper_titles: optional {paper_id: title} map. When retrieving across
    MULTIPLE papers (comparison queries), each evidence block is labeled
    with which paper it came from, so both the LLM and a human reader can
    tell the sources apart — essential for "compare Paper A and Paper B"
    style questions. When omitted or single-paper, this label is skipped
    (no point labeling every chunk with the same single paper name).
    """
    paper_titles = paper_titles or {}
    evidence_by_label = {}
    blocks = []

    for i, e in enumerate(evidence, start=1):
        label = f"E{i:02d}"
        title = paper_titles.get(e.paper_id, f"Paper {e.paper_id}")
        evidence_by_label[label] = EvidenceItem(
            label=label, chunk_id=e.chunk_id, paper_id=e.paper_id, paper_title=title, page=e.page, text=e.text
        )

        # Only show the paper name in the prompt when there's more than one
        # distinct paper in this evidence set — keeps single-paper prompts
        # exactly as clean as before.
        distinct_papers = {ev.paper_id for ev in evidence}
        if len(distinct_papers) > 1:
            blocks.append(f"[{label}] (Paper: \"{title}\", page {e.page})\n{e.text}")
        else:
            blocks.append(f"[{label}] (page {e.page})\n{e.text}")

    pack_text = "\n\n".join(blocks)
    return pack_text, evidence_by_label


SYSTEM_PROMPT = """You are a scientific evidence synthesis assistant. You will be given a QUESTION and a set of EVIDENCE passages, each labeled (e.g. [E01], [E02]). When evidence comes from more than one paper, each block is also tagged with which paper it's from.

STRICT RULES:
1. Answer using ONLY the provided evidence. Do NOT use any outside knowledge about the topic.
2. Every factual claim you make MUST cite the evidence label(s) that support it.
3. ATOMIC CLAIMS ONLY: each claim in "claims" must state exactly ONE fact, not several facts joined together. If you find yourself writing "X, and also Y" or "X, which means Y" or "X instead of Y" as one claim, split it into two separate claims instead — one for X, one for Y — each with its own citation(s). This makes claims easier to verify precisely.
   BAD (compound, do not do this): "ZJIT lifts locals into SSA values, unlike other compilers which use memory loads and stores."
   GOOD (split into atomic claims):
     - claim: "ZJIT lifts local variables into SSA values." citations: [...]
     - claim: "Other Ruby JIT compilers keep local variables as memory loads and stores instead." citations: [...]
4. If the question asks you to COMPARE multiple papers, address each paper's position explicitly and note where they agree or disagree — don't just describe one and ignore the other. Still keep each individual claim atomic per rule 3.
5. If the evidence does not fully answer the question, say so explicitly in "uncertainties" rather than guessing.
6. If evidence conflicts (within one paper or across papers), reflect that honestly rather than picking one side silently.
7. Respond with ONLY valid JSON, no markdown code fences, no explanation before or after. The JSON must match this exact schema:

{
  "answer": "<a readable paragraph answering the question>",
  "claims": [
    {"id": "C01", "text": "<a single, atomic, specific factual claim>", "citations": ["E01", "E02"]}
  ],
  "consensus": "<one of: SUPPORTED, MIXED, CONTRADICTED, INSUFFICIENT_EVIDENCE>",
  "uncertainties": ["<any caveats, gaps, or things the evidence doesn't cover>"]
}

Output ONLY the JSON object above. Nothing else."""


def _extract_json(raw_text: str) -> dict:
    """
    LLMs sometimes wrap JSON in ```json fences despite instructions not to.
    Strip those defensively before parsing.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def synthesize_answer(
    query: str, evidence: list[HybridResult], paper_titles: dict[int, str] | None = None
) -> SynthesisResult:
    """
    Full Stage 19-21 pipeline: build evidence pack -> call Grok -> parse
    structured output. Raises GrokAPIError or ValueError (bad JSON) on
    failure — caller should handle and surface a clean error to the user.

    paper_titles: see build_evidence_pack() — needed for multi-paper
    comparison queries so the LLM (and evidence labels) can distinguish
    which paper each piece of evidence came from.
    """
    if not evidence:
        return SynthesisResult(
            answer="No relevant evidence was found for this question.",
            claims=[],
            consensus="INSUFFICIENT_EVIDENCE",
            uncertainties=["No evidence retrieved from the paper(s) for this query."],
            evidence_by_label={},
        )

    pack_text, evidence_by_label = build_evidence_pack(evidence, paper_titles=paper_titles)

    user_prompt = f"QUESTION: {query}\n\nEVIDENCE:\n\n{pack_text}"

    # Day 15: max_tokens raised from the default (1500) to 2500 — atomic-
    # claim prompting (Fix 2) intentionally produces MORE, shorter claims
    # instead of fewer, longer compound ones, which increases total JSON
    # output length and risks truncated/invalid JSON at the old limit
    # (observed directly: a real request failed JSON parsing after this
    # change increased claim count, before this max_tokens fix was applied).
    # Day 15: temperature=0.0 (down from default 0.2) — lower temperature
    # makes claim decomposition more deterministic run-to-run. Observed
    # directly: citation validity varied 74-82% across identical repeated
    # runs at temperature=0.2, since the LLM phrased/split claims slightly
    # differently each time, changing what the NLI verifier saw. This
    # doesn't change WHAT the system can verify, only how consistently it
    # reports the same result for the same question.
    raw_response = call_grok(SYSTEM_PROMPT, user_prompt, max_tokens=2500, temperature=0.0)

    try:
        parsed = _extract_json(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Grok did not return valid JSON. Raw response was:\n{raw_response}"
        ) from e

    claims = [
        Claim(id=c.get("id", f"C{i:02d}"), text=c["text"], citation_labels=c.get("citations", []))
        for i, c in enumerate(parsed.get("claims", []), start=1)
    ]

    return SynthesisResult(
        answer=parsed.get("answer", ""),
        claims=claims,
        consensus=parsed.get("consensus", "INSUFFICIENT_EVIDENCE"),
        uncertainties=parsed.get("uncertainties", []),
        evidence_by_label=evidence_by_label,
    )
