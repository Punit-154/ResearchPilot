"""
Day 13: Automatic subject-name extraction.
Previously, subject_name (e.g. "ZJIT") had to be manually typed by the user
on every /answer or /verify/claim call to work around the NLI coreference
limitation found on Day 9 ("we" -> paper's system name). This module
extracts it ONCE at ingestion time and stores it on the Paper row, so
queries never need it passed manually again.

This is a best-effort heuristic step: if extraction fails (rate limit,
LLM error, or the paper doesn't clearly name a single system), ingestion
still succeeds — subject_name is simply left NULL, and NLI falls back to
its non-substituted behavior (still functional, just with the known
coreference limitation from Day 9). Failures ARE logged (not silent) so
you can tell rate-limiting from "paper has no clear subject" at a glance.
"""

import logging

from backend.llm.grok_client import call_grok, GrokAPIError

logger = logging.getLogger(__name__)

SUBJECT_EXTRACTION_PROMPT = """You will be given a research paper's title and the beginning of its abstract. Identify the name of the NEW system, tool, or technique that THIS PAPER ITSELF introduces or presents as its own contribution.

IMPORTANT: Abstracts often mention OTHER existing systems first for background or comparison (e.g. "Unlike EXISTING_SYSTEM, we present NEW_SYSTEM..."). Do NOT pick a background/comparison system — pick ONLY the paper's own new contribution, typically introduced with phrasing like "we present", "we introduce", "this paper presents", "we propose".

Respond with ONLY that name — no explanation, no punctuation, no extra words. If there is genuinely no single clear new system introduced (e.g. a general survey paper), respond with exactly: NONE

Example:
Abstract: "CRuby has an existing JIT compiler called YJIT. In order to support more optimizations, we present a new compiler called ZJIT."
Correct answer: ZJIT
(NOT "YJIT" — that's the pre-existing background system, not this paper's contribution)"""


def extract_subject_name(title: str, abstract_snippet: str) -> str | None:
    """
    Returns a short subject/system name (e.g. "ZJIT"), or None if
    extraction failed or the paper has no single clear subject.
    """
    user_prompt = f"TITLE: {title}\n\nABSTRACT: {abstract_snippet[:1500]}"

    try:
        raw = call_grok(SUBJECT_EXTRACTION_PROMPT, user_prompt, max_tokens=200, temperature=0.0)
    except GrokAPIError as e:
        logger.warning(f"subject_name extraction FAILED for '{title}': {e}")
        print(f"[subject_name extraction] FAILED for '{title}': {e}")
        return None  # non-fatal — ingestion continues without a subject_name

    print(f"[subject_name extraction] raw response for '{title}': {raw!r}")
    cleaned = raw.strip().strip('"').strip("'")

    if not cleaned or cleaned.upper() == "NONE":
        print(f"[subject_name extraction] No clear subject found for '{title}' (LLM said: {raw!r})")
        return None

    # Defensive sanity check: a real subject name should be short (a few
    # words at most) — if the LLM ignored instructions and returned a full
    # sentence, discard it rather than storing garbage.
    if len(cleaned.split()) > 6 or len(cleaned) > 60:
        print(f"[subject_name extraction] Rejected suspiciously long response for '{title}': {cleaned!r}")
        return None

    print(f"[subject_name extraction] SUCCESS for '{title}': '{cleaned}'")
    return cleaned
