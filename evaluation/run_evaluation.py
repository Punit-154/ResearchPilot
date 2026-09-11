"""
Day 12: Evaluation framework.
Runs a shared set of test questions through BOTH the full architecture
(/answer) and the simple baseline (/baseline/answer), then compares them
on measurable dimensions.

HONEST SCOPE NOTE: this is a lightweight, keyword-based proxy evaluation,
not a rigorous human-annotated benchmark. With more time, "did the answer
mention the right facts" would be replaced by expert human judgment or a
reference-answer overlap metric (e.g. ROUGE against a gold answer). What
this script DOES give you, honestly:
  1. Keyword recall — a crude but real proxy for "did the answer cover the
     facts a knowledgeable reader would expect"
  2. Latency — full architecture has more stages, so is expected to be
     slower; worth knowing exactly how much slower
  3. Citation health — ONLY the full architecture can report this at all.
     The baseline has no verification step, so it structurally cannot self-
     report whether its claims are grounded. That asymmetry IS a finding.

Run with:  python evaluation/run_evaluation.py
(requires the API server to be running at localhost:8000)
"""

import time
import json
import requests

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────
# Test set: edit PAPER_ID to match whichever paper you've ingested, and
# adjust SUBJECT_NAME to that paper's system/subject name (see Day 9).
# expected_keywords are phrases a correct, complete answer should contain
# — used only for the crude keyword-recall proxy metric described above.
# ─────────────────────────────────────────────────────────────────────
PAPER_ID = 3
SUBJECT_NAME = "ZJIT"

TEST_QUESTIONS = [
    {
        "question": "How does ZJIT handle local variables?",
        "expected_keywords": ["SSA", "local variable", "memory"],
    },
    {
        "question": "What happens when the environment escapes?",
        "expected_keywords": ["Proc", "heap", "escape"],
    },
    {
        "question": "How does ZJIT differ from YJIT?",
        "expected_keywords": ["basic block", "method", "whole method"],
    },
    {
        "question": "What is a PatchPoint used for?",
        "expected_keywords": ["redefin", "side-exit", "side exit"],
    },
    {
        "question": "How does ZJIT's performance compare to other Ruby implementations?",
        "expected_keywords": ["TruffleRuby", "JRuby", "benchmark"],
    },
]


def keyword_recall(answer_text: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found (case-insensitive substring match)."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer_text.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def _request_with_retry(url: str, params: dict, max_retries: int = 3) -> dict:
    """
    Wraps requests.get with retry + backoff — Groq's free tier has fairly
    tight rate limits, and firing many LLM calls back-to-back (as this
    evaluation script does) can trigger 429s that surface as 502s from our
    own API. Retrying with a short wait usually clears this.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            last_error = e
            error_body = e.response.text if e.response is not None else "(no response body)"
            print(f"    attempt {attempt}/{max_retries} failed: {e}")
            print(f"    server said: {error_body[:300]}")
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"    waiting {wait}s before retry...")
                time.sleep(wait)
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"    attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(5)
    raise last_error


def call_full_architecture(question: str) -> dict:
    start = time.time()
    data = _request_with_retry(
        f"{API_BASE}/answer",
        params={"q": question, "paper_id": PAPER_ID, "subject_name": SUBJECT_NAME},
    )
    elapsed = time.time() - start
    return {
        "answer": data["answer"],
        "latency_seconds": round(elapsed, 2),
        "citation_health": data["citation_health"],
        "num_claims": len(data["claims"]),
    }


def call_baseline(question: str) -> dict:
    start = time.time()
    data = _request_with_retry(
        f"{API_BASE}/baseline/answer",
        params={"q": question, "paper_id": PAPER_ID},
    )
    elapsed = time.time() - start
    return {
        "answer": data["answer"],
        "latency_seconds": round(elapsed, 2),
    }


def run_evaluation():
    results = []

    for item in TEST_QUESTIONS:
        question = item["question"]
        expected_keywords = item["expected_keywords"]

        print(f"\n{'=' * 70}")
        print(f"Q: {question}")
        print("=" * 70)

        print("  Running full architecture...")
        try:
            full_result = call_full_architecture(question)
            full_recall = keyword_recall(full_result["answer"], expected_keywords)
        except Exception as e:
            print(f"  ERROR (full architecture): {e}")
            full_result = None
            full_recall = None

        print("  Running baseline...")
        try:
            baseline_result = call_baseline(question)
            baseline_recall = keyword_recall(baseline_result["answer"], expected_keywords)
        except Exception as e:
            print(f"  ERROR (baseline): {e}")
            baseline_result = None
            baseline_recall = None

        results.append(
            {
                "question": question,
                "expected_keywords": expected_keywords,
                "full_architecture": full_result,
                "full_architecture_keyword_recall": full_recall,
                "baseline": baseline_result,
                "baseline_keyword_recall": baseline_recall,
            }
        )

        if full_result:
            print(f"  Full arch: {full_result['latency_seconds']}s, "
                  f"keyword recall={full_recall:.0%}, "
                  f"citations valid={full_result['citation_health']['valid_claims']}/"
                  f"{full_result['citation_health']['total_claims']}")
        if baseline_result:
            print(f"  Baseline:  {baseline_result['latency_seconds']}s, "
                  f"keyword recall={baseline_recall:.0%}, "
                  f"citations: N/A (no verification capability)")

        # Pace requests to avoid tripping Groq's rate limits — each question
        # already involves 2+ LLM calls (baseline + full architecture, which
        # itself calls the LLM once plus several NLI calls); adding a short
        # gap between QUESTIONS (not within them) keeps us well under typical
        # free-tier rate limits.
        time.sleep(3)

    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    valid_full = [r for r in results if r["full_architecture"] is not None]
    valid_baseline = [r for r in results if r["baseline"] is not None]

    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    if valid_full:
        avg_full_latency = sum(r["full_architecture"]["latency_seconds"] for r in valid_full) / len(valid_full)
        avg_full_recall = sum(r["full_architecture_keyword_recall"] for r in valid_full) / len(valid_full)
        total_claims = sum(r["full_architecture"]["citation_health"]["total_claims"] for r in valid_full)
        total_valid_claims = sum(r["full_architecture"]["citation_health"]["valid_claims"] for r in valid_full)
        print(f"Full architecture:")
        print(f"  Avg latency: {avg_full_latency:.2f}s")
        print(f"  Avg keyword recall: {avg_full_recall:.0%}")
        print(f"  Citation validity rate: {total_valid_claims}/{total_claims} "
              f"({total_valid_claims/total_claims:.0%})" if total_claims else "  Citation validity rate: N/A")

    if valid_baseline:
        avg_baseline_latency = sum(r["baseline"]["latency_seconds"] for r in valid_baseline) / len(valid_baseline)
        avg_baseline_recall = sum(r["baseline_keyword_recall"] for r in valid_baseline) / len(valid_baseline)
        print(f"\nBaseline:")
        print(f"  Avg latency: {avg_baseline_latency:.2f}s")
        print(f"  Avg keyword recall: {avg_baseline_recall:.0%}")
        print(f"  Citation validity rate: N/A (baseline has no verification step)")

    # Save full results to disk for the presentation
    with open("evaluation/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to evaluation/results.json")


if __name__ == "__main__":
    run_evaluation()
