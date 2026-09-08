"""
Day 11 (Stage 26): Baseline system.
A deliberately simple RAG pipeline to compare against the full architecture:

    Question -> Embedding (semantic) retrieval -> Top-K -> LLM -> Answer

No hybrid retrieval, no query planning, no NLI verification, no consensus
analysis, no citation verification. This is what most naive "chat with your
PDF" tools do. The point of building this is to let the evaluation stage
(Day 12) measure whether the full architecture's added complexity actually
buys measurable improvement — groundedness, citation accuracy, correctness —
over this much simpler and cheaper approach.
"""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from backend.retrieval.semantic import semantic_search, SemanticResult
from backend.llm.grok_client import call_grok, GrokAPIError

BASELINE_SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question using the provided context passages from a research paper. Write a clear, direct answer. If the context doesn't contain enough information, say so."""


@dataclass
class BaselineResult:
    answer: str
    retrieved_chunk_ids: list[int]
    retrieved_pages: list[int]


def run_baseline(
    query: str,
    db: Session,
    top_k: int = 8,
    paper_ids: list[int] | None = None,
) -> BaselineResult:
    """
    The simple baseline pipeline. No labeled evidence pack, no citation
    enforcement, no verification — just "here's some context, answer the
    question" the way most basic RAG tutorials implement it.
    """
    results: list[SemanticResult] = semantic_search(query, db, limit=top_k, paper_ids=paper_ids)

    if not results:
        return BaselineResult(
            answer="No relevant information was found in the paper(s) for this question.",
            retrieved_chunk_ids=[],
            retrieved_pages=[],
        )

    context = "\n\n".join(f"(page {r.page}) {r.text}" for r in results)
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

    try:
        answer = call_grok(BASELINE_SYSTEM_PROMPT, user_prompt, temperature=0.2)
    except GrokAPIError as e:
        raise

    return BaselineResult(
        answer=answer.strip(),
        retrieved_chunk_ids=[r.chunk_id for r in results],
        retrieved_pages=[r.page for r in results],
    )
