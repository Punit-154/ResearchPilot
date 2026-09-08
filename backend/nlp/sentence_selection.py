"""
Day 9 fix: sentence-level evidence selection for NLI.

ROOT CAUSE (confirmed empirically via diagnose_nli.py): NLI models are
trained on short, single-sentence premise/hypothesis pairs. Feeding an
entire ~180-word chunk as the premise dilutes the signal so badly that
even a chunk containing an exact supporting sentence gets scored as
"neutral" by the model (winning chunk 136: 99.8%+ neutral confidence,
despite literally containing "we lift locals to SSA values").

FIX: split each evidence chunk into sentences, embed them (reusing the
same embedding model from Day 7), and pick the sentence(s) most similar
to the claim as the actual NLI premise — instead of the whole chunk.
This mirrors how real fact-verification systems (e.g. SciFact-style
pipelines) work: verify against RATIONALE SENTENCES, not full documents.
"""

import re
import numpy as np

from backend.nlp.embeddings import embed_text, embed_texts

# Simple sentence splitter — not perfect (won't handle "Dr. Smith" correctly,
# for example) but adequate for research-paper prose at MVP scope. A proper
# sentence tokenizer (e.g. nltk/spacy) would be a natural upgrade later.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text.strip())
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s) <= 15:
            continue
        # A sentence starting with a lowercase letter is almost always a
        # fragment left over from a chunk boundary cutting a sentence in
        # half (the real sentence start was in the PREVIOUS chunk). These
        # fragments read as confusing, out-of-context negations/clauses to
        # the NLI model (confirmed empirically via debug_chunk_nli.py) —
        # exclude them from sentence-selection candidates entirely.
        if s[0].islower():
            continue
        cleaned.append(s)
    return cleaned


def select_relevant_sentences(
    chunk_text: str, claim: str, top_n: int = 2, return_debug: bool = False
):
    """
    Returns the top_n sentences from chunk_text most semantically similar
    to `claim`, joined into a short premise string. Falls back to the
    original chunk_text if splitting produces too few sentences to bother
    (e.g. a chunk that's already short, or a chunk with unusual formatting
    that doesn't split cleanly).

    If return_debug=True, returns (selected_text, debug_info) where
    debug_info lists every sentence considered and its similarity score —
    useful for diagnosing why a particular sentence was/wasn't picked.
    """
    sentences = split_sentences(chunk_text)

    if len(sentences) <= top_n:
        if return_debug:
            return chunk_text, {"note": "too few sentences to select from", "sentences": sentences}
        return chunk_text

    claim_vec = np.array(embed_text(claim))
    sentence_vecs = np.array(embed_texts(sentences))

    similarities = sentence_vecs @ claim_vec

    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_indices_sorted = sorted(top_indices)

    selected = [sentences[i] for i in top_indices_sorted]
    result = " ".join(selected)

    if return_debug:
        debug_info = {
            "all_sentences_ranked": [
                {"sentence": sentences[i], "similarity": float(similarities[i])}
                for i in np.argsort(similarities)[::-1]
            ],
            "selected": selected,
        }
        return result, debug_info

    return result
