"""
Day 5: Chunking.
Splits each detected section's text into overlapping word-window chunks —
the actual retrieval unit used by lexical/semantic search later.

Design choice: chunk PER PAGE within a section (not the whole section as one
blob) so each chunk keeps an accurate `page` number. This matters for
citations later ("[E02] Paper B, Results, p.10") — losing page accuracy
would make evidence citations useless.

Chunk size is in words, not characters/tokens, for simplicity at MVP stage.
Overlap keeps context from being cut off mid-thought at chunk boundaries.
"""

from dataclasses import dataclass

CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40


@dataclass
class ChunkCandidate:
    section_id: int | None
    section_name: str
    page: int
    chunk_index: int
    text: str


def _split_words_with_overlap(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if not words:
        return []
    if len(words) <= size:
        return [words]

    step = max(size - overlap, 1)
    windows = []
    start = 0
    while start < len(words):
        window = words[start : start + size]
        if window:
            windows.append(window)
        if start + size >= len(words):
            break
        start += step
    return windows


def chunk_paper(
    pages: list, sections: list, section_id_by_order: dict[int, int]
) -> list[ChunkCandidate]:
    """
    pages: list of PageText (from pdf_parser.parse_pdf) — has .page_number, .text
    sections: list of DetectedSection (from structure.detect_sections) — has
              .name, .order, .start_page, .end_page
    section_id_by_order: maps DetectedSection.order -> the DB Section.id it was
                          saved as (caller looks this up after inserting sections)

    Returns a flat, ordered list of ChunkCandidate ready to persist as Chunk rows.
    """
    page_text_by_num = {p.page_number: p.text for p in pages}

    chunks: list[ChunkCandidate] = []
    global_index = 0

    for section in sorted(sections, key=lambda s: s.order):
        section_id = section_id_by_order.get(section.order)

        for page_num in range(section.start_page, section.end_page + 1):
            text = page_text_by_num.get(page_num, "")
            words = text.split()
            if not words:
                continue

            for window in _split_words_with_overlap(words, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS):
                chunk_text = " ".join(window).strip()
                if len(chunk_text) < 20:
                    continue  # skip near-empty fragments (e.g. page footers only)

                chunks.append(
                    ChunkCandidate(
                        section_id=section_id,
                        section_name=section.name,
                        page=page_num,
                        chunk_index=global_index,
                        text=chunk_text,
                    )
                )
                global_index += 1

    return chunks
