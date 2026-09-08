"""
Day 3: PDF structure extraction.
Detects section boundaries (Title, Abstract, Introduction, Methodology,
Results, Discussion, Conclusion, References, etc.) so later stages can
weight retrieval by section relevance.

Heuristic (two signals combined):
  1. FONT SIZE — headings are usually larger/bolder than body text.
     We compute the dominant (most common) font size on each page = "body size",
     and treat any line noticeably larger than that as a heading candidate.
  2. KNOWN SECTION NAMES — a line matching a common research-paper section
     name (case-insensitive, ignoring numbering like "3." or "III.") is
     treated as a heading even if font-size detection is ambiguous
     (common in single-column PDFs where headings aren't much bigger).

This is intentionally a heuristic, not a perfect parser — two-column layouts
and non-standard templates will be less accurate. Good enough for MVP-level
section-aware retrieval, and can be swapped for a stronger tool later
without touching anything downstream (chunking just reads `sections`).
"""

import re
from dataclasses import dataclass

import fitz  # PyMuPDF

# Canonical section names we try to recognize, in typical paper order.
# Matching is case-insensitive and tolerant of numbering prefixes like "3." / "III."
KNOWN_SECTION_NAMES = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "methodology",
    "methods",
    "materials and methods",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "results and discussion",
    "discussion",
    "evaluation",
    "analysis",
    "conclusion",
    "conclusions",
    "future work",
    "limitations",
    "acknowledgments",
    "acknowledgements",
    "references",
    "appendix",
]

# Matches optional numbering ("3.", "III.", "3.1") then one of the known names
_NUMBERING = r"^\s*([0-9]+[\.\)]?|[IVXLC]+[\.\)]?)?\s*"
_SECTION_REGEX = re.compile(
    _NUMBERING + r"(" + "|".join(re.escape(n) for n in KNOWN_SECTION_NAMES) + r")\s*$",
    re.IGNORECASE,
)

# Lines that are just a number/numeral (leftover section numbering split
# onto its own line, e.g. "3" before "Architecture of ZJIT") — never useful alone.
_PURE_NUMBERING = re.compile(r"^\s*([0-9]+[\.\)]?|[IVXLC]+[\.\)]?)\s*$")

# PyMuPDF span flag bit for bold text (flags & 2**4)
_BOLD_FLAG = 1 << 4


def _line_is_bold(line: dict) -> bool:
    spans = line.get("spans", [])
    if not spans:
        return False
    return any((s.get("flags", 0) & _BOLD_FLAG) for s in spans)


@dataclass
class DetectedSection:
    name: str
    order: int
    start_page: int
    end_page: int


def _body_font_size(page_dict: dict) -> float:
    """Return the most common font size on a page — treated as 'body text' size."""
    sizes = {}
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = round(span.get("size", 0), 1)
                text = span.get("text", "").strip()
                if not text:
                    continue
                sizes[size] = sizes.get(size, 0) + len(text)
    if not sizes:
        return 0.0
    return max(sizes, key=sizes.get)


def _line_max_font_size(line: dict) -> float:
    spans = line.get("spans", [])
    if not spans:
        return 0.0
    return max(round(s.get("size", 0), 1) for s in spans)


def _line_text(line: dict) -> str:
    return "".join(s.get("text", "") for s in line.get("spans", [])).strip()


def detect_sections(filepath: str, num_pages: int) -> list[DetectedSection]:
    """
    Scan the PDF and return an ordered list of DetectedSection, each with the
    page range it spans. If nothing is detected at all, returns a single
    'Full Text' section spanning the whole document (safe fallback so
    chunking/retrieval never has zero sections to work with).
    """
    doc = fitz.open(filepath)

    # (page_number, heading_text, canonical_name_or_None)
    candidates: list[tuple[int, str]] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        page_dict = page.get_text("dict")
        body_size = _body_font_size(page_dict)

        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                text = _line_text(line)
                if not text or len(text) > 80:
                    continue  # headings are short; skip long body lines fast

                if _PURE_NUMBERING.match(text):
                    continue  # e.g. a lone "3" before "Architecture of ZJIT" — not useful alone

                is_known_name = bool(_SECTION_REGEX.match(text))
                max_size = _line_max_font_size(line)
                is_larger_font = body_size > 0 and max_size >= body_size + 1.5
                is_bold = _line_is_bold(line)

                # Known section names are trusted outright. Font-based candidates
                # (custom headings like "Architecture of ZJIT") additionally require
                # bold text — this is what filters out author bylines / footer text
                # that happen to be a slightly larger, non-bold font.
                if is_known_name or (is_larger_font and is_bold and _looks_like_heading(text)):
                    candidates.append((page_num, text))

    doc.close()

    sections = _resolve_candidates_to_sections(candidates, num_pages)

    if not sections:
        sections = [DetectedSection(name="Full Text", order=0, start_page=1, end_page=num_pages)]

    return sections


def _looks_like_heading(text: str) -> bool:
    """Extra filter for font-size-based candidates that aren't in our known list —
    avoids treating random bold/large text (e.g. a table header) as a section."""
    words = text.split()
    if len(words) == 0 or len(words) > 8:
        return False
    # Headings are usually Title Case or ALL CAPS, not full sentences ending in punctuation.
    if text.endswith((".", ",", ";")):
        return False
    return True


def _canonicalize(text: str) -> str:
    match = _SECTION_REGEX.match(text)
    if match:
        return match.group(2).title()
    return text.title()


def _resolve_candidates_to_sections(
    candidates: list[tuple[int, str]], num_pages: int
) -> list[DetectedSection]:
    """
    Convert raw (page, heading_text) hits into non-overlapping sections
    spanning from one heading to the next. Deduplicates repeated hits
    (e.g. running headers matching a section name on every page).
    """
    if not candidates:
        return []

    # Deduplicate consecutive repeats of the same heading (common with running headers)
    deduped: list[tuple[int, str]] = []
    seen_names_recent = None
    for page_num, text in candidates:
        canon = _canonicalize(text)
        if canon == seen_names_recent:
            continue
        deduped.append((page_num, canon))
        seen_names_recent = canon

    # Prefer the FIRST occurrence of each canonical name (avoids running-header noise
    # re-triggering "Results" on every page of that section)
    first_seen: dict[str, int] = {}
    ordered_unique: list[tuple[int, str]] = []
    for page_num, canon in deduped:
        if canon not in first_seen:
            first_seen[canon] = page_num
            ordered_unique.append((page_num, canon))

    ordered_unique.sort(key=lambda x: x[0])

    sections = []
    for i, (page_num, name) in enumerate(ordered_unique):
        end_page = (
            ordered_unique[i + 1][0] - 1 if i + 1 < len(ordered_unique) else num_pages
        )
        end_page = max(end_page, page_num)  # guard against 1-page sections
        sections.append(
            DetectedSection(name=name, order=i, start_page=page_num, end_page=end_page)
        )

    return sections
