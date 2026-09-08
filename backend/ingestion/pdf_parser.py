"""
Day 2: PDF ingestion — extraction only.
Turns a PDF file into plain text per page + basic metadata.
No structure detection yet (that's Day 3), no OCR yet (Day 4), no chunking yet (Day 5).
"""

from dataclasses import dataclass, field
from typing import Optional
import fitz  # PyMuPDF


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str


@dataclass
class ParsedPDF:
    filename: str
    filepath: str
    num_pages: int
    title: Optional[str]
    pages: list[PageText] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def avg_chars_per_page(self) -> float:
        if not self.pages:
            return 0.0
        return sum(len(p.text) for p in self.pages) / len(self.pages)


def parse_pdf(filepath: str) -> ParsedPDF:
    """
    Extract page-level text and basic metadata from a PDF using PyMuPDF.
    Raises FileNotFoundError / fitz errors if the file is missing or corrupt.
    """
    doc = fitz.open(filepath)

    metadata = doc.metadata or {}
    title = metadata.get("title") or None

    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append(PageText(page_number=i, text=text))

    filename = filepath.split("/")[-1].split("\\")[-1]

    parsed = ParsedPDF(
        filename=filename,
        filepath=filepath,
        num_pages=len(pages),
        title=title,
        pages=pages,
    )

    doc.close()
    return parsed


def guess_title_from_text(parsed: ParsedPDF) -> str:
    """
    Fallback title guesser: PDF metadata titles are often missing or garbage
    (e.g. 'Microsoft Word - untitled.docx'). If metadata title looks unusable,
    detect the title via font size (the largest-font line on page 1 is
    almost always the title) — this is far more reliable than a naive
    "first reasonably-sized line" heuristic, which can be fooled by
    boilerplate/copyright text that appears earlier in the PDF's raw text
    extraction order despite being positioned as a footer/sidebar visually
    (common on arXiv/NeurIPS PDFs with copyright notices in the margin).
    """
    if parsed.title and len(parsed.title.strip()) > 3 and ".doc" not in parsed.title.lower():
        return parsed.title.strip()

    if not parsed.pages:
        return parsed.filename

    font_based_title = _guess_title_by_font_size(parsed.filepath)
    if font_based_title:
        return font_based_title

    # Fallback: naive first-reasonable-line heuristic (previous behavior)
    first_page_lines = [ln.strip() for ln in parsed.pages[0].text.split("\n") if ln.strip()]
    for line in first_page_lines:
        if 10 < len(line) < 200 and not line.isdigit():
            return line

    return parsed.filename


import re

# arXiv PDFs print a watermark identifier (e.g. "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023")
# in the page margin, sometimes in a font size LARGER than the actual title —
# must be excluded from title candidates or it wins the "largest font" check.
_ARXIV_WATERMARK = re.compile(r"^arXiv:\d+\.\d+", re.IGNORECASE)


def _guess_title_by_font_size(filepath: str) -> str | None:
    """
    Finds the line with the LARGEST font size on page 1 — the title is
    almost always the biggest text on the first page of a research paper.
    Returns None if page 1 has no usable text (falls back to naive heuristic).
    """
    doc = fitz.open(filepath)
    if len(doc) == 0:
        doc.close()
        return None

    page = doc[0]
    page_dict = page.get_text("dict")

    best_line_text = None
    best_size = 0.0

    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text or len(text) < 5 or len(text) > 250:
                continue
            if _ARXIV_WATERMARK.match(text):
                continue  # skip arXiv margin watermark — not the title
            max_size = max(round(s.get("size", 0), 1) for s in spans)
            if max_size > best_size:
                best_size = max_size
                best_line_text = text

    doc.close()
    return best_line_text
