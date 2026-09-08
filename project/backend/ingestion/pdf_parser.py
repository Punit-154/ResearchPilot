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
    fall back to the first non-empty, reasonably-short line on page 1 —
    a decent heuristic for research paper titles.
    """
    if parsed.title and len(parsed.title.strip()) > 3 and ".doc" not in parsed.title.lower():
        return parsed.title.strip()

    if not parsed.pages:
        return parsed.filename

    first_page_lines = [ln.strip() for ln in parsed.pages[0].text.split("\n") if ln.strip()]
    for line in first_page_lines:
        # Titles are usually not too short, not too long, and not all-numeric
        if 10 < len(line) < 200 and not line.isdigit():
            return line

    return parsed.filename
