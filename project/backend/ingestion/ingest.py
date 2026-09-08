"""
Day 2: Ingestion orchestration.
Takes a saved PDF file path, parses it, and inserts a row into `papers`.
Does NOT create chunks yet — that's Day 5. This stage's checkpoint is just:
PDF file -> row in `papers` table with correct metadata and page count.
"""

from sqlalchemy.orm import Session

from backend.database.models import Paper, Section, Chunk
from backend.ingestion.pdf_parser import parse_pdf, guess_title_from_text
from backend.ingestion.structure import detect_sections
from backend.ingestion.chunker import chunk_paper
from backend.nlp.embeddings import embed_texts


def ingest_pdf(filepath: str, db: Session) -> Paper:
    """
    Parse the PDF at `filepath`, persist a Paper row, and (Day 3) detect +
    persist its Section rows. Returns the created Paper ORM object
    (with .id populated and .sections loadable).
    """
    parsed = parse_pdf(filepath)
    title = guess_title_from_text(parsed)

    # crude abstract guess: first ~500 chars of page 1 minus the title line
    # (a real abstract extractor comes in Day 3's structure detection)
    abstract_guess = None
    if parsed.pages:
        abstract_guess = parsed.pages[0].text[:800].strip() or None

    paper = Paper(
        title=title,
        filename=parsed.filename,
        filepath=parsed.filepath,
        year=None,  # left for future metadata pass or manual entry
        abstract=abstract_guess,
        num_pages=parsed.num_pages,
    )

    db.add(paper)
    db.commit()
    db.refresh(paper)

    # Day 3: structure detection
    detected = detect_sections(filepath, parsed.num_pages)
    section_id_by_order = {}
    for d in detected:
        section = Section(
            paper_id=paper.id,
            name=d.name,
            section_order=d.order,
            start_page=d.start_page,
            end_page=d.end_page,
        )
        db.add(section)
        db.flush()  # populate section.id without a full commit yet
        section_id_by_order[d.order] = section.id
    db.commit()
    db.refresh(paper)

    # Day 5: chunking — reuses already-parsed page text, no re-parsing needed
    candidates = chunk_paper(parsed.pages, detected, section_id_by_order)

    # Day 7: embeddings — batch-encode all chunk texts at once (much faster
    # than one-by-one) before inserting
    chunk_texts = [c.text for c in candidates]
    embeddings = embed_texts(chunk_texts)

    for c, emb in zip(candidates, embeddings):
        chunk = Chunk(
            paper_id=paper.id,
            section_id=c.section_id,
            page=c.page,
            chunk_index=c.chunk_index,
            text=c.text,
            embedding=emb,
        )
        db.add(chunk)
    db.commit()
    db.refresh(paper)

    return paper
