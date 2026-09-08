"""
Day 13 (simplified Stage 10): Entity resolution.
Lets users refer to papers by name ("ZJIT", "the lazy basic block
versioning paper") instead of memorizing numeric paper_ids. Matching is
intentionally simple — case-insensitive substring match against a paper's
title AND its auto-extracted subject_name (Day 13) — not full fuzzy/alias
resolution (Stage 10 in the original architecture doc describes a fuller
version: fuzzy matching, aliases, author names, etc. — out of scope here
given time constraints, documented as a simplification).
"""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from backend.database.models import Paper


@dataclass
class ResolvedPaper:
    paper_id: int
    title: str
    subject_name: str | None
    matched_on: str  # "title" | "subject_name" | "exact_id"


def resolve_paper_name(name: str, db: Session) -> list[ResolvedPaper]:
    """
    Resolves a single name string to matching papers. A name that's purely
    numeric is treated as a direct paper_id (convenience for callers mixing
    IDs and names). Otherwise, does a case-insensitive substring match
    against title and subject_name. Returns ALL matches — caller decides
    how to handle zero, one, or multiple matches (e.g. ambiguous name).
    """
    name = name.strip()

    if name.isdigit():
        paper = db.query(Paper).filter(Paper.id == int(name)).first()
        if paper:
            return [
                ResolvedPaper(
                    paper_id=paper.id,
                    title=paper.title,
                    subject_name=paper.subject_name,
                    matched_on="exact_id",
                )
            ]
        return []

    name_lower = name.lower()
    all_papers = db.query(Paper).all()

    matches = []
    for p in all_papers:
        if p.subject_name and name_lower == p.subject_name.lower():
            # Exact subject_name match is the strongest signal — prioritize it
            matches.append(ResolvedPaper(p.id, p.title, p.subject_name, "subject_name"))
        elif p.subject_name and name_lower in p.subject_name.lower():
            matches.append(ResolvedPaper(p.id, p.title, p.subject_name, "subject_name"))
        elif name_lower in p.title.lower():
            matches.append(ResolvedPaper(p.id, p.title, p.subject_name, "title"))

    return matches


def resolve_paper_names(names: list[str], db: Session) -> tuple[list[int], list[str]]:
    """
    Resolves multiple name strings to a deduplicated list of paper_ids.
    Returns (paper_ids, warnings) — warnings describe any name that
    matched zero or multiple papers, so the caller can surface this to
    the user rather than silently guessing.
    """
    paper_ids = []
    warnings = []

    for name in names:
        matches = resolve_paper_name(name, db)

        if not matches:
            warnings.append(f"No paper found matching '{name}'.")
        elif len(matches) > 1:
            titles = ", ".join(f"{m.title} (id={m.paper_id})" for m in matches)
            warnings.append(
                f"'{name}' matched multiple papers: {titles}. Using the first match — "
                f"consider using a more specific name or the numeric paper_id."
            )
            paper_ids.append(matches[0].paper_id)
        else:
            paper_ids.append(matches[0].paper_id)

    # dedupe while preserving order
    seen = set()
    deduped = []
    for pid in paper_ids:
        if pid not in seen:
            seen.add(pid)
            deduped.append(pid)

    return deduped, warnings
