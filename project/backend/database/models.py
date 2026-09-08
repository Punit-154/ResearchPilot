"""
Day 1: ORM models mirroring schema.sql.
These let the rest of the backend (ingestion, retrieval, etc.) interact with
the DB using Python objects instead of raw SQL.
"""

from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from backend.database.db import Base


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    filename = Column(Text, nullable=False)
    filepath = Column(Text, nullable=False)
    year = Column(Integer)
    abstract = Column(Text)
    num_pages = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    sections = relationship("Section", back_populates="paper", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="paper", cascade="all, delete-orphan")


class Author(Base):
    __tablename__ = "authors"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False, unique=True)


class Section(Base):
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"))
    name = Column(Text, nullable=False)
    section_order = Column(Integer)
    start_page = Column(Integer)
    end_page = Column(Integer)

    paper = relationship("Paper", back_populates="sections")
    chunks = relationship("Chunk", back_populates="section")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"))
    section_id = Column(Integer, ForeignKey("sections.id", ondelete="SET NULL"))
    page = Column(Integer)
    chunk_index = Column(Integer)
    text = Column(Text, nullable=False)
    # search_vector is managed by a DB trigger — not set from Python
    embedding = Column(Vector(384))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    paper = relationship("Paper", back_populates="chunks")
    section = relationship("Section", back_populates="chunks")


class Citation(Base):
    __tablename__ = "citations"

    id = Column(Integer, primary_key=True)
    citing_paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"))
    cited_paper_id = Column(Integer, ForeignKey("papers.id", ondelete="SET NULL"))
    raw_text = Column(Text)
