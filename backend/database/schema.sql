-- Day 1: Core schema (papers, sections, chunks, authors, citations)
-- Run automatically by docker-compose on first container start.
-- To re-run manually: psql -U research_admin -d research_qa -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- ─────────────────────────────
-- Papers
-- ─────────────────────────────
CREATE TABLE IF NOT EXISTS papers (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    subject_name    TEXT,                  -- e.g. "ZJIT" — auto-extracted at ingestion (Day 13)
    filename        TEXT NOT NULL,
    filepath        TEXT NOT NULL,
    year            INTEGER,
    abstract        TEXT,
    num_pages       INTEGER,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ─────────────────────────────
-- Authors
-- ─────────────────────────────
CREATE TABLE IF NOT EXISTS authors (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id        INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    author_id       INTEGER REFERENCES authors(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_id, author_id)
);

-- ─────────────────────────────
-- Sections
-- ─────────────────────────────
CREATE TABLE IF NOT EXISTS sections (
    id              SERIAL PRIMARY KEY,
    paper_id        INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,        -- e.g. "Results", "Methodology"
    section_order   INTEGER,              -- order within the paper
    start_page      INTEGER,
    end_page        INTEGER
);

-- ─────────────────────────────
-- Chunks (the core retrieval unit)
-- ─────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    paper_id        INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    section_id      INTEGER REFERENCES sections(id) ON DELETE SET NULL,
    page            INTEGER,
    chunk_index     INTEGER,               -- order of chunk within the paper
    text            TEXT NOT NULL,
    search_vector   tsvector,               -- lexical (full-text) search
    embedding       vector(384),            -- matches all-MiniLM-L6-v2 dim; change if you swap models
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Keep search_vector automatically in sync with text
CREATE OR REPLACE FUNCTION chunks_search_vector_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', NEW.text);
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunks_search_vector ON chunks;
CREATE TRIGGER trg_chunks_search_vector
    BEFORE INSERT OR UPDATE OF text ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_search_vector_update();

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_search_vector ON chunks USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks (paper_id);
-- Vector index (IVFFlat) — build only after you have data (needs ANALYZE); safe to create empty too.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ─────────────────────────────
-- Citations (paper A cites paper B) — used in later/future stages
-- ─────────────────────────────
CREATE TABLE IF NOT EXISTS citations (
    id              SERIAL PRIMARY KEY,
    citing_paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    cited_paper_id  INTEGER REFERENCES papers(id) ON DELETE SET NULL,
    raw_text        TEXT
);
