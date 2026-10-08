-- Add query_history table
DROP TABLE IF EXISTS query_history CASCADE;

CREATE TABLE query_history (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    question TEXT NOT NULL,
    paper_ids INTEGER[] NOT NULL,
    paper_titles TEXT[] NOT NULL,
    full_response JSONB NOT NULL,
    is_baseline BOOLEAN DEFAULT false
);

CREATE INDEX IF NOT EXISTS idx_query_history_paper_ids ON query_history USING GIN (paper_ids);
