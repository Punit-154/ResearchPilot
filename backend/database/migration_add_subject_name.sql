-- Migration: add subject_name column to existing papers table.
-- Your database already has data (papers 1-8), so schema.sql's CREATE TABLE
-- IF NOT EXISTS won't touch it — run this migration once to add the new
-- column without losing existing data.
--
-- Run with:
--   docker exec -i research_qa_db psql -U research_admin -d research_qa < backend/database/migration_add_subject_name.sql
-- (adjust container name if yours differs — check with `docker ps`)

ALTER TABLE papers ADD COLUMN IF NOT EXISTS subject_name TEXT;
