# Research QA System

An evidence-verified, multi-paper question-answering system. Upload research
paper PDFs, ask natural-language questions, and get answers where every
factual claim is independently checked against its cited evidence, not
just generated and trusted.

See Research_QA_System_Report.docx (if included) for the full architecture
write-up, findings, and comparison against a simple RAG baseline.

---

## What's included

- Backend: FastAPI + PostgreSQL/pgvector, hybrid (lexical + semantic)
  retrieval, NLI-based claim verification, LLM synthesis via Groq
- Frontend: a plain HTML/JS/CSS single-page app, served directly by
  FastAPI (no Node/npm required)
- Baseline system: a simple RAG pipeline for side-by-side comparison
- Evaluation script: runs shared questions through both systems and
  reports latency, keyword recall, and citation health

---

## First-time setup

### 1. Prerequisites
- Docker Desktop (running)
- Python 3.11+
- A Groq API key from https://console.groq.com (free tier works. Note: this
  is Groq, the fast-inference company, not xAI's "Grok" - different
  products with similar names)

### 2. Configure environment
```
cd project
cp .env.example .env
```
Edit .env and set:
```
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_API_BASE=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.3-70b-versatile
```
(GROQ_MODEL may need updating if Groq deprecates this model - check
https://console.groq.com/docs/models if you get a "model not found" error.)

### 3. Start the database
```
docker compose up -d
docker ps
```
Confirm "research_qa_db" shows as Up.

### 4. Set up Python environment
```
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
```
Note: torch/transformers/sentence-transformers are large downloads
(several GB total) - this step takes a while the first time.

### 5. Run the app
```
uvicorn backend.api.main:app --reload
```

### 6. Open the app
Go to http://localhost:8000 in your browser. Upload a PDF, select it,
and ask a question.

---

## If you're setting this up on a database that already has data

If you previously ran an older version of this project, your database may
be missing the subject_name column added later. Run this once:
```
docker exec -i research_qa_db psql -U research_admin -d research_qa < backend/database/migration_add_subject_name.sql
```

---

## Using the system

### Web UI (recommended)
1. Upload one or more PDFs from the left panel
2. Click a paper card to select it (click multiple for a comparison query)
3. Type a question, optionally check "also run baseline"
4. Click Ask

Claims are color-coded:
- VALID (green) - independently confirmed by a second AI model checking the
  cited evidence
- FLAGGED (amber) - could not be confirmed automatically. This does NOT
  mean the claim is false - often it's correct but combines multiple facts
  in a way the automated checker can't verify with confidence. Treat it as
  "worth a manual double-check," not "wrong."
- NO_CITATIONS (gray) - a claim with no cited evidence (should be rare)

Click "What do citation and verification status mean?" in the UI for the
same explanation, ready to show if asked.

### API (for testing/scripting)
Key endpoints:
```
POST /ingest/pdf                         upload a PDF
GET  /papers                             list all papers + auto-detected subject names
GET  /answer?q=...&paper_id=1            full verified answer (repeat paper_id for multiple papers)
GET  /answer?q=...&paper_name=ZJIT       same, but reference a paper by name instead of ID
GET  /baseline/answer?q=...&paper_id=1   simple RAG, no verification
GET  /verify/claim?q=...&claim=...       check one specific claim against retrieved evidence
DELETE /papers?confirm=true              wipe all papers and reset IDs (careful, irreversible)
```
Full interactive API docs: http://localhost:8000/docs

### Evaluation script
Compares the full architecture against the baseline on a shared question set:
```
python evaluation/run_evaluation.py
```
Edit PAPER_ID and SUBJECT_NAME near the top of the script to match your
currently-ingested paper before running. Results are saved to
evaluation/results.json.

---

## Known limitations (see full report for details)

- Citation verification is conservative, especially for claims that
  combine multiple facts in one sentence (common in LLM-generated answers
  and cross-paper comparisons). A FLAGGED claim often means "unconfirmed,"
  not "wrong."
- Multi-paper queries don't apply the subject-name pronoun-resolution
  fix (would require guessing which paper a "we" refers to), so citation
  verification is somewhat less reliable on comparison questions than
  single-paper ones.
- Groq's free-tier rate limits can occasionally cause a request to be
  slow (automatic retry is built in) - if a request fails outright, wait
  30-60 seconds and retry.
- Entity resolution (matching a paper by name) uses simple substring
  matching, not full fuzzy/alias matching.

---

## Project structure
```
project/
  docker-compose.yml
  .env.example
  backend/
    api/routes/       - all HTTP endpoints
    ingestion/        - PDF parsing, structure detection, chunking
    retrieval/        - lexical, semantic, hybrid search; entity resolution
    nlp/              - embeddings, NLI, sentence selection
    llm/              - Groq client, answer synthesis, baseline, subject extraction
    verification/     - consensus + citation verification
    database/         - schema, models, migrations
  frontend/           - index.html, app.js, style.css (served by FastAPI)
  evaluation/         - comparison script
  diagnose_nli.py, debug_chunk_nli.py - standalone debugging tools used during development
```
