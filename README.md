# Research QA System

An evidence-verified, multi-paper question-answering system. Upload research
paper PDFs, ask natural-language questions, and get answers where every
factual claim is independently checked against its cited evidence, not
just generated and trusted.

See Research_QA_System_Report.docx (if included) for the full architecture
write-up, findings, and the accuracy-improvement methodology (Fixes 1-3).

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

## Quick start (every time you run it)

```
cd project
docker compose up -d
venv\Scripts\activate
uvicorn backend.api.main:app --reload
```
Open http://localhost:8000 in your browser.

To stop: Ctrl+C to stop uvicorn, then run docker compose down.

Checklist if something's not working:
- Docker Desktop must be open BEFORE running docker compose up -d
- docker ps should show research_qa_db as Up
- Your terminal prompt should show (venv) after activating
- .env needs a valid GROQ_API_KEY with quota remaining (check
  https://console.groq.com if requests are failing with 429 errors)

---

## First-time setup (only needed once)

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
(several GB total) - this step takes a while the first time. The NLI
verification model alone (~870MB) downloads automatically on first use.

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

To start fresh: the red "Clear all papers" button wipes everything and
resets paper IDs back to 1.

### API (for testing/scripting)
Key endpoints:
```
POST /ingest/pdf                         upload a PDF
GET  /papers                             list all papers + auto-detected subject names
POST /papers/{id}/extract-subject-name   retry subject-name detection without re-uploading
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

Note: each run makes 10+ calls to the Groq API (5 questions x 2 systems),
which can add up against daily token quotas if run repeatedly. Citation
validity naturally varies by 10-15 points run-to-run due to LLM
non-determinism - for an honest number, run it 3-5 times and average,
rather than trusting a single run.

### Debugging tools
Standalone scripts used during development, kept for reference:
```
python diagnose_nli.py <chunk_id> "<claim text>"     inspect NLI scoring for the original model
python diagnose_nli_fix3.py                            sanity-check the upgraded NLI model
python debug_chunk_nli.py <chunk_id> "<claim text>"  inspect sentence-selection + NLI for a real chunk
```

---

## Accuracy improvement history

Starting citation validity (before any accuracy-focused fixes): 50%

Changes made, in order:
1. Fix 1 - joint citation verification: for claims citing 2+ evidence
   chunks, also check the combined evidence, not just each chunk alone.
2. Fix 2 - atomic claim prompting: the LLM is now instructed to write one
   fact per claim instead of compound sentences, with a worked example
   in the prompt. This increased output length, which required raising
   the generation token limit from 1500 to 2500 to avoid truncated JSON.
3. Tested and reverted - temperature 0.0: hypothesis was that lower
   temperature (more deterministic output) would improve consistency.
   Measured result: 78.2% at temperature 0.2 vs 69.0% at temperature 0.0
   across repeated runs - a real regression. Reverted to 0.2.
4. Tested and reverted - larger chunk size (180 to 250 words): hypothesis
   was that bigger chunks keep compound facts together. Measured result:
   76.3% at 180 words vs 70.2% at 250 words - a regression (note: this
   test was run alongside a prompt wording change, so the two are
   confounded and the cause cannot be fully isolated). Reverted to 180.
5. Fix 3 - NLI model upgrade: switched from cross-encoder/nli-deberta-v3-base
   to MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, a larger
   model also trained on FEVER and ANLI (fact-verification-focused
   datasets). Result: 76.9% average, a modest further improvement with
   no runs below the prior model's floor.

Final citation validity: approximately 76.9% average (measured across
4-6 runs per configuration, individual runs ranging 69-83%), up from 50%
at the start of this effort - a +26.9 percentage point improvement.

Citation validity varies run-to-run by roughly 10-15 points even with
zero code changes, due to LLM non-determinism in how claims get phrased.
All numbers above are multi-run averages, not single best-case results.
See the full report document for detailed methodology and evidence
behind each change.

---

## Known limitations (see full report for details)

- Citation verification is conservative, especially for claims that
  combine multiple facts in one sentence (common in LLM-generated answers
  and cross-paper comparisons). A FLAGGED claim often means "unconfirmed,"
  not "wrong."
- Run-to-run variance of 10-15 points is normal even with zero code
  changes, due to LLM non-determinism affecting exact claim phrasing.
- Multi-paper queries don't apply the subject-name pronoun-resolution
  fix (would require guessing which paper a "we" refers to), so citation
  verification is somewhat less reliable on comparison questions than
  single-paper ones.
- Groq's rate limits (both per-minute and per-day) can occasionally cause
  a request to be slow (automatic retry is built in for per-minute limits)
  or, for the daily quota, simply unavailable until it resets - if a
  request fails outright with a 429 error, check your quota at
  console.groq.com.
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
    ingestion/         - PDF parsing, structure detection, chunking
    retrieval/         - lexical, semantic, hybrid search; entity resolution
    nlp/               - embeddings, NLI, sentence selection
    llm/               - Groq client, answer synthesis, baseline, subject extraction
    verification/      - consensus + citation verification (incl. joint checking)
    database/          - schema, models, migrations
  frontend/            - index.html, app.js, style.css (served by FastAPI)
  evaluation/          - comparison script
  diagnose_nli.py, diagnose_nli_fix3.py, debug_chunk_nli.py - debugging tools
```