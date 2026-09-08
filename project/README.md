# Research QA System — Day 1 Setup

## What Day 1 gives you
- A running Postgres + pgvector database (via Docker) with the full schema applied
- A FastAPI app that can talk to it
- Two endpoints to prove it all works: `/health` and `/health/db`

Everything below happens **on your machine** — I can't run Docker, start servers,
or execute long-running processes from here, so this checklist is what you run locally.

---

## 1. Unzip / place the project
Put the `project/` folder somewhere on your machine, e.g. `~/dev/research-qa/`.

## 2. Create your .env file
```bash
cd project
cp .env.example .env
```
Open `.env` and fill in `GROK_API_KEY` when you have it (not needed until Day 6 — fine to leave blank for now).

## 3. Start Postgres + pgvector via Docker
Make sure Docker Desktop is running, then:
```bash
docker compose up -d
```
This pulls the `pgvector/pgvector:pg16` image, starts Postgres on port 5432,
and automatically runs `backend/database/schema.sql` to create all tables
(papers, sections, chunks, authors, citations) on first boot.

Verify it's running:
```bash
docker ps
```
You should see a container named `research_qa_db`.

## 4. Set up Python environment
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```
Note: `torch` + `transformers` are large downloads (~2-3 GB) — this is why we're
installing them now even though we don't use them until Day 3/5.

## 5. Run the API
From the `project/` root:
```bash
uvicorn backend.api.main:app --reload
```
It should start on `http://localhost:8000`.

## 6. Verify the checkpoint
Open in browser or curl:
```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/health/db
# {"database":"connected","pgvector_installed":true}

curl http://localhost:8000/papers/count
# {"papers_count":0}
```

If `pgvector_installed` is `true` and `papers_count` returns `0` (not an error),
**Day 1 checkpoint is complete**: the API is alive and talking to a database
that understands vectors.

---

## Troubleshooting
- **Docker container won't start / port 5432 already in use**: you likely have
  a local Postgres already running. Either stop it, or change the port mapping
  in `docker-compose.yml` (e.g. `"5433:5432"`) and update `DATABASE_URL` in `.env` to match.
- **`ModuleNotFoundError: backend`**: run `uvicorn` from the `project/` root
  directory, not from inside `backend/`.
- **pgvector extension missing**: confirm you're using the `pgvector/pgvector:pg16`
  image (not plain `postgres`) — plain Postgres doesn't have the extension.

---

## Next: Day 2
PDF ingestion — PyMuPDF text extraction + basic metadata into the `papers` table.
