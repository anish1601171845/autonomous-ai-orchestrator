# Deployment

## Local (development)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
uvicorn app.api.main:app --host 0.0.0.0 --port 8000      # API + OpenAPI docs at /docs
streamlit run dashboard/streamlit_app.py                  # dashboard at :8501 (API_URL points to the API)
```

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

The image installs CPU PyTorch, copies the pretrained models in `models/`, and runs as a non-root user.
`./data` and `./workspace` are mounted so workflows and generated artifacts survive restarts.

## PostgreSQL

```bash
pip install "psycopg[binary]"
export DATABASE_URL=postgresql+psycopg://user:password@host:5432/orchestrator
```

Tables are created on startup (`app/database/models.py:init_db`). For schema evolution, add Alembic migrations.

## Production checklist

- [ ] Put the API behind an authenticating reverse proxy (the API has no user auth of its own).
- [ ] Run sandboxed test execution in a container with `--network none --read-only --pids-limit 128 --memory 1g`,
      or set `ALLOW_SANDBOX_EXECUTION=false` and run QA in CI instead.
- [ ] Keep `GITHUB_WRITE_ENABLED=false` unless PR creation is needed. Use a fine-grained token scoped to one repository
      with `contents:write` and `pull_requests:write` only.
- [ ] Store secrets in a secret manager and inject them as environment variables. Never commit `.env`.
- [ ] Retrain on labelled real data (`scripts/collect_github.py` → label → `scripts/train_all.py --data ...`) and
      review `reports/model_evaluation.md` before trusting model outputs.
- [ ] Run several API workers (`uvicorn --workers 4`) only with PostgreSQL. SQLite allows a single writer.
