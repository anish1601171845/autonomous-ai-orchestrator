# 🛡️ AegisFlow AI — Self-Healing Multi-Agent Workflow Orchestrator

*Formerly "Autonomous AI Multi-Agent Workflow Orchestrator". Version 2 adds RAG memory, predictive risk detection,
an AI agent debate engine and self-healing recovery to the existing system without rebuilding it.*

## What's new in AegisFlow (v2)

| Enhancement | What it does | Key measured result |
|---|---|---|
| **RAG agent memory** (`app/memory/`) | Stores only QA-verified solutions (dedupe, secret redaction, role/project filters) and retrieves similar past solutions into the Coding Agent's prompt as *untrusted* context. FAISS index; Sentence-Transformers or TF-IDF fallback | TF-IDF Recall@5 **{lex['recall_at_5']}** / MRR **{lex['mrr']}** on lexical queries, but only **{par['recall_at_5']}** / **{par['mrr']}** on paraphrases → the case for embeddings |
| **Predictive failure detection** (`app/ml/failure_predictor.py`) | Calibrated probability that a task fails or misses its deadline, from pre-execution features only; SHAP/linear explanations; Low/Medium/High categories | Deadline-miss PR-AUC **{rk['deadline_missed']['test_calibrated']['pr_auc']}** vs dummy {rk['deadline_missed']['test_dummy']['pr_auc']}; failure PR-AUC **{rk['task_failed']['test_calibrated']['pr_auc']}** vs {rk['task_failed']['test_dummy']['pr_auc']} (**synthetic** execution history) |
| **AI agent debate** (`app/agents/debate_engine.py`) | Developer → Critic → revision → Judge → real QA; ≤3 rounds, time/token budgets, stop reason recorded, LLM failures degrade to a static AST critic | On {d['no_debate']['runs']} planted-defect runs: defects escaping QA **{d['no_debate']['defects_escaping_qa']} → {d['debate']['defects_escaping_qa']}**, QA pass {d['no_debate']['qa_pass_rate']} → **{d['debate']['qa_pass_rate']}**, at +{round(d['debate']['mean_latency_ms']-d['no_debate']['mean_latency_ms'])} ms per task |
| **Self-healing recovery** (`app/recovery/`) | Classified failures, bounded retries with jittered backoff, durable checkpoints + resume, safe reassignment, idempotent external writes, human escalation, fault injection | Recovery success none/fixed/checkpoint/full = **{rc['none']['recovery_success_rate']:.2f} / {rc['fixed_retry']['recovery_success_rate']:.2f} / {rc['checkpoint']['recovery_success_rate']:.2f} / {rc['full']['recovery_success_rate']:.2f}**; lost work {rc['none']['lost_work_tasks']} → **{rc['full']['lost_work_tasks']}** tasks; duplicate PR writes: fixed-retry {rc['fixed_retry']['duplicate_external_actions']}, full **{rc['full']['duplicate_external_actions']}** (simulated faults) |

Full numbers and limitations: [`reports/enhancement_evaluation.md`](reports/enhancement_evaluation.md) · design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §7 ·
audit and change log: [`docs/ENHANCEMENT_AUDIT.md`](docs/ENHANCEMENT_AUDIT.md).

```bash
python main.py evaluate        # re-run all four baseline experiments (~5 min CPU, isolated temp DB)
python main.py test            # 97 tests (57 original + 40 for the enhancements)
```

Try it in the dashboard: **Agent Memory** (similarity search), **Failure Risk** (what-if predictor + SHAP factors),
**Agent Debate** (chat-style Developer/Critic/Judge/QA history) and **Self-Healing Monitor** (inject a fault and watch the
recovery events). Every enhancement can be switched off in `.env` (`MEMORY_ENABLED`, `RISK_ENABLED`, `DEBATE_ENABLED`,
`RECOVERY_MODE=none`).

---

## Original system

A self-governing multi-agent system that takes a plain-English enterprise requirement and **scopes it into tasks,
classifies and prioritises them with ML, predicts their dependencies with a Graph Neural Network, schedules them
across specialised agents with a reinforcement-learning-ready scheduler, generates code and tests, verifies them in a
sandbox, and waits for human approval before touching GitHub**, with a FastAPI backend and a live Streamlit dashboard.

```text
"Build a student management system with registration, login, database integration,
 student dashboard, and automated testing."
        │
        ▼  Scoping → Classification → Dependency GNN → Priority → Workload RL/heuristic → Coding ⇄ QA → Approval
        │
        ▼  11 tasks · valid DAG · 5 agents · generated modules + tests · 37 sandboxed tests passed · awaiting approval
```

> **Data note.** Two models are trained on **real labelled data** (`data/real/`): the public **PROMISE NFR**
> requirements dataset and a **4,000-ticket multilingual support dataset** (en/de/es/fr/pt). The task-category,
> priority, GNN and RL components use a reproducible **synthetic** dataset and a **simulated** scheduler, and are
> labelled as such everywhere. See `reports/model_evaluation.md`.

## Real-data features

| Feature | Data | Result (held-out) |
|---|---|---|
| **Requirements-document scoping:** paste an SRS, each statement is classified functional / 11 NFR types; functional ones are clustered into features, NFRs become cross-cutting tasks | PROMISE NFR: 553 train / 70 test statements, cross-project | F vs NFR accuracy **0.786** (baseline 0.36); 12-type accuracy **0.700** (baseline 0.64) |
| **Ticket intake agent:** triages a ticket in any of 5 languages (type, priority, queue) and can turn it into an engineering workflow | 4,000 multilingual tickets, 80/20 stratified split | type **0.646** (baseline 0.40), priority **0.626** (0.41), queue **0.569** (0.33) |

Short one-line tickets: training on subject-only copies raised priority accuracy on subject-only tickets from 0.531 to **0.625**.

```bash
python main.py train-real      # retrain both (tickets take ~8-10 min on a laptop CPU); models are already included
```

---

## Highlights

| Capability | Implementation |
|---|---|
| Multi-agent orchestration | **LangGraph** `StateGraph`, typed state, structured agent messages, checkpoints, bounded QA retry loop, failure recovery |
| 9 agents | Scoping (+ requirements-document mode) · Planning/Dependency · Classification · Priority · Coding · QA · Workload · GitHub · Ticket Intake (`app/agents/`) |
| Task classification | TF-IDF + Logistic Regression / Random Forest / Gradient Boosting, **GridSearchCV** with GroupKFold |
| Priority prediction | text + urgency + complexity + business impact + dependency fan-out, same model comparison |
| Semantic mapping | BERT / RoBERTa / Sentence-Transformers encoders, offline TF-IDF+LSA fallback; similarity ≠ dependency |
| Dependency GNN | **PyTorch Geometric GraphSAGE** directed link prediction + NetworkX rule baseline + cycle-safe DAG builder |
| RL scheduling | **Gymnasium** environment + **PPO** (Stable-Baselines3) vs round-robin and least-loaded |
| Safe code execution | AST security scan → ruff → pytest in a resource-limited, secret-free subprocess sandbox |
| GitHub | REST client with pagination, rate limits, retries, reviewer recommendation, approval-gated branch + PR (never merges) |
| API / UI / DB | FastAPI (OpenAPI docs) · Streamlit dark dashboard (7 views) · SQLAlchemy (SQLite/PostgreSQL) |
| Quality | 97 pytest tests (unit, integration, end-to-end, GitHub mocks, recovery, approvals, real-data models, enhancements) · ruff clean |

## Results (synthetic / simulated, held-out test projects)

| Component | Result |
|---|---|
| Task classifier (Logistic Regression, selected by GridSearchCV) | accuracy **0.969**, macro-F1 **0.965** |
| Priority predictor (Logistic Regression) | accuracy **0.766**, macro-F1 **0.763** (4 classes, noisy labels by design) |
| Dependency edges: GraphSAGE GNN | avg precision **0.992**, F1 **0.971** |
| Dependency edges: NetworkX rule baseline | avg precision 0.127, F1 0.266 |
| Dependency edges: cosine similarity only | ROC-AUC 0.714, avg precision 0.143 |
| Scheduler makespan, 200 episodes: round-robin / least-loaded / PPO | 33.40 / **26.21** / 26.84 |
| Load-imbalance CV: round-robin / least-loaded / PPO | 0.539 / 0.414 / **0.395** |

PPO learned the most balanced allocation and the highest success rate (0.940), but **did not beat least-loaded on
makespan**, so `SCHEDULER=auto` keeps least-loaded. The GNN's near-perfect score reflects the regular structure of
synthetic project graphs; expect lower numbers on real data. Full details: [`reports/model_evaluation.md`](reports/model_evaluation.md).

## Quick start

```bash
git clone <your-repo-url> && cd autonomous-ai-orchestrator
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env                                          # optional: LLM / GitHub settings

python main.py train                 # synthetic data → EDA → classifier → priority → encoder → GNN (≈2–3 min CPU)
python -m app.rl.train --timesteps 300000                     # optional: PPO scheduler (≈5 min CPU)
python scripts/make_report.py        # regenerate reports/model_evaluation.md + figures
python main.py test                  # 97 tests
python main.py demo --approve        # one end-to-end workflow in the terminal
```

Pretrained models are included in `models/`, so `demo`, `api` and `dashboard` work without retraining.

Run the services in two terminals:

```bash
python main.py api          # http://localhost:8000/docs
python main.py dashboard    # http://localhost:8501
```

Or with Docker: `docker compose up --build`.

> The first API start can take 20–60 s while PyTorch / PyG import.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/workflows` | create and run a workflow (`run_async` default true) |
| GET | `/api/workflows`, `/api/workflows/{id}` | list / detail (tasks, events, summary) |
| POST | `/api/workflows/{id}/approve` | human approval; optional `publish_to_github` |
| GET | `/api/tasks`, `/api/tasks/{task_id}` | tasks across workflows |
| GET | `/api/agents` | agent status, workload, history |
| GET | `/api/metrics` | runtime metrics + model reports |
| POST | `/api/predict/category`, `/api/predict/priority` | model inference |
| POST | `/api/dependencies/predict` | GNN dependency DAG + execution order |
| POST | `/api/requirements/classify` | classify requirement statements (functional / NFR type) |
| POST | `/api/tickets/triage` | triage a ticket (type, priority, queue); `create_workflow: true` starts a workflow |
| GET | `/api/datasets` | real-dataset availability and metrics |
| GET/POST/DELETE | `/api/memory`, `/api/memory/search`, `/api/memory/{id}`, `/api/memory/rebuild`, `/api/memory/retrievals` | agent memory |
| GET/POST | `/api/risk/predictions`, `/api/risk/metrics`, `/api/risk/predict` | failure risk |
| GET | `/api/debates`, `/api/debates/{id}` | debate history |
| GET/POST | `/api/recovery/events`, `/api/recovery/checkpoints/{wf}`, `/api/recovery/demo`, `/api/workflows/{id}/resume` | self-healing |
| GET | `/api/enhancements/status`, `/api/enhancements/evaluation` | enhancement config and results |
| GET | `/api/health`, `/api/github` | readiness, GitHub overview |

```bash
curl -X POST localhost:8000/api/workflows -H 'content-type: application/json' \
  -d '{"requirement":"Create an e-commerce platform with product catalog, shopping cart, payment processing and deploy to production with docker"}'
```

## Configuration

All settings come from environment variables (`.env.example`):

- **No LLM key?** Scoping uses deterministic rules and coding uses tested templates (auth, SQLite repository, analytics, config, CRUD service).
  Set `LLM_API_KEY` (and optionally `LLM_BASE_URL` for Ollama/vLLM) to use an LLM. Invalid LLM output falls back automatically.
- **Transformer embeddings:** set `SEMANTIC_BACKEND=auto` and `SEMANTIC_MODEL=bert-base-uncased` / `roberta-base` / a Sentence-Transformers id, then retrain.
- **GitHub:** read-only with `GITHUB_TOKEN` + `GITHUB_REPO`. PR creation additionally needs `GITHUB_WRITE_ENABLED=true` and an explicit approval.
- **Real data:** `python scripts/collect_github.py owner/repo` exports commits, issues and PRs (paginated, rate-limit aware). Label the issues, then run `python scripts/train_all.py --data your.csv --graphs your_graphs.json`.

## Project layout

```text
app/
  agents/        scoping, classification, planning, priority, workload, coding (+templates), qa, github
  api/           FastAPI app, schemas, routes
  core/          config, typed state, LangGraph orchestrator, security/sandbox, LLM client
  ml/            synthetic data, preprocessing, classification, priority, semantic mapping, GNN
  rl/            Gymnasium environment, schedulers, PPO training
  integrations/  GitHub REST client
  database/      SQLAlchemy models, versioned migrations, repository
  memory/        RAG memory: vector_store.py (FAISS + embeddings/TF-IDF), rag_memory.py, evaluation.py
  recovery/      recovery_controller.py, checkpoint_manager.py, fault_injection.py
dashboard/       Streamlit app
scripts/         train_all, make_report, run_demo, collect_github
tests/           97 tests
reports/         metrics JSON, model_evaluation.md, figures, sample_run/ (generated artifacts of a real run)
docs/            ARCHITECTURE.md, FINAL_REPORT.md, DEPLOYMENT.md, INTERVIEW_AND_RESUME.md
```

## Known limitations

- Task-category, priority, GNN and RL models still use synthetic or simulated data; only requirement classification and ticket triage are measured on real data.
- The ticket dataset was written in a long e-mail style. Queue routing (accuracy about 0.57) is the weakest model and should be used as a suggestion.
- Hugging Face models were not downloadable in the build environment, so shipped checkpoints use the TF-IDF+LSA encoder. The transformer path is implemented, but you need to retrain after enabling it.
- Template code generation covers common archetypes. Arbitrary features become a validated CRUD service unless an LLM is configured.
- The sandbox is process-level isolation. For untrusted LLM code in production, wrap it in a network-less container (see `docs/ARCHITECTURE.md` §5).
- LangGraph checkpoints are in memory. Workflow state is persisted to the database after every node, but a crashed run is marked failed, not resumed.
