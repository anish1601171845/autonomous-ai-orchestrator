# AegisFlow AI — Implementation Audit, Checklist and Change Log

## 1. Audit of the existing repository (before changes)

| Area | Found | Notes |
|---|---|---|
| Orchestration | LangGraph `StateGraph`, typed `WorkflowState`, nodes scope → classify → plan → prioritize → allocate → select_task ⇄ execute → qa → finalize | `InMemorySaver` checkpointer: **not durable** across a process crash |
| Agents | scoping (+requirements-document mode), classification, planning (GNN), priority, workload (RL/heuristic), coding (LLM or templates), QA (AST scan + ruff + sandboxed pytest), GitHub, ticket intake | QA retry loop bounded by `MAX_QA_RETRIES` |
| Persistence | SQLAlchemy, SQLite default, PostgreSQL optional; tables `workflows`, `tasks`, `events` | `create_all` only, so no migrations. `save_state` **deleted and re-inserted** task rows each node, so task primary keys were unstable |
| Datasets | synthetic task/graph data; real PROMISE NFR requirements; real 4k multilingual tickets | **None contains agent execution outcomes.** No GitHub repository-metrics dataset was provided |
| Dashboard | Streamlit, 9 pages, API-or-local data access | local mode re-implemented each endpoint by hand |
| Tests | 57 passing (`pytest`) | baseline before any change |
| Missing | memory, failure prediction, debate, durable recovery, fault injection, idempotency for external writes | all four requested capabilities absent |

## 2. Smallest set of changes (decisions)

- **Reuse LangGraph:** debate is a new node between `execute` and `qa`. Risk runs inside `allocate`, memory inside `execute` and `qa`, and recovery wraps execution and persistence. No second orchestration framework was added.
- **Durable checkpoints in the existing database** (`workflow_checkpoints`), not LangGraph's separate SQLite/Postgres saver packages. This avoids a second store and works with both SQLite and PostgreSQL.
- **Task rows are upserted** by `(workflow_id, task_id)`, which gives stable primary keys that the new tables reference with foreign keys.
- **Versioned migrations** (`app/database/migrations.py`, table `schema_version`). Migration 1 creates the 9 new tables. Migration 2 adds `tasks.risk_probability` and `risk_category` to existing databases, including the Docker volume.
- **Each enhancement is independently configurable** and fail-safe: an exception inside an enhancement is logged and the core workflow continues.
- **Dashboard local mode** now calls the FastAPI app in-process (TestClient). Every new endpoint therefore works without the API server and with no duplicated logic.

## 3. Phase checklist (all executed; tests were run after each phase)

| Phase | Deliverable | Status |
|---|---|---|
| 1 | 9 tables with foreign keys to `workflows` and `tasks`, timezone-aware timestamps, migrations, task upsert | ✅ legacy-database migration verified |
| 2 | `app/memory/` (`save_verified_solution`, `retrieve_similar_solutions`, `get_agent_memory`, `delete_memory`, `build_memory_context`, `rebuild_index`), labelled retrieval benchmark | ✅ |
| 3 | `app/ml/failure_predictor.py` (`prepare_failure_dataset`, `train_failure_model`, `evaluate_failure_model`, `predict_failure_probability`, `predict_deadline_risk`, `explain_risk_prediction`) | ✅ |
| 4 | `app/agents/debate_engine.py` (`start_debate`, `submit_proposal`, `review_proposal`, `revise_solution`, `evaluate_solution`, `get_debate_history`) | ✅ |
| 5 | `app/recovery/` (`detect_failure`, `classify_failure`, `save_checkpoint`, `restore_checkpoint`, `retry_task`, `reassign_task`, `record_recovery_event`, `escalate_failure`, `run_external_once`, fault injection with 8 scenarios) | ✅ |
| 6 | Workflow integration; idempotent GitHub publish on approval; crash resume from checkpoint | ✅ |
| 7 | 4 dashboard pages, 15 API endpoints, 5 new test files (40 tests) | ✅ 97 / 97 tests pass |
| 8 | `scripts/evaluate_enhancements.py`, `reports/enhancement_evaluation.{json,md}`, 3 figures, docs | ✅ |

## 4. Regressions found and fixed during integration

- The coding-agent mock in an existing test lacked the new `memory_context` argument. The mock was updated; production code was unchanged.
- QA reassignment adds one deliberate extra attempt after the QA budget is spent. The existing bounded-retry test now asserts `MAX_QA_RETRIES + 2` attempts and checks the escalation flag.
- LangGraph silently drops undeclared state keys, so `restored_from` and `resumes` were added to `WorkflowState`.
- The task API schema did not expose the new risk columns, which surfaced as an integration-test failure. The schema was extended.
- The fault-demo endpoint prefixed the requirement text, which polluted scoping feature names. The scenario is now recorded as a separate event.

## 5. Remaining limitations (honest)

1. **Embeddings were not exercised in the build environment.** The Hugging Face model could not be downloaded there, so every memory number shown is TF-IDF. The embedding path is implemented, and `python main.py evaluate` fills the embedding rows wherever the model loads.
2. **The failure predictor is trained on synthetic execution history.** Live executions are logged to `execution_metrics`. Retrain with `prepare_failure_dataset(source="live")` once enough real runs exist; live deadline semantics use `TASK_DEADLINE_S_PER_COMPLEXITY`.
3. **The debate defaults to a static AST critic plus the template developer.** It cannot judge semantics. For example, removed validation is detected only when no `raise` remains. Contract tests added during revision check existence and callability, not behaviour. With `LLM_API_KEY` set, an LLM critic is used.
4. **Memory cannot change task success with the template backend**, because generated code ignores prompt context. The measured effect is extra prompt tokens. A success-rate effect needs an LLM backend.
5. **All recovery results use injected faults.** Process-level checkpoint resume runs in the same Python process. A real host crash is covered by `POST /api/workflows/{id}/resume` from the durable checkpoint, but there is no automatic supervisor that restarts the process.
6. **The sandbox is process-level isolation** (unchanged from v1). Use a network-less container for untrusted LLM code.
