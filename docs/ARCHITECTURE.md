# Architecture

## 1. Component view

```text
                ┌────────────────────────── Streamlit dashboard (dashboard/streamlit_app.py) ─────────────────────────┐
                │ Overview · Workflow mgmt + approval · Agents · Dependency DAG · ML analytics · RL analytics · GitHub  │
                └───────────────────────────────▲─────────────────────────────────────────────────────────────────────┘
                                                │ REST (httpx)  — falls back to direct DB reads if the API is offline
┌───────────────────────────────────────────────┴───────────── FastAPI (app/api) ─────────────────────────────────────┐
│ /api/workflows · /api/tasks · /api/agents · /api/metrics · /api/predict/* · /api/dependencies/predict · /approve   │
└──────────────┬─────────────────────────────────────────────────────────────────────────────────┬───────────────────┘
               │ BackgroundTasks                                                                  │ SQLAlchemy
┌──────────────▼──────────── LangGraph orchestrator (app/core/orchestrator.py) ──────────┐  ┌─────▼──────────────┐
│ typed WorkflowState · InMemorySaver checkpoints (thread_id = workflow_id)                │  │ SQLite / Postgres  │
│                                                                                          │  │ workflows · tasks  │
│ scope → classify → plan → prioritize → allocate → select_task ⇄ execute → qa → finalize  │─▶│ events (messages)  │
└───┬─────────┬──────────┬──────────┬───────────┬────────────┬──────────┬──────────┬──────┘  └────────────────────┘
    │         │          │          │           │            │          │          │
 Scoping  Classif.   Planning   Priority    Workload      Coding       QA       GitHub
 agent    agent      agent      agent       agent         agent        agent    agent
 (LLM or  (TF-IDF+   (semantic  (text+num   (PPO policy   (LLM or      (AST     (REST client:
 rules)   LogReg/RF/ +GraphSAGE features,   or least-     templates)   scan,    pagination,
          GB, Grid-  GNN + DAG  GridSearch) loaded/RR)                 ruff,    rate limits,
          SearchCV)  repair)                                           sandbox  approval-gated
                                                                       pytest)  branch + PR)
```

### Real-data entry points

- **Requirements-document mode** (`app/agents/requirements_scoping.py`): if the input has 3 or more `shall`/`must` statements, each is
  classified by the PROMISE-trained model. Functional statements are clustered (TF-IDF + KMeans) into features, each with
  implement and test tasks whose acceptance criteria are the original sentences. NFRs are grouped by type into cross-cutting
  tasks: security controls, integrations, performance tests, UI review, availability, compliance and maintainability docs.
- **Ticket intake** (`app/agents/intake_agent.py`): a multilingual ticket is triaged (type, priority, queue). With
  `create_workflow`, it becomes a workflow whose task template depends on the ticket type, and whose urgency comes from the
  predicted priority.

## 2. Workflow state machine

| Node | Agent | Reads | Writes | Next |
|---|---|---|---|---|
| `scope` | Requirement Scoping | requirement | requirements, tasks (complexity, urgency, impact, acceptance criteria) | `classify` |
| `classify` | Task Classification | task descriptions | category + confidence | `plan` |
| `plan` | Planning & Dependency | tasks | dependency_ids, edges, rejected edges | `prioritize` |
| `prioritize` | Priority Prediction | tasks + dependency fan-out | priority + confidence | `allocate` |
| `allocate` | Workload Allocation | tasks, agent pool | assigned_agent, execution_order, queue | `select_task` |
| `select_task` | orchestrator | queue, task statuses | current_task_id; marks tasks **blocked** if a prerequisite failed | `execute` or `finalize` |
| `execute` | Coding / docs / devops producer | current task | artifacts in `workspace/<workflow_id>/` | `qa` |
| `qa` | Quality Assurance | artifacts | qa_report, status | `execute` (retry ≤ `MAX_QA_RETRIES`) or `select_task` |
| `finalize` | orchestrator | everything | summary, final sandboxed test run, `workflow_summary.md` | END (`awaiting_approval`) |

Every node is wrapped by `_persisting()`, which (a) appends structured `AgentMessage`s, (b) writes the merged state
and tasks to the database, (c) converts an unexpected stage exception into a `failed` workflow instead of a crash.
Task-level exceptions inside `execute` are recorded on the task, which is then failed, and its dependants are
**blocked** while independent branches keep running (tested in `tests/test_orchestrator_api.py`).

**Why priority runs after planning:** the priority model uses dependency fan-out (how many tasks a task unblocks)
as a feature, so it needs the dependency graph first. Classification still precedes planning because the
GNN uses the predicted category as a node feature.

## 3. Dependency engine

1. **Semantic layer** (`app/ml/semantic_mapping.py`): task embeddings from BERT / RoBERTa (mean pooling),
   Sentence-Transformers, or the offline TF-IDF + LSA fallback. Produces *unordered* similarity candidates.
   Similarity is never treated as dependency: it is symmetric and, in our evaluation, a weak signal.
2. **GNN** (`app/ml/gnn_model.py`): 2-layer GraphSAGE encoder over a message-passing graph built only from
   information available before dependencies exist (semantic kNN + same-feature edges). Node features =
   embedding ‖ category one-hot ‖ project-level flag ‖ complexity. An MLP scores every *ordered* pair
   `[h_u, h_v, h_u⊙h_v, cos, same_feature]`, so the model learns direction.
3. **Validation**: edges are inserted in descending confidence and any edge that would close a cycle is
   rejected (`build_dag`); a transitive reduction keeps the plan readable; execution order is a
   priority-aware lexicographical topological sort.
4. **Baseline**: a NetworkX rule heuristic (stage order within the same feature) is evaluated side by side and
   used automatically if no GNN checkpoint matching the active encoder exists.

## 4. RL scheduler

`SchedulingEnv` (Gymnasium) is a list-scheduling simulator: each step presents the highest-priority ready task;
the action picks an agent. Duration depends on complexity, agent skill and specialisation; failures (rework)
depend on complexity and skill. Reward = success bonus − priority-weighted latency − load imbalance; invalid
(capability-mismatched) actions are penalised and not executed. PPO (Stable-Baselines3) is compared with
round-robin and least-loaded on 200 unseen seeded episodes. `SCHEDULER=auto` uses PPO **only** if the saved
evaluation shows it beats least-loaded on makespan; otherwise least-loaded is used.

## 5. Safety model

| Risk | Control |
|---|---|
| Generated code doing harm | AST policy scan (no subprocess/socket/network/eval/exec/delete) **before** any execution; failing scan ⇒ tests are not run |
| Resource abuse | sandbox subprocess: isolated interpreter (`-I`), scrubbed environment (no tokens), dead proxy for network, CPU/memory/file-size rlimits, wall-clock timeout, separate session |
| Path traversal | workspace writes resolved and confined to `workspace/<workflow_id>/` |
| Secret leakage | tokens only from env vars; log filter and `redact()` on logs, events, sandbox output and API errors |
| Unwanted repo changes | GitHub client is read-only unless `GITHUB_WRITE_ENABLED=true` **and** the human approves the workflow; merging is not implemented |
| Infinite agent loops | QA retries bounded by `MAX_QA_RETRIES`; LangGraph `recursion_limit` |

**Production hardening still recommended:** run the sandbox inside a container or microVM with no network
namespace (e.g. Docker `--network none --read-only --pids-limit`), because rlimits + scrubbed env are process-level
isolation only.

## 6. Data model

`workflows(id, requirement, status, created_at, updated_at, state JSON, summary JSON, approval JSON)`
`tasks(workflow_id, task_id, title, description, feature, category, category_confidence, priority, priority_confidence, complexity, status, assigned_agent, attempts, dependency_ids, artifacts, qa_report, error)`
`events(workflow_id, timestamp, sender, recipient, kind, content JSON)` — the agent message log.


## 7. AegisFlow enhancements (v2)

```text
scope → classify → plan → prioritize → allocate ──► [risk predictor: calibrated P(fail), P(late), SHAP factors]
                                          │
                                   select_task ◄─────────────────────────────────────────────┐
                                          │                                                   │
                                  execute ── [RAG memory: retrieve verified solutions → untrusted prompt context]
                                          │   [recovery: classify → retry w/ jittered backoff → reassign → escalate]
                                       debate ── Developer → Critic → revision → Judge (≤3 rounds, time/token budget)
                                          │
                                         qa ── sandboxed tests decide; pass → save to memory; fail → bounded retry /
                                          │     reassignment → escalate; every outcome → execution_metrics
                                          └──────────────────────────────────────────────────┘
every node: upsert tasks + events + durable checkpoint (DB-outage retries)      worker crash → restore checkpoint → resume at select_task
approval: GitHub publish through the idempotency ledger (never duplicated; unknown outcome → reconcile)
```

| Component | Persistence | Failure behaviour |
|---|---|---|
| RAG memory | `agent_memories`, `memory_retrievals`; FAISS index rebuilt from DB if stale | retrieval errors are logged, coding continues without context |
| Risk predictor | `task_risk_predictions`, `tasks.risk_*`, `execution_metrics` | model missing → workflow runs without risk scores |
| Debate | `agent_debates`, `debate_messages` | LLM error → static critic; budget exceeded → stop with reason |
| Recovery | `workflow_checkpoints`, `recovery_events`, `external_operations` | bounded; escalates to a human instead of looping |

Configuration flags: `MEMORY_ENABLED`, `MEMORY_BACKEND`, `RISK_ENABLED`, `RISK_*_THRESHOLD`, `DEBATE_ENABLED`,
`DEBATE_MAX_ROUNDS`, `DEBATE_TIME_BUDGET_S`, `DEBATE_TOKEN_BUDGET`, `RECOVERY_MODE`, `RETRY_*`, `MAX_WORKFLOW_RESUMES`.
