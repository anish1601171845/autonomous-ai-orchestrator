# Example workflow: all four AegisFlow enhancements in one run

These are real outputs from two consecutive runs of the same requirement. Faults in run 2 are **simulated** with the
fault injector. Reproduce it with the dashboard's **Self-Healing Monitor**, or with the API:
`POST /api/workflows` and then `POST /api/recovery/demo {"scenario": "agent_crash"}`.

**Requirement:** *"Build a notes app with login, note search and automated testing."*

## Run 1: normal execution (memory empty)

```
status: awaiting_approval   tasks: 8 completed
enhancements: memory_hits 2, memories_saved 6, debates 3, high_risk_tasks 0, recovery_mode full

T01 Project setup          development  medium  coder-1  risk=Low     p_fail=0.08  p_late=0.08
T02 Requirements analysis  analysis     medium  qa-1     risk=Low     p_fail=0.13  p_late=0.11
T03 Implement login        development  high    coder-1  risk=Medium  p_fail=0.20  p_late=0.13
T04 Test login             testing      high    qa-1     risk=Low     p_fail=0.14  p_late=0.12
T05 Implement note search  development  high    coder-2  risk=Medium  p_fail=0.21  p_late=0.16
T06 Test note search       testing      medium  coder-1  risk=Medium  p_fail=0.24  p_late=0.14
T07 Integration tests      testing      medium  qa-1     risk=Medium  p_fail=0.18  p_late=0.14
T08 System documentation   documentation low    docs-1   risk=Low     p_fail=0.10  p_late=0.13
```

- **Risk explanation (T04):** *"Low risk (14% calibrated probability of failure). Raised by priority (high), task
  category (testing). Lowered by assigned agent (qa-1), agent's current workload (1)."*
- **Debate (T05):**
  1. The Developer proposes `note_search.py`.
  2. The Critic finds 1 major issue (`Search` is never referenced by tests).
  3. The Developer adds contract tests.
  4. The Critic re-checks and finds 0 major issues.
  5. The Judge accepts.
  6. QA passes: 8 passed, 0 failed.
- **Memory:** 6 QA-verified solutions are saved after their tests pass.

## Run 2: same requirement, `coder-1` down, and the orchestrator crashes once at the QA step

```
status: awaiting_approval   tasks: 8 completed
enhancements: memory_hits 5, memories_saved 0 (all duplicates of run 1), reassigned_tasks [T01, T03, T06],
              restored_from_checkpoint {seq: 8, node: debate}
```

**Memory feeds the Coding Agent:**

```
T01 -> mem_a8c3… (score 0.76)
T03 -> mem_377d… (0.52), mem_2ac4… (0.47)
T05 -> mem_2ac4… (0.59), mem_377d… (0.43)
```

**Recovery trail** (from `recovery_events`):

```
T01 coder-1 agent_unavailable  detected            failed
T01 coder-2 agent_unavailable  reassign            success  {from: coder-1, to: coder-2}
T01 coder-2 agent_unavailable  retry → recovered   success
--  worker_crash               detected            failed
--  worker_crash               checkpoint_restore  success  {restored_from: seq 8 (debate), resume_entry: select_task, requeued: [T01]}
T01 coder-2                    recovered           success  (re-executed after restore; workspace writes are idempotent)
T03 coder-1 → coder-2          reassign → retry → recovered
T06 coder-1 → qa-1             reassign → retry → recovered
```

All 8 tasks completed despite a dead agent and a crashed orchestrator. Only the one in-flight task (T01) was
re-executed, and no completed work was lost.
