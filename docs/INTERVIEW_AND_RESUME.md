# Interview explanation (2–3 minutes)

**The problem.** "Big software requests like 'build a student management system with login, a database and tests'
have to be broken down, ordered, prioritised and handed to the right people before any work starts. I built a system
where AI agents do that coordination themselves and then actually do and verify the work."

**How it works.** "It's a LangGraph state machine with eight agents. A scoping agent turns the requirement into tasks
with complexity, urgency and acceptance criteria. A classifier labels each task as development, testing, docs,
deployment or analysis. Then the planning agent works out what depends on what. I deliberately didn't use plain text
similarity for that: I measured it, and similarity has an average precision of only 0.14, because 'test login' and
'test billing' look similar but are independent. Instead, a GraphSAGE graph neural network in PyTorch Geometric scores
every ordered pair of tasks, and I insert edges by confidence while rejecting anything that would create a cycle, so
the plan is always a valid DAG. A priority model uses the dependency fan-out as a feature, then a workload agent
assigns tasks to agents with the right capabilities. The coding agent writes modules and unit tests into an isolated
workspace, and the QA agent scans them for dangerous calls, lints them and runs the tests in a resource-limited
sandbox. Failures go back to the coder with feedback, with a retry limit. Nothing touches GitHub until a human
approves, and even then it only opens a pull request, never merges."

**What I learned.** "Two honest results. First, I trained a PPO reinforcement-learning scheduler. It gave the most
balanced workload but didn't beat a simple least-loaded heuristic on total completion time, so the system picks the
heuristic automatically from the saved evaluation. I'd rather ship the better policy than the fancier one. Second,
my data was synthetic. When I ran unseen requirements, the GNN missed some prerequisites. That was distribution
shift, so I added structural variety to the generator and the plans became correct. All metrics are labelled
synthetic in the dashboard and the reports."

**Real data.** "I also trained two models on real labelled data. One classifies requirements as functional or one of 11
non-functional types, using the PROMISE dataset; the scoping agent uses it to read a whole requirements document. The
other triages multilingual support tickets in English, German, Spanish, French and Portuguese, and turns tickets into
workflows. An ablation showed that one input field biased the ticket model and that one-line tickets were handled badly,
so I removed the field and trained on subject-only copies. Short-ticket priority accuracy rose from 53% to 62%."

**Engineering.** "FastAPI backend, Streamlit dashboard, SQLAlchemy persistence of every agent message, 48 tests
including failure recovery and mocked GitHub rate limits, and an LLM is optional: with no API key, everything
falls back to deterministic backends, so the demo always runs."

---

# Resume entry

**Autonomous AI Multi-Agent Workflow Orchestrator** | *LangGraph, PyTorch Geometric, Stable-Baselines3, scikit-learn, FastAPI, Streamlit*

- Built an 8-agent LangGraph system that turns natural-language requirements into a scoped, prioritised, dependency-ordered task plan, then generates code and unit tests and verifies them in a resource-limited sandbox, with bounded QA retry loops and human-in-the-loop approval.
- Trained a GraphSAGE link-prediction GNN for directed task dependencies (average precision 0.99 vs 0.13 for a rule baseline on held-out synthetic projects); showed that semantic similarity alone is a weak dependency signal (AP 0.14).
- Tuned task-category (macro-F1 0.97) and priority (macro-F1 0.76) classifiers with GridSearchCV and project-grouped cross-validation to prevent leakage.
- Built a Gymnasium scheduling environment and PPO agent, benchmarked it against round-robin and least-loaded over 200 episodes, and kept the heuristic in auto mode when it won on makespan.
- Trained real-data models: a requirement-type classifier on the PROMISE NFR dataset (functional-vs-NFR accuracy 0.79 cross-project vs a 0.36 baseline) and multilingual ticket triage on 4,000 tickets in 5 languages (priority accuracy 0.63 vs 0.41); an ablation raised one-line-ticket priority accuracy from 0.53 to 0.62.
- Shipped a FastAPI backend (16 endpoints), an 8-view Streamlit dashboard, an approval-gated GitHub integration (pagination, rate limits, reviewer recommendation) and 57 automated tests.

**Short version (one line):** Multi-agent LangGraph orchestrator with a GNN dependency planner, ML task triage, real-data requirement classification and multilingual ticket triage, a PPO-vs-heuristic scheduler, sandboxed code QA, FastAPI and Streamlit; 57 tests.


---

# AegisFlow AI (v2) — additions

**Interview add-on (about 45 seconds).** "Version 2 makes the orchestrator self-healing and self-improving. There are four additions.
First, RAG memory: after QA verifies a solution, it goes into a FAISS index, and similar past solutions are retrieved into
the coding prompt. The retrieved text is marked untrusted, secrets are redacted, and duplicates are rejected. Second, a calibrated
failure-risk model with SHAP explanations. Third, a bounded Developer–Critic–Judge debate before QA. On a planted-defect benchmark,
it cut the defects that slip past unit tests from 15 to 0. Fourth,
self-healing recovery: classified retries with jittered backoff, durable checkpoints, agent reassignment, and an idempotency
ledger so a lost-response pull request is never created twice. With injected faults, recovery success went from 14%
with no recovery to 100%. I measured the limitations too: TF-IDF memory collapses on paraphrased queries, which
justifies embeddings, and the risk model is trained on synthetic execution history until enough live runs exist."

**Resume bullets (AegisFlow v2).**
- Extended a LangGraph multi-agent orchestrator with self-healing recovery: classified retries, exponential backoff with jitter, durable DB checkpoints with crash resume, safe agent reassignment, idempotent external writes and human escalation. Across 7 injected fault types, recovery success rose from 14% to 100% and lost work fell from 42 to 2 tasks.
- Built RAG agent memory (FAISS, Sentence-Transformers with a TF-IDF fallback) that stores only QA-verified solutions. A labelled benchmark (Recall@5 1.0 lexical vs 0.0821 paraphrased for TF-IDF) quantified the need for semantic embeddings.
- Added a bounded Developer–Critic–Judge debate engine (Pydantic messages, round, time and token budgets, LLM fallback). It reduced defects escaping unit tests from 15 to 0 on a 32-run benchmark.
- Trained calibrated task-failure and deadline-risk models (Logistic Regression, RF, XGBoost with GridSearchCV, workflow-grouped splits and SHAP explanations): deadline-miss PR-AUC 0.7846 vs a 0.3305 baseline, on clearly labelled synthetic execution data.
