"""Streamlit dashboard. Run: streamlit run dashboard/streamlit_app.py

Reads live data from the FastAPI backend (API_URL). If the API is unreachable it falls back to
reading the same SQLite database directly ("local mode"). No metric shown here is invented: every
number comes from the database or from report files written by the training scripts.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import settings  # noqa: E402

st.set_page_config(page_title="AegisFlow AI", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem}
div[data-testid="stMetric"]{background:#151a23;border:1px solid #232a36;border-radius:10px;padding:12px 14px}
.badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:.78rem;background:#232a36;color:#c9d1d9}
</style>""", unsafe_allow_html=True)

STATUS_COLORS = {"completed": "#3fb950", "failed": "#f85149", "blocked": "#d29922", "in_progress": "#58a6ff",
                 "assigned": "#8b949e", "qa_failed": "#db6d28", "pending": "#6e7681"}
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10))


# ------------------------------------------------------------------ data access
class Backend:
    """Talks to the FastAPI backend over HTTP; if the API is offline, calls the same app in-process
    (FastAPI TestClient), so every page works identically in both modes."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.mode = "api"
        try:
            httpx.get(f"{self.url}/api/health", timeout=3).raise_for_status()
            self._client = httpx.Client(base_url=self.url, timeout=900)
        except Exception:
            from fastapi.testclient import TestClient

            from app.api.main import app

            self.mode = "local"
            self._client = TestClient(app)

    def _check(self, r):
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(detail)
        return r.json()

    def get(self, path: str, **params):
        return self._check(self._client.get(path, params=params))

    def post(self, path: str, body: dict):
        return self._check(self._client.post(path, json=body))

    def delete(self, path: str):
        return self._check(self._client.delete(path))


@st.cache_resource
def backend() -> Backend:
    return Backend(settings.api_url)


B = backend()

with st.sidebar:
    st.title("🛡️ AegisFlow AI")
    st.caption(f"Data source: **{'FastAPI ' + B.url if B.mode == 'api' else 'local database (API offline)'}**")
    page = st.radio("View", ["Overview", "Workflow Management", "Agent Monitoring", "Dependency Graph",
                             "ML Analytics", "RL Analytics", "Real Datasets", "Agent Memory", "Failure Risk",
                             "Agent Debate", "Self-Healing Monitor", "GitHub"], label_visibility="collapsed")
    if st.button("↻ Refresh", use_container_width=True):
        st.rerun()
    health = B.get("/api/health")
    st.markdown("**Models**")
    for k, v in health["models"].items():
        st.markdown(f"{'🟢' if v else '🔴'} {k}")
    st.markdown(f"**LLM:** {'configured' if health['llm_configured'] else 'template backend'}  \n"
                f"**Scheduler (auto):** {health['scheduler_auto_choice']}")

metrics = B.get("/api/metrics")
runtime, reports = metrics["runtime"], metrics["models"]


def task_df(workflow_id: str | None = None) -> pd.DataFrame:
    rows = B.get("/api/tasks", **({"workflow_id": workflow_id} if workflow_id else {}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pages
def overview():
    st.header("Overview")
    c = st.columns(6)
    c[0].metric("Total tasks", runtime["total_tasks"])
    c[1].metric("Completed", runtime["completed_tasks"])
    c[2].metric("Pending", runtime["pending_tasks"])
    c[3].metric("Failed", runtime["failed_tasks"])
    c[4].metric("Active agents", runtime["active_agents"])
    c[5].metric("Completion", f"{runtime['completion_pct']}%")
    if not runtime["total_tasks"]:
        st.info("No workflows yet — create one under **Workflow Management**.")
        return
    a, b = st.columns(2)
    cat = pd.DataFrame(runtime["by_category"].items(), columns=["category", "tasks"])
    a.plotly_chart(px.bar(cat, x="category", y="tasks", title="Tasks by predicted category").update_layout(**PLOT), use_container_width=True)
    pri = pd.DataFrame(runtime["by_priority"].items(), columns=["priority", "tasks"])
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    pri = pri.sort_values("priority", key=lambda s: s.map(order))
    b.plotly_chart(px.bar(pri, x="priority", y="tasks", title="Tasks by predicted priority",
                          color="priority", color_discrete_sequence=["#f85149", "#db6d28", "#d29922", "#3fb950"]).update_layout(**PLOT, showlegend=False),
                   use_container_width=True)
    st.subheader("Recent workflows")
    wfs = pd.DataFrame(B.get("/api/workflows", limit=20))
    if not wfs.empty:
        wfs["completion %"] = wfs["summary"].map(lambda s: (s or {}).get("completion_pct"))
        st.dataframe(wfs[["workflow_id", "status", "completion %", "requirement", "created_at"]], use_container_width=True, hide_index=True)


def workflow_management():
    st.header("Workflow Management")
    with st.form("new"):
        req = st.text_area("Enterprise requirement", height=100,
                           value="Build a student management system with registration, login, database integration, student dashboard, and automated testing.")
        go_ = st.form_submit_button("🚀 Create workflow", type="primary")
    if go_:
        try:
            res = B.post("/api/workflows", {"requirement": req, "run_async": True})
            st.session_state["wf"] = res["workflow_id"]
            st.success(f"Workflow {res['workflow_id']} started")
        except Exception as e:
            st.error(f"Could not create workflow: {e}")
    wfs = B.get("/api/workflows", limit=50)
    if not wfs:
        return
    ids = [w["workflow_id"] for w in wfs]
    default = ids.index(st.session_state["wf"]) if st.session_state.get("wf") in ids else 0
    wid = st.selectbox("Workflow", ids, index=default, format_func=lambda i: f"{i} — {next(w['status'] for w in wfs if w['workflow_id'] == i)}")
    wf = B.get(f"/api/workflows/{wid}")
    s = wf.get("summary") or {}
    st.markdown(f"**Status:** <span class='badge'>{wf['status']}</span> &nbsp; **Scheduler:** {s.get('scheduler', '-')} &nbsp; "
                f"**Dependencies:** {s.get('dependency_method', '-')} &nbsp; **Code backend:** {s.get('llm_backend', '-')}", unsafe_allow_html=True)
    tasks = pd.DataFrame(wf["tasks"])
    if not tasks.empty:
        done = (tasks["status"] == "completed").mean()
        st.progress(float(done), text=f"{done:.0%} of {len(tasks)} tasks completed")
        view = tasks[["task_id", "title", "category", "priority", "assigned_agent", "status", "attempts", "dependency_ids"]].copy()
        view["dependency_ids"] = view["dependency_ids"].map(lambda d: ", ".join(d or []))
        st.dataframe(view, use_container_width=True, hide_index=True)
        pick = st.selectbox("Inspect task", tasks["task_id"])
        t = tasks.set_index("task_id").loc[pick]
        a, b = st.columns(2)
        a.json({"description": t["description"], "category_confidence": t["category_confidence"],
                "priority_confidence": t["priority_confidence"], "artifacts": t["artifacts"], "error": t["error"]})
        b.json(t["qa_report"] or {})
    if s.get("final_test_suite"):
        st.caption(f"Final sandboxed test suite: {s['final_test_suite'].get('passed')} passed, {s['final_test_suite'].get('failed')} failed")
    if wf["status"] == "running":
        st.info("Workflow running… refreshing in 3 s")
        time.sleep(3)
        st.rerun()
    if wf["status"] == "awaiting_approval":
        st.subheader("Human approval")
        pub = st.checkbox("Also open a GitHub pull request (requires GITHUB_TOKEN, GITHUB_REPO, GITHUB_WRITE_ENABLED=true)")
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve artifacts", type="primary"):
            st.json(B.post(f"/api/workflows/{wid}/approve", {"approved": True, "publish_to_github": pub}))
        if c2.button("❌ Reject"):
            st.json(B.post(f"/api/workflows/{wid}/approve", {"approved": False}))
    with st.expander("Agent message log"):
        ev = pd.DataFrame(wf.get("events", []))
        if not ev.empty:
            st.dataframe(ev[["timestamp", "sender", "recipient", "kind", "content"]], use_container_width=True, hide_index=True)


def agent_monitoring():
    st.header("Agent Monitoring")
    agents = B.get("/api/agents")
    cols = st.columns(len(agents))
    for c, a in zip(cols, agents):
        c.metric(a["name"], a["status"], f"{a['current_workload']} active")
        c.caption(f"{a['role']} · skill {a['skill']}  \n{', '.join(a['capabilities'])}")
    df = pd.DataFrame([{"agent": a["name"], "completed": a["completed"], "failed": a["failed"], "total": a["total_assigned"]} for a in agents])
    st.plotly_chart(px.bar(df.melt("agent", ["completed", "failed"]), x="agent", y="value", color="variable", barmode="stack",
                           title="Execution history per agent", color_discrete_map={"completed": "#3fb950", "failed": "#f85149"}).update_layout(**PLOT),
                    use_container_width=True)
    name = st.selectbox("Agent", [a["name"] for a in agents])
    st.dataframe(pd.DataFrame(next(a for a in agents if a["name"] == name)["recent_tasks"]), use_container_width=True, hide_index=True)


def dependency_graph():
    st.header("Dependency Visualization")
    wfs = B.get("/api/workflows", limit=50)
    if not wfs:
        st.info("No workflows yet.")
        return
    wid = st.selectbox("Workflow", [w["workflow_id"] for w in wfs])
    wf = B.get(f"/api/workflows/{wid}")
    tasks = {t["task_id"]: t for t in wf["tasks"]}
    g = nx.DiGraph()
    g.add_nodes_from(tasks)
    for t in tasks.values():
        g.add_edges_from((d, t["task_id"]) for d in (t["dependency_ids"] or []) if d in tasks)
    if not g.number_of_nodes():
        return
    pos = {}
    for layer, nodes in enumerate(nx.topological_generations(g)):
        for k, n in enumerate(sorted(nodes)):
            pos[n] = (layer, -k + len(nodes) / 2)
    ex, ey = [], []
    for u, v in g.edges:
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure([go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#4b5563", width=1.2), hoverinfo="skip", showlegend=False)])
    for status in sorted({t["status"] for t in tasks.values()}):
        ns = [n for n in g.nodes if tasks[n]["status"] == status]
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers+text", name=status,
                                 text=ns, textposition="top center", marker=dict(size=22, color=STATUS_COLORS.get(status, "#888")),
                                 hovertext=[f"{n}: {tasks[n]['title']}<br>{tasks[n]['category']} · {tasks[n]['priority']} · {tasks[n]['assigned_agent']}" for n in ns],
                                 hoverinfo="text"))
    fig.update_layout(**PLOT, height=520, title=f"Execution DAG ({(wf.get('summary') or {}).get('dependency_method', '')}) — left to right",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)
    order = (wf.get("state") or {}).get("execution_order", [])
    st.markdown("**Execution order:** " + " → ".join(order))
    blocked = [t for t in tasks.values() if t["status"] in ("blocked", "failed")]
    if blocked:
        st.warning("Blocked / failed: " + ", ".join(f"{t['task_id']} ({t['error']})" for t in blocked))
    cands = (wf.get("summary") or {}).get("semantic_candidates", [])
    if cands:
        st.caption("Top semantic-similarity candidates (similar ≠ dependent; shown for transparency)")
        st.dataframe(pd.DataFrame(cands), hide_index=True)


def _cm(m: dict, title: str):
    fig = px.imshow(m["confusion_matrix"], x=m["labels"], y=m["labels"], text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="predicted", y="actual"), title=title)
    return fig.update_layout(**PLOT)


def ml_analytics():
    st.header("Machine Learning Analytics")
    st.caption("All model metrics below were measured on the SYNTHETIC demonstration dataset (held-out projects).")
    for key, title in (("classification_metrics", "Task classification"), ("priority_metrics", "Priority prediction")):
        m = reports.get(key)
        if not m:
            st.warning(f"{title}: no report — run `python scripts/train_all.py`.")
            continue
        st.subheader(title)
        c = st.columns(3)
        c[0].metric("Selected model", m["selected"])
        c[1].metric("Test accuracy", f"{m['test_accuracy']:.3f}")
        c[2].metric("Test macro-F1", f"{m['test_f1_macro']:.3f}")
        a, b = st.columns(2)
        comp = pd.DataFrame([{"model": k, "cv_f1_macro": v["cv_f1_macro"], "val_f1_macro": v["val_f1_macro"]} for k, v in m["candidates"].items()])
        a.plotly_chart(px.bar(comp.melt("model"), x="model", y="value", color="variable", barmode="group",
                              title="GridSearchCV model comparison").update_layout(**PLOT), use_container_width=True)
        b.plotly_chart(_cm(m, "Confusion matrix (test)"), use_container_width=True)
    g = reports.get("gnn_metrics")
    if g:
        st.subheader("Dependency prediction — GNN vs rule baseline")
        if g.get("status") != "trained":
            st.warning(g.get("note"))
        else:
            rows = [{"model": k, **{m: g[k][m] for m in ("roc_auc", "average_precision", "precision", "recall", "f1")}}
                    for k in ("gnn_graphsage", "rule_baseline_networkx")]
            sem = reports.get("semantic_metrics")
            if sem:
                rows.append({"model": "cosine similarity only", "roc_auc": sem["roc_auc"], "average_precision": sem["average_precision"]})
            a, b = st.columns(2)
            a.plotly_chart(px.bar(pd.DataFrame(rows).melt("model"), x="variable", y="value", color="model", barmode="group",
                                  title="Directed edge prediction (test graphs)").update_layout(**PLOT), use_container_width=True)
            h = pd.DataFrame(g["history"])
            b.plotly_chart(px.line(h, x="epoch", y=["train_loss", "val_ap"], title="GNN training history").update_layout(**PLOT), use_container_width=True)


def rl_analytics():
    st.header("Reinforcement Learning Analytics")
    m = reports.get("rl_metrics")
    if not m:
        st.warning("No RL report — run `python -m app.rl.train`.")
        return
    st.caption(f"{m['algorithm']} · {m['timesteps']:,} timesteps · {m['eval_episodes']} evaluation episodes on {m['data']}")
    st.info(m["verdict"])
    ev = pd.DataFrame(m["evaluation"]).T.reset_index(names="scheduler")
    a, b = st.columns(2)
    a.plotly_chart(px.line(pd.DataFrame(m["training_curve"]), x="timesteps", y="mean_episode_reward", title="Training reward curve").update_layout(**PLOT),
                   use_container_width=True)
    b.plotly_chart(px.bar(ev, x="scheduler", y="makespan", error_y="makespan_std", title="Mean makespan (lower is better)").update_layout(**PLOT),
                   use_container_width=True)
    c, d = st.columns(2)
    c.plotly_chart(px.bar(ev, x="scheduler", y="load_imbalance_cv", title="Load imbalance (CV of busy time, lower is better)").update_layout(**PLOT),
                   use_container_width=True)
    d.plotly_chart(px.bar(ev, x="scheduler", y="success_rate", title="Task success rate").update_layout(**PLOT, yaxis_range=[0.8, 1]),
                   use_container_width=True)
    st.subheader("Live workload distribution")
    st.plotly_chart(px.pie(names=list(runtime["tasks_per_agent"]), values=list(runtime["tasks_per_agent"].values()), hole=0.5,
                           title="Tasks assigned per agent (all workflows)").update_layout(**PLOT), use_container_width=True)


def github_page():
    st.header("GitHub Integration")
    gh = B.get("/api/github")
    if not gh.get("available"):
        st.warning(f"GitHub not available: {gh.get('reason')}")
        st.code("GITHUB_TOKEN=...\nGITHUB_REPO=owner/name\nGITHUB_WRITE_ENABLED=false   # true only to allow approved PR creation", language="bash")
        return
    r = gh["repository"]
    c = st.columns(4)
    c[0].metric("Repository", r["full_name"])
    c[1].metric("Stars", r["stargazers_count"])
    c[2].metric("Open issues", r["open_issues_count"])
    c[3].metric("Default branch", r["default_branch"])
    a, b = st.columns(2)
    a.subheader("Open issues")
    a.dataframe(pd.DataFrame(gh["issues"]), hide_index=True, use_container_width=True)
    b.subheader("Open pull requests")
    b.dataframe(pd.DataFrame(gh["pull_requests"]), hide_index=True, use_container_width=True)
    commits = pd.DataFrame(gh["recent_commits"])
    if not commits.empty:
        st.plotly_chart(px.bar(commits["author"].value_counts().reset_index(), x="author", y="count",
                               title="Recent commit activity (reviewer recommendation signal)").update_layout(**PLOT), use_container_width=True)


def real_datasets():
    st.header("Real Datasets")
    st.caption("Models on this page are trained and evaluated on REAL labelled data: the public PROMISE NFR requirements "
               "dataset and a 4,000-ticket multilingual (en/de/es/fr/pt) customer-support dataset.")
    ds = B.get("/api/datasets")
    tab_req, tab_tix = st.tabs(["📄 Requirements classifier (PROMISE)", "🎫 Ticket triage (multilingual)"])

    with tab_req:
        rm = ds.get("requirements_metrics")
        if not rm:
            st.warning("Not trained — run `python main.py train-real`.")
        else:
            b, mc = rm["binary_functional_vs_nfr"], rm["multiclass_requirement_type"]
            c = st.columns(4)
            c[0].metric("Train / test statements", f"{b['n_train']} / {b['n_test']}")
            c[1].metric("Functional vs NFR accuracy", f"{b['test_accuracy']:.3f}", f"baseline {b['majority_baseline_accuracy']:.2f}", delta_color="off")
            c[2].metric("12-type accuracy", f"{mc['test_accuracy']:.3f}", f"baseline {mc['majority_baseline_accuracy']:.2f}", delta_color="off")
            c[3].metric("12-type macro-F1", f"{mc['test_f1_macro']:.3f}")
            a, bb = st.columns(2)
            counts = pd.DataFrame([{"type": rm["label_names"][k], "train": v} for k, v in rm["train_label_counts"].items()]).sort_values("train")
            a.plotly_chart(px.bar(counts, x="train", y="type", orientation="h", title="Training statements per requirement type").update_layout(**PLOT),
                           use_container_width=True)
            bb.plotly_chart(_cm(mc, "Requirement type — confusion matrix (PROMISE test file)"), use_container_width=True)
        st.subheader("Try it: paste a requirements document")
        txt = st.text_area("One requirement per line", height=150, value=(
            "The system shall allow staff to add a new student to a cohort.\n"
            "Only users with Update access must be able to initiate dispute requests.\n"
            "The system shall refresh the display every 60 seconds.\n"
            "The application must conform to Regulation E for disputes processing.\n"
            "90% of new users shall be able to create a case after a 2-day training course."))
        c1, c2 = st.columns(2)
        if c1.button("Classify requirements"):
            try:
                res = B.post("/api/requirements/classify", {"text": txt})
                st.dataframe(pd.DataFrame(res["statements"])[["label_name", "confidence", "text"]], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(str(e))
        if c2.button("🚀 Create workflow from this document", type="primary"):
            try:
                res = B.post("/api/workflows", {"requirement": txt, "run_async": True})
                st.session_state["wf"] = res["workflow_id"]
                st.success(f"Workflow {res['workflow_id']} started — open Workflow Management to follow it.")
            except Exception as e:
                st.error(str(e))

    with tab_tix:
        tm = ds.get("ticket_metrics")
        if not tm:
            st.warning("Not trained — run `python main.py train-real`.")
        else:
            c = st.columns(4)
            c[0].metric("Tickets", tm["rows"])
            for col, t in zip(c[1:], ("type", "priority", "queue")):
                x = tm["targets"][t]
                col.metric(f"{t.title()} accuracy", f"{x['test_accuracy']:.3f}",
                           f"short tickets {x.get('short_text_test_accuracy', 0):.2f} · baseline {x['majority_baseline_accuracy']:.2f}", delta_color="off")
            lang = pd.DataFrame([{"language": lg, "target": t, "accuracy": acc} for t, x in tm["targets"].items()
                                 for lg, acc in x["accuracy_by_language"].items()])
            a, bb = st.columns(2)
            a.plotly_chart(px.bar(lang, x="language", y="accuracy", color="target", barmode="group",
                                  title="Hold-out accuracy by language").update_layout(**PLOT), use_container_width=True)
            pick = bb.selectbox("Confusion matrix", ["type", "priority", "queue"])
            bb.plotly_chart(_cm(tm["targets"][pick], f"Ticket {pick} — confusion matrix (hold-out)"), use_container_width=True)
        st.subheader("Try it: triage a ticket")
        subj = st.text_input("Subject", "Payment gateway down for all customers")
        body = st.text_area("Body", "Since this morning every checkout fails with a timeout at the payment step. "
                                    "Customers cannot complete orders. Please fix urgently.", height=110)
        lang_in = st.selectbox("Language", ["en", "de", "es", "fr", "pt"])
        mk = st.checkbox("Also create an engineering workflow from this ticket")
        if st.button("🎫 Triage ticket", type="primary"):
            try:
                res = B.post("/api/tickets/triage", {"subject": subj, "body": body, "language": lang_in, "create_workflow": mk})
                t = res["triage"]
                c = st.columns(4)
                c[0].metric("Type", t["type"], f"{t['type_confidence']:.0%}", delta_color="off")
                c[1].metric("Priority", t["priority"], f"{t['priority_confidence']:.0%}", delta_color="off")
                c[2].metric("Queue", t["queue"], f"{t['queue_confidence']:.0%}", delta_color="off")
                c[3].metric("Owner role", t["owner_role"])
                if res.get("workflow_id"):
                    st.session_state["wf"] = res["workflow_id"]
                    st.success(f"Workflow {res['workflow_id']} created — open Workflow Management to follow it.")
            except Exception as e:
                st.error(str(e))


def agent_memory():
    st.header("Agent Memory (RAG)")
    st.caption("Only solutions that passed sandboxed QA (or human approval) are stored. Retrieved text is injected into the "
               "Coding Agent prompt as UNTRUSTED context.")
    try:
        data = B.get("/api/memory", limit=300)
    except Exception as e:
        st.error(str(e))
        return
    mems = pd.DataFrame(data["memories"])
    c = st.columns(3)
    c[0].metric("Stored memories", len(mems))
    c[1].metric("Index backend", data["backend"].split("+")[0])
    hist = pd.DataFrame(B.get("/api/memory/retrievals", limit=500))
    c[2].metric("Retrievals logged", len(hist))
    st.subheader("Similarity search")
    q = st.text_input("Describe a task", "Build REST API endpoints for user login")
    cat = st.selectbox("Category filter", ["(any)", "development", "testing"])
    if st.button("🔎 Search memory"):
        res = B.post("/api/memory/search", {"query": q, "task_category": None if cat == "(any)" else cat, "top_k": 5})
        if res:
            st.dataframe(pd.DataFrame(res)[["rank", "score", "task_category", "task_description", "solution_summary", "verified_outcome"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No memory above the similarity threshold.")
    st.subheader("Stored memories")
    if mems.empty:
        st.info("No memories yet: run a workflow; tasks that pass QA are saved automatically.")
    else:
        st.dataframe(mems[["memory_id", "task_category", "task_description", "solution_summary", "verified_outcome", "agent_id",
                           "retrieval_count", "created_at"]], use_container_width=True, hide_index=True)
    if not hist.empty:
        st.subheader("Retrieval history")
        st.plotly_chart(px.histogram(hist, x="score", nbins=20, title="Retrieved similarity scores").update_layout(**PLOT), use_container_width=True)
        st.dataframe(hist, use_container_width=True, hide_index=True)


def failure_risk():
    st.header("Failure Risk Dashboard")
    try:
        m = B.get("/api/risk/metrics")
    except Exception as e:
        st.warning(f"Model metrics unavailable: {e}")
        m = None
    if m:
        st.caption(f"Training data: **{m['data']}**, {m['rows']:,} task executions in {m['workflows']} workflows, {m['split']}. "
                   "No provided real dataset contains agent execution outcomes, so these numbers describe the simulator, not production.")
        cols = st.columns(2)
        for col, (tname, x) in zip(cols, m["targets"].items()):
            cm = x["test_calibrated"]
            col.markdown(f"**{tname.replace('_', ' ')}** — {x['selected']}")
            k = col.columns(4)
            k[0].metric("PR-AUC", f"{cm['pr_auc']:.3f}", f"dummy {x['test_dummy']['pr_auc']:.2f}", delta_color="off")
            k[1].metric("Recall", f"{cm['recall']:.2f}")
            k[2].metric("Precision", f"{cm['precision']:.2f}")
            k[3].metric("Brier", f"{cm['brier']:.3f}", f"dummy {x['test_dummy']['brier']:.3f}", delta_color="off")
            comp = pd.DataFrame([{"model": n, "val PR-AUC": v["val_pr_auc"]} for n, v in x["candidates"].items()])
            col.plotly_chart(px.bar(comp, x="model", y="val PR-AUC", title="Model comparison (validation workflows)").update_layout(**PLOT),
                             use_container_width=True)
            col.plotly_chart(px.imshow(cm["confusion_matrix"], x=["pred 0", "pred 1"], y=["actual 0", "actual 1"], text_auto=True,
                                       color_continuous_scale="Blues", title="Confusion matrix (test)").update_layout(**PLOT),
                             use_container_width=True)
    preds = pd.DataFrame(B.get("/api/risk/predictions", limit=1000))
    st.subheader("Live task risk predictions")
    if preds.empty:
        st.info("No predictions yet — create a workflow.")
    else:
        c = st.columns(3)
        c[0].metric("Tasks assessed", len(preds))
        c[1].metric("High risk", int((preds.risk_category == "High").sum()))
        c[2].metric("Medium risk", int((preds.risk_category == "Medium").sum()))
        high = preds[preds.risk_category.isin(["High", "Medium"])]
        st.markdown("**High / medium-risk tasks**")
        st.dataframe((high if not high.empty else preds)[["workflow_id", "task_id", "category", "priority", "agent_id", "failure_probability",
                                                           "deadline_miss_probability", "risk_category", "explanation"]],
                     use_container_width=True, hide_index=True)
    st.subheader("What-if: predict a task's risk")
    c = st.columns(4)
    rec = {"task_category": c[0].selectbox("Category", ["development", "testing", "documentation", "deployment", "analysis"]),
           "task_priority": c[1].selectbox("Priority", ["low", "medium", "high", "critical"], index=2),
           "agent_id": c[2].selectbox("Agent", [a["name"] for a in settings.agent_pool]),
           "task_complexity": c[3].slider("Complexity", 1, 5, 4)}
    c = st.columns(4)
    rec.update({"dependency_count": c[0].slider("Dependencies", 0, 8, 2), "current_workload": c[1].slider("Agent workload", 0, 10, 3),
                "estimated_duration": c[2].slider("Estimated hours", 1.0, 20.0, 10.0), "deadline_slack": c[3].slider("Deadline slack", 0.5, 3.0, 1.2)})
    if st.button("Predict risk"):
        r = B.post("/api/risk/predict", rec)
        for tname in ("task_failed", "deadline_missed"):
            e = r[tname]
            st.markdown(f"**{tname.replace('_', ' ')}:** {e['explanation']}  _(explainer: {e['method']}, model: {e['model']})_")
            f = pd.DataFrame(e["factors"])
            st.plotly_chart(px.bar(f, x="contribution", y="feature", orientation="h", title=f"Main risk factors — {tname}").update_layout(**PLOT),
                            use_container_width=True)


def agent_debate():
    st.header("AI Agent Debate")
    st.caption("Developer proposal → Critic review → Developer revision → Judge → QA. The Judge never replaces the sandboxed QA tests.")
    deb = pd.DataFrame(B.get("/api/debates", limit=300))
    if deb.empty:
        st.info("No debates yet — create a workflow (DEBATE_ENABLED=true).")
        return
    c = st.columns(5)
    c[0].metric("Debates", len(deb))
    c[1].metric("Accepted by judge", int((deb.judge_verdict == "accept").sum()))
    c[2].metric("QA pass rate", f"{deb.qa_passed.dropna().mean():.0%}" if deb.qa_passed.notna().any() else "-")
    c[3].metric("Avg revisions", f"{deb.revisions.mean():.2f}")
    c[4].metric("Tokens used", int(deb.tokens_used.sum()))
    st.dataframe(deb[["debate_id", "workflow_id", "task_id", "judge_verdict", "qa_passed", "rounds", "revisions", "stop_reason",
                      "tokens_used", "estimated_cost_usd", "latency_ms", "backend"]], use_container_width=True, hide_index=True)
    pick = st.selectbox("Open discussion", deb.debate_id)
    d = B.get(f"/api/debates/{pick}")
    st.markdown(f"**Stop reason:** {d['stop_reason']} · **Judge:** {d['judge_verdict']} · **QA:** {d['qa_passed']} · **Backend:** {d['backend']}")
    avatars = {"developer": "🧑‍💻", "critic": "🔍", "judge": "⚖️", "qa": "🧪", "system": "📋"}
    for m in d["messages"]:
        with st.chat_message(m["role"] if m["role"] in ("user", "assistant") else "assistant", avatar=avatars.get(m["role"], "🤖")):
            st.markdown(f"**{m['role'].title()}** · {m['kind']} · round {m['round']} · {m['tokens']} tokens")
            st.write(m["content"])
            if m["issues"]:
                st.dataframe(pd.DataFrame(m["issues"]), hide_index=True, use_container_width=True)


def self_healing():
    st.header("Self-Healing Monitor")
    st.caption("Faults below are SIMULATED via controlled fault injection and labelled as such in every event.")
    ev = pd.DataFrame(B.get("/api/recovery/events", limit=2000))
    if not ev.empty:
        c = st.columns(5)
        c[0].metric("Failures detected", int((ev.action == "detected").sum()))
        c[1].metric("Recovered", int((ev.action == "recovered").sum()))
        c[2].metric("Retries", int((ev.action == "retry").sum()))
        c[3].metric("Reassignments", int((ev.action == "reassign").sum()))
        c[4].metric("Escalated to human", int((ev.action == "escalate").sum()))
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(ev, x="failure_type", color="action", barmode="group", title="Recovery actions by failure type").update_layout(**PLOT),
                       use_container_width=True)
        rec = ev[(ev.action == "recovered") & ev.duration_ms.notna()]
        if not rec.empty:
            b.plotly_chart(px.box(rec, x="failure_type", y="duration_ms", title="Recovery duration (ms)").update_layout(**PLOT), use_container_width=True)
        st.dataframe(ev[["created_at", "workflow_id", "task_id", "agent_id", "failure_type", "action", "attempt", "status", "duration_ms",
                         "simulated", "detail"]], use_container_width=True, hide_index=True)
    else:
        st.info("No recovery events yet.")
    st.subheader("Fault injection demo")
    sc = st.selectbox("Scenario", ["api_timeout", "rate_limit", "network_error", "agent_crash", "db_outage", "qa_rejection", "repeated_failure", "worker_crash"])
    st.caption({"api_timeout": "LLM call times out twice, then succeeds", "rate_limit": "HTTP 429 twice (honours Retry-After)",
                "network_error": "connection reset once", "agent_crash": "coder-1 is down for the whole run → tasks are reassigned",
                "db_outage": "database writes fail twice → persistence is retried", "qa_rejection": "QA rejects once (simulated)",
                "repeated_failure": "task T04 fails every time → retries exhausted → human escalation, dependants blocked",
                "worker_crash": "orchestrator crashes once at the QA step → resume from the latest durable checkpoint"}[sc])
    if st.button("💥 Run workflow with this fault", type="primary"):
        with st.spinner("Running workflow with injected fault…"):
            r = B.post("/api/recovery/demo", {"scenario": sc})
        st.success(f"Workflow {r['workflow_id']} → {r['workflow_status']} (faults fired: {r['faults_fired']})")
        st.dataframe(pd.DataFrame(r["recovery_events"]), use_container_width=True, hide_index=True)


{"Overview": overview, "Workflow Management": workflow_management, "Agent Monitoring": agent_monitoring,
 "Dependency Graph": dependency_graph, "ML Analytics": ml_analytics, "RL Analytics": rl_analytics, "Real Datasets": real_datasets,
 "Agent Memory": agent_memory, "Failure Risk": failure_risk, "Agent Debate": agent_debate, "Self-Healing Monitor": self_healing,
 "GitHub": github_page}[page]()
