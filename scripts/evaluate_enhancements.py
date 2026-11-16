"""Reproducible baseline experiments for the four AegisFlow enhancements.

    python scripts/evaluate_enhancements.py                 # everything (~5-10 min on a laptop CPU)
    python scripts/evaluate_enhancements.py --only memory   # memory | risk | debate | recovery

Runs in an ISOLATED temporary database/workspace (your dashboard data is untouched), except that the
trained failure model is saved to models/ for the app to use. Writes:
    reports/enhancement_evaluation.json, reports/enhancement_evaluation.md, reports/figures/enh_*.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="aegis_eval_"))
os.environ.setdefault("EVAL_KEEP_DB", "0")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'eval.db'}"
os.environ["WORKSPACE_DIR"] = str(_TMP / "workspace")
os.environ["MEMORY_INDEX_DIR"] = str(_TMP / "memory_index")
os.environ["RETRY_BASE_DELAY_S"] = "0.02"
os.environ["RETRY_MAX_DELAY_S"] = "0.2"
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.core.config import settings  # noqa: E402

FIG = settings.reports_dir / "figures"
OUT_JSON = settings.reports_dir / "enhancement_evaluation.json"
OUT_MD = settings.reports_dir / "enhancement_evaluation.md"


def _set(**kw):
    for k, v in kw.items():
        object.__setattr__(settings, k, v)


# ------------------------------------------------------------------ 1. memory
def eval_memory() -> dict:
    from app.memory.evaluation import evaluate

    rows = []
    for qs in (False, True):
        rows.append(evaluate("none", paraphrase=qs))
        rows.append(evaluate("tfidf", paraphrase=qs))
        try:
            rows.append(evaluate("embedding", paraphrase=qs))
        except Exception as e:
            rows.append({"backend": "sentence-embeddings (unavailable)", "query_set": "paraphrase" if qs else "lexical",
                         "error": f"{type(e).__name__}: model could not be loaded in this environment"})
    # end-to-end: does memory change task success? (same requirements, memory off vs on)
    from app.core import orchestrator
    from app.memory import rag_memory

    reqs = ["Build a notes app with login, note search and automated testing.",
            "Create an inventory tracking tool with product catalog, stock alerts and a reporting dashboard."]
    e2e = {}
    for enabled in (False, True):
        _set(memory_enabled=enabled, debate_enabled=False, risk_enabled=False)
        rag_memory.reset_service()
        first_pass, total, hits, tokens = 0, 0, 0, 0
        for r in reqs * 2:  # second pass can reuse memories saved by the first
            s = orchestrator.run_workflow(r)
            for t in s["tasks"]:
                if t["category"] == "development":
                    total += 1
                    first_pass += t["status"] == "completed" and t.get("attempts", 1) == 1
                    hits += len(t.get("memory_hits", []))
            ctx = [rag_memory.build_memory_context(rag_memory.retrieve_similar_solutions(t["description"], task_category="development", record=False))
                   for t in s["tasks"] if t["category"] == "development"] if enabled else []
            tokens += sum(len(c) // 4 for c in ctx)
        e2e["memory_on" if enabled else "memory_off"] = {"dev_tasks": total, "first_attempt_success_rate": round(first_pass / max(1, total), 4),
                                                        "memory_hits": hits, "avg_context_tokens_per_dev_task": round(tokens / max(1, total), 1)}
    _set(memory_enabled=True, debate_enabled=True, risk_enabled=True)
    e2e["note"] = ("With the deterministic template backend the generated code does not depend on prompt context, so memory "
                   "cannot change task success here; it adds prompt tokens. Its effect on success must be measured with an LLM backend.")
    return {"retrieval": rows, "end_to_end": e2e}


# ------------------------------------------------------------------ 2. risk
def eval_risk() -> dict:
    from app.ml import failure_predictor as fp

    return fp.train_failure_model(fp.prepare_failure_dataset())


# ------------------------------------------------------------------ 3. debate
DEFECTS = {
    "none": lambda s: s,
    "untested_placeholder": lambda s: s + "\n\ndef export_records():\n    pass\n",
    "removed_validation": lambda s: s.replace("raise ValidationError", "return None  # ").replace("raise AuthError", "return None  # ")
                                     .replace("raise ValueError", "return None  # "),
    "unsafe_import": lambda s: "import subprocess\n" + s,
}


def eval_debate() -> dict:
    from app.agents import coding_agent, debate_engine, qa_agent, scoping_agent
    from app.agents.debate_engine import static_review
    from app.database import repository as repo

    reqs = ["Build a notes app with login, note search and automated testing.",
            "Create an HR portal with employee onboarding, leave management and an analytics dashboard.",
            "Build an e-commerce store with product catalog, shopping cart and order management."]
    tasks = []
    for r in reqs:
        _, ts = scoping_agent.run(r)
        tasks += [t.model_dump() for t in ts if t.title.startswith("Implement")]
    results = {"no_debate": [], "debate": []}
    for cond in results:
        for i, t in enumerate(tasks):
            for dname, defect in DEFECTS.items():
                wid = f"wf_deb_{cond}_{i}_{dname}"
                repo.create_workflow(wid, "debate benchmark")
                out = coding_agent.generate_code(wid, t)
                mod = coding_agent.workspace(wid) / f"{out['module']}.py"
                mod.write_text(defect(mod.read_text(encoding="utf-8")), encoding="utf-8")
                t0 = time.perf_counter()
                rev = tok = 0
                if cond == "debate":
                    res = debate_engine.DebateEngine(wid, t).run()
                    rev, tok = res.revisions, res.tokens_used
                qa = qa_agent.review(out["files"])
                src = mod.read_text(encoding="utf-8")
                tst = (mod.parent / f"test_{out['module']}.py").read_text(encoding="utf-8")
                remaining = [i.code for i in static_review(src, tst, t.get("acceptance_criteria", [])) if i.severity != "minor"]
                results[cond].append({"defect": dname, "qa_passed": qa["passed"], "revisions": rev, "tokens": tok,
                                      "latency_ms": (time.perf_counter() - t0) * 1000, "issues_remaining": len(remaining)})
    summary = {}
    for cond, rows in results.items():
        summary[cond] = {
            "runs": len(rows), "qa_pass_rate": round(np.mean([r["qa_passed"] for r in rows]), 4),
            "task_completion_rate": round(np.mean([r["qa_passed"] and r["issues_remaining"] == 0 for r in rows]), 4),
            "defects_escaping_qa": int(sum(r["qa_passed"] and r["issues_remaining"] > 0 for r in rows)),
            "mean_revisions": round(np.mean([r["revisions"] for r in rows]), 3),
            "mean_latency_ms": round(np.mean([r["latency_ms"] for r in rows]), 1),
            "mean_tokens": round(np.mean([r["tokens"] for r in rows]), 1),
            "estimated_api_cost_usd": 0.0 if settings.llm_api_key == "" else round(np.sum([r["tokens"] for r in rows]) / 1000 * settings.llm_cost_per_1k_tokens, 4),
            "by_defect": {d: {"qa_pass_rate": round(np.mean([r["qa_passed"] for r in rows if r["defect"] == d]), 3),
                              "clean_rate": round(np.mean([r["qa_passed"] and r["issues_remaining"] == 0 for r in rows if r["defect"] == d]), 3)}
                          for d in DEFECTS}}
    summary["benchmark"] = f"{len(tasks)} implementation tasks x {len(DEFECTS)} planted-defect variants (incl. none), same inputs in both conditions"
    summary["backend"] = "llm" if settings.llm_api_key else "static critic + template developer (no API cost)"
    summary["task_completion_definition"] = "QA tests pass AND no blocking/major static-review issue remains"
    return summary


# ------------------------------------------------------------------ 4. recovery
SCENARIOS = {"api_timeout": {}, "rate_limit": {}, "agent_crash": {"agent": "coder-1"}, "db_outage": {},
             "qa_rejection": {"times": 3}, "repeated_failure": {"task_id": "T04"}, "worker_crash": {"node": "qa"}}


def _duplicate_write_trial(mode: str) -> dict:
    """External write whose request succeeds but whose response is lost (classic duplicate-PR hazard)."""
    from app.recovery.recovery_controller import NeedsReconciliation, RecoveryController, RecoveryExhausted

    writes = {"n": 0}

    def create_pr():
        writes["n"] += 1
        if writes["n"] == 1:
            raise ConnectionError("connection reset after the PR was created")
        return {"pr": writes["n"]}

    ctl = RecoveryController(mode, sleep=lambda s: None)
    outcome = "ok"
    try:
        if mode in ("checkpoint", "full"):
            try:
                ctl.run_external_once(f"bench:{mode}:{time.time_ns()}", "github_pr", create_pr)
            except ConnectionError:
                outcome = "escalated_for_reconciliation"
        else:
            ctl.retry_task(create_pr, workflow_id=None, operation="github_pr")
    except (RecoveryExhausted, NeedsReconciliation):
        outcome = "failed"
    return {"writes": writes["n"], "duplicates": max(0, writes["n"] - 1), "outcome": outcome}


def eval_recovery() -> dict:
    from app.core import orchestrator
    from app.recovery import fault_injection as fi
    from app.recovery.recovery_controller import list_recovery_events

    _set(debate_enabled=False, memory_enabled=False, risk_enabled=False)
    req = "Build a notes app with login, note search and automated testing."
    base_t0 = time.perf_counter()
    orchestrator.run_workflow(req)
    baseline_s = time.perf_counter() - base_t0
    table = {}
    for mode in ("none", "fixed_retry", "checkpoint", "full"):
        _set(recovery_mode=mode)
        rows = []
        for sc, kw in SCENARIOS.items():
            t0 = time.perf_counter()
            wid = orchestrator.new_workflow_id()
            with fi.inject(sc, **kw):
                try:
                    s = orchestrator.run_workflow(req, wid)
                    tasks = s["tasks"]
                except Exception:
                    tasks = []
            dur = time.perf_counter() - t0
            ev = list_recovery_events(5000, wid)
            acts = Counter(e["action"] for e in ev)
            total = 8
            done = sum(t["status"] == "completed" for t in tasks)
            mttr = [e["duration_ms"] for e in ev if e["action"] in ("recovered", "checkpoint_restore") and e["duration_ms"] is not None]
            recoverable = sc != "repeated_failure"
            rows.append({"scenario": sc, "workflow_completed": bool(tasks) and done == len(tasks),
                         "recovered": bool(tasks) and done == len(tasks) if recoverable else bool(tasks) and acts["escalate"] > 0,
                         "tasks_completed": done, "lost_work_tasks": (len(tasks) or total) - done,
                         "mttr_ms": round(float(np.mean(mttr)), 1) if mttr else None,
                         "extra_attempts": acts["retry"] + acts["reassign"] + acts["checkpoint_restore"],
                         "escalations": acts["escalate"], "overhead_s": round(dur - baseline_s, 2)})
        dup = _duplicate_write_trial(mode)
        table[mode] = {
            "recovery_success_rate": round(np.mean([r["recovered"] for r in rows]), 4),
            "workflow_completion_rate": round(np.mean([r["workflow_completed"] for r in rows]), 4),
            "mean_time_to_recovery_ms": round(float(np.mean([r["mttr_ms"] for r in rows if r["mttr_ms"]])), 1) if any(r["mttr_ms"] for r in rows) else None,
            "lost_work_tasks": int(sum(r["lost_work_tasks"] for r in rows)),
            "duplicate_external_actions": dup["duplicates"], "duplicate_trial_outcome": dup["outcome"],
            "recovery_cost_extra_attempts": int(sum(r["extra_attempts"] for r in rows)),
            "scenarios": rows}
    _set(recovery_mode="full", debate_enabled=True, memory_enabled=True, risk_enabled=True)
    return {"strategies": table, "baseline_workflow_seconds": round(baseline_s, 2), "simulated": True,
            "note": "All faults are injected (simulated). 'recovered' for repeated_failure means it was escalated to a human rather than silently dropped."}


# ------------------------------------------------------------------ figures + report
def figures(res: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if "memory" in res:
        rows = [r for r in res["memory"]["retrieval"] if "error" not in r]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        labels = [f"{r['backend'].split('+')[0].split('-v1')[0]}\n{r['query_set']}" for r in rows]
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [r["recall_at_5"] for r in rows], 0.4, label="Recall@5")
        ax.bar(x + 0.2, [r["mrr"] for r in rows], 0.4, label="MRR")
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title("Memory retrieval: lexical vs paraphrased queries")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "enh_memory_retrieval.png", dpi=120)
        plt.close(fig)
    if "debate" in res:
        d = res["debate"]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        keys = ["qa_pass_rate", "task_completion_rate"]
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [d["no_debate"][k] for k in keys], 0.4, label="Coding + QA")
        ax.bar(x + 0.2, [d["debate"][k] for k in keys], 0.4, label="Coding + Critic + Judge + QA")
        ax.set_xticks(x, ["QA pass rate", "Clean completion rate"])
        ax.set_ylim(0, 1.05)
        ax.set_title("Debate vs no debate (planted-defect benchmark)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "enh_debate_comparison.png", dpi=120)
        plt.close(fig)
    if "recovery" in res:
        st = res["recovery"]["strategies"]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
        names = list(st)
        for ax, k, title in zip(axs, ["recovery_success_rate", "lost_work_tasks", "duplicate_external_actions"],
                                ["Recovery success rate", "Lost work (tasks)", "Duplicate external writes"]):
            ax.bar(names, [st[n][k] for n in names], color=["#9aa5b1", "#d29922", "#4c78a8", "#3fb950"])
            ax.set_title(title)
            ax.tick_params(axis="x", labelsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "enh_recovery_comparison.png", dpi=120)
        plt.close(fig)


def _tbl(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def report(res: dict) -> str:
    md = ["# AegisFlow AI — Enhancement Evaluation", "",
          "Generated by `scripts/evaluate_enhancements.py`. All numbers come from executed runs; simulated components are labelled.", ""]
    if "memory" in res:
        m = res["memory"]
        md += ["## 1. RAG memory retrieval", "",
               "Labelled benchmark: 140 verified memories (28 features x 5 categories), 280 queries per set; one relevant memory per query.", "",
               _tbl([[r["backend"], r["query_set"], r.get("recall_at_1", "-"), r.get("recall_at_5", "-"), r.get("mrr", "-"),
                      r.get("mean_latency_ms", "-"), r.get("context_tokens_per_task", "-")] if "error" not in r else
                     [r["backend"], r["query_set"], "-", "-", "-", "-", r["error"]] for r in m["retrieval"]],
                    ["backend", "queries", "Recall@1", "Recall@5", "MRR", "latency ms", "context tokens / task"]), "",
               "**Reading.** TF-IDF retrieves well when wording overlaps (lexical) but collapses on paraphrases, which is the case "
               "sentence embeddings are designed for. Run this script where the embedding model can be downloaded to fill the embedding rows.", "",
               "End-to-end (same requirements, memory off vs on):", "",
               _tbl([[k, v["dev_tasks"], v["first_attempt_success_rate"], v["memory_hits"], v["avg_context_tokens_per_dev_task"]]
                     for k, v in m["end_to_end"].items() if isinstance(v, dict)],
                    ["condition", "dev tasks", "first-attempt success", "memory hits", "avg context tokens"]), "",
               f"_{m['end_to_end']['note']}_", "", "![memory](figures/enh_memory_retrieval.png)", ""]
    if "risk" in res:
        r = res["risk"]
        md += ["## 2. Predictive failure detection", "", f"Data: **{r['data']}** ({r['rows']:,} executions, {r['workflows']} workflows); {r['split']}.", ""]
        rows = []
        for tname, x in r["targets"].items():
            c, d = x["test_calibrated"], x["test_dummy"]
            rows.append([tname, x["selected"], c["pr_auc"], d["pr_auc"], c["precision"], c["recall"], c["f1"], c["brier"],
                         x["test_uncalibrated_brier"], d["brier"], c["false_positive_rate"]])
        md += [_tbl(rows, ["target", "model", "PR-AUC", "dummy PR-AUC", "precision", "recall", "F1", "Brier (cal.)", "Brier (uncal.)",
                           "Brier (dummy)", "FPR"]), "",
               "**Limitations.** No provided real dataset contains agent execution outcomes, so the model learns a documented simulator. "
               "Live executions are recorded in `execution_metrics` for retraining (`prepare_failure_dataset(source='live')`). "
               "Task failure is intrinsically noisy in the simulator; deadline misses are more predictable (driven by slack and workload).", ""]
    if "debate" in res:
        d = res["debate"]
        md += ["## 3. AI agent debate", "", f"Benchmark: {d['benchmark']}. Backend: {d['backend']}. Completion = {d['task_completion_definition']}.", "",
               _tbl([[k, v["runs"], v["qa_pass_rate"], v["task_completion_rate"], v["defects_escaping_qa"], v["mean_revisions"],
                      v["mean_latency_ms"], v["mean_tokens"], v["estimated_api_cost_usd"]] for k, v in d.items() if isinstance(v, dict) and "runs" in v],
                    ["condition", "runs", "QA pass rate", "clean completion", "defects escaping QA", "revisions", "latency ms", "tokens", "API cost $"]), "",
               "By planted defect (QA pass rate / clean rate):", "",
               _tbl([[dn, d["no_debate"]["by_defect"][dn]["qa_pass_rate"], d["no_debate"]["by_defect"][dn]["clean_rate"],
                      d["debate"]["by_defect"][dn]["qa_pass_rate"], d["debate"]["by_defect"][dn]["clean_rate"]] for dn in d["debate"]["by_defect"]],
                    ["defect", "no debate: QA", "no debate: clean", "debate: QA", "debate: clean"]), "",
               "**Reading.** Unit tests alone let structural defects through (an untested placeholder function passes QA). The Critic "
               "catches those and the revision repairs them, at the cost of extra latency (~+35%). Debate is not free and is not a substitute "
               "for tests: the Judge only forwards work to QA.", "",
               "**Limitations found by this benchmark.** (1) The static critic detects removed input validation only when *no* `raise` "
               "statement remains; modules that still raise elsewhere (CRUD/config) slip through, hence 0.25 on that row - an LLM critic "
               "is needed for semantic checks. (2) The clean-completion criterion also counts public classes never referenced by tests "
               "(e.g. a dataclass) as a major issue, which is why unmodified template code scores 0.125 without debate; the debate fixes "
               "this only with existence/callability contract tests, not behavioural tests.", "", "![debate](figures/enh_debate_comparison.png)", ""]
    if "recovery" in res:
        st = res["recovery"]["strategies"]
        md += ["## 4. Self-healing recovery (simulated faults)", "",
               f"7 injected scenarios per strategy on the same 8-task workflow (fault-free baseline {res['recovery']['baseline_workflow_seconds']} s). "
               f"{res['recovery']['note']}", "",
               _tbl([[n, v["recovery_success_rate"], v["workflow_completion_rate"], v["mean_time_to_recovery_ms"], v["lost_work_tasks"],
                      v["duplicate_external_actions"], v["duplicate_trial_outcome"], v["recovery_cost_extra_attempts"]] for n, v in st.items()],
                    ["strategy", "recovery success", "workflow completion", "MTTR ms", "lost work (tasks)", "duplicate writes",
                     "lost-response PR trial", "extra attempts"]), "",
               "Per-scenario outcome (recovered?):", "",
               _tbl([[sc] + [next(r for r in st[n]["scenarios"] if r["scenario"] == sc)["recovered"] for n in st] for sc in SCENARIOS],
                    ["scenario", *st.keys()]), "", "![recovery](figures/enh_recovery_comparison.png)", ""]
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["memory", "risk", "debate", "recovery"])
    ap.add_argument("--report-only", action="store_true", help="regenerate markdown/figures from the saved JSON")
    a = ap.parse_args()
    if a.report_only:
        res = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        figures(res)
        OUT_MD.write_text(report(res), encoding="utf-8")
        print(f"wrote {OUT_MD}")
        return
    res = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    steps = {"memory": eval_memory, "risk": eval_risk, "debate": eval_debate, "recovery": eval_recovery}
    for name, fn in steps.items():
        if a.only and a.only != name:
            continue
        t0 = time.time()
        print(f"[{name}] running...", flush=True)
        res[name] = fn()
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    figures(res)
    OUT_MD.write_text(report(res), encoding="utf-8")
    print(f"wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
