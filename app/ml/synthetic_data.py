"""Reproducible SYNTHETIC enterprise task dataset generator.

Every row is labelled `is_synthetic=True`. Results obtained on this data demonstrate that
the pipeline works end to end; they are NOT measurements of real enterprise performance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import PRIORITIES, settings

FEATURES = [
    "user authentication", "user registration", "payment processing", "invoice generation",
    "reporting dashboard", "inventory tracking", "notification service", "search engine",
    "order management", "customer profile", "audit logging", "file upload", "role based access control",
    "data export", "recommendation module", "chat support", "analytics pipeline", "shipping integration",
    "subscription billing", "admin console", "student records", "attendance tracking", "course catalog",
    "employee onboarding", "leave management", "product catalog", "shopping cart", "feedback survey",
]

TEMPLATES = {
    "analysis": [
        "Gather and analyse requirements for {f}",
        "Investigate current workflow and pain points of {f}",
        "Define acceptance criteria and user stories for {f}",
        "Evaluate data model options for {f}",
        "Assess performance requirements of {f} and produce findings",
        "Research third-party options for {f} and compare trade-offs",
    ],
    "development": [
        "Implement {f} backend service",
        "Build REST API endpoints for {f}",
        "Develop business logic for {f}",
        "Create database models and repository layer for {f}",
        "Refactor {f} module and add validation",
        "Write the {f} feature with input validation and error handling",
        "Code the {f} component and integrate with the service layer",
    ],
    "testing": [
        "Write unit tests for {f}",
        "Create integration tests covering {f}",
        "Add regression test suite for {f}",
        "Verify {f} behaviour with automated test cases",
        "Perform QA validation of {f} edge cases",
        "Measure test coverage of {f} and add missing tests",
    ],
    "documentation": [
        "Document the {f} API",
        "Write user guide for {f}",
        "Update README with {f} usage instructions",
        "Prepare architecture notes describing {f}",
        "Create onboarding documentation for {f}",
    ],
    "deployment": [
        "Deploy {f} to staging environment",
        "Configure CI/CD pipeline for {f}",
        "Containerise {f} with Docker",
        "Set up monitoring and alerts for {f} in production",
        "Release {f} to production with rollback plan",
    ],
}
PROJECT_TASKS = {
    "setup": ("development", "Set up project repository, configuration and dependency management"),
    "schema": ("development", "Design and implement the relational database schema"),
    "integration": ("testing", "Run end-to-end integration tests across all modules"),
    "docs": ("documentation", "Write the overall system documentation and deployment guide"),
    "deploy": ("deployment", "Deploy the complete application to production"),
}
PROJECT_ANALYSIS = [
    "Gather and analyse requirements and acceptance criteria for the system",
    "Define the overall functional and non-functional requirements",
    "Analyse stakeholder needs and write user stories for the platform",
]
PROJECT_DEPLOY = [
    "Deploy the complete application to production",
    "Deploy the complete system to production with Docker and a CI/CD pipeline",
    "Containerise the application with Docker and release it to production",
    "Configure the CI/CD pipeline and deploy the platform to the cloud",
]
URGENCY_CUES = ["urgent:", "blocking release:", "high-visibility:", "customer escalation:"]
LOW_CUES = ["nice to have:", "when time permits:", "low risk:"]
CATEGORY_BONUS = {"deployment": 0.4, "development": 0.2, "testing": 0.1, "analysis": 0.0, "documentation": -0.4}


def _rand_feature_tasks(rng: np.random.Generator, feature: str) -> list[tuple[str, str, str]]:
    """Return (role, category, description) for the standard per-feature task chain."""
    out = []
    if rng.random() < 0.7:
        out.append(("analysis", "analysis", rng.choice(TEMPLATES["analysis"]).format(f=feature)))
    out.append(("dev", "development", rng.choice(TEMPLATES["development"]).format(f=feature)))
    out.append(("test", "testing", rng.choice(TEMPLATES["testing"]).format(f=feature)))
    if rng.random() < 0.5:
        out.append(("doc", "documentation", rng.choice(TEMPLATES["documentation"]).format(f=feature)))
    if rng.random() < 0.3:
        out.append(("dep", "deployment", rng.choice(TEMPLATES["deployment"]).format(f=feature)))
    return out


def generate_projects(n_projects: int = 300, seed: int = 42) -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    graphs: list[dict] = []
    agents = settings.agent_pool
    tid = 0
    for p in range(n_projects):
        pid = f"P{p:04d}"
        feats = list(rng.choice(FEATURES, size=int(rng.integers(3, 7)), replace=False))
        tasks: list[dict] = []
        edges: list[tuple[int, int]] = []

        def add(cat: str, desc: str, feature: str) -> int:
            nonlocal tid
            tid += 1
            tasks.append({"task_id": f"T{tid:06d}", "project_id": pid, "task_description": desc,
                          "task_category": cat, "feature": feature})
            return len(tasks) - 1

        setup = add(*PROJECT_TASKS["setup"], "project")
        # structural variety: projects without a DB schema, and with a project-level analysis task
        has_schema = rng.random() < 0.7
        base = setup
        if has_schema:
            schema = add(*PROJECT_TASKS["schema"], "project")
            edges.append((setup, schema))
            base = schema
        proj_analysis = None
        if rng.random() < 0.5:
            proj_analysis = add("analysis", str(rng.choice(PROJECT_ANALYSIS)), "project")
            edges.append((setup, proj_analysis))
        test_nodes, dev_nodes = [], []
        for f in feats:
            idx = {}
            for role, cat, desc in _rand_feature_tasks(rng, f):
                idx[role] = add(cat, desc, f)
            if "analysis" in idx:
                edges.append((proj_analysis if proj_analysis is not None else setup, idx["analysis"]))
                edges.append((idx["analysis"], idx["dev"]))
            elif proj_analysis is not None:
                edges.append((proj_analysis, idx["dev"]))
            edges.append((base, idx["dev"]))
            edges.append((idx["dev"], idx["test"]))
            if "doc" in idx:
                edges.append((idx["dev"], idx["doc"]))
            if "dep" in idx:
                edges.append((idx["test"], idx["dep"]))
            test_nodes.append(idx["test"])
            dev_nodes.append(idx["dev"])
        # cross-feature dependency (e.g. dashboard depends on auth) ~ realistic coupling
        if len(dev_nodes) > 2 and rng.random() < 0.6:
            a, b = rng.choice(len(dev_nodes), size=2, replace=False)
            a, b = min(a, b), max(a, b)
            edges.append((dev_nodes[a], dev_nodes[b]))
        integ = add(*PROJECT_TASKS["integration"], "project")
        edges += [(t, integ) for t in test_nodes]
        if rng.random() < 0.8:
            docs = add(*PROJECT_TASKS["docs"], "project")
            edges += [(d, docs) for d in dev_nodes]
        deploy = add("deployment", str(rng.choice(PROJECT_DEPLOY)), "project")
        edges.append((integ, deploy))

        n_dependents = np.zeros(len(tasks))
        for s, _t in edges:
            n_dependents[s] += 1
        for i, t in enumerate(tasks):
            complexity = int(np.clip(rng.normal({"development": 3.6, "testing": 2.8, "deployment": 3.2,
                                                 "analysis": 2.5, "documentation": 2.0}[t["task_category"]], 1.0), 1, 5))
            urgency = int(rng.integers(1, 6))
            impact = int(rng.integers(1, 6))
            desc = t["task_description"]
            if urgency >= 5 and rng.random() < 0.6:
                desc = f"{rng.choice(URGENCY_CUES)} {desc}"
            elif urgency <= 1 and rng.random() < 0.5:
                desc = f"{rng.choice(LOW_CUES)} {desc}"
            score = (0.45 * urgency + 0.4 * impact + 0.15 * complexity + 0.5 * math.log1p(n_dependents[i])
                     + CATEGORY_BONUS[t["task_category"]] + rng.normal(0, 0.35))
            est = round(float(complexity * 2.5 * rng.uniform(0.8, 1.2)), 1)
            eligible = [a for a in agents if t["task_category"] in a["capabilities"]]
            agent = eligible[int(rng.integers(len(eligible)))]
            actual = round(float(est / agent["skill"] * rng.lognormal(0, 0.25)), 1)
            deps = [tasks[s]["task_id"] for s, d in edges if d == i]
            t.update({"task_description": desc, "estimated_complexity": complexity, "urgency": urgency,
                      "business_impact": impact, "num_dependents": int(n_dependents[i]),
                      "dependency_ids": json.dumps(deps), "estimated_duration": est,
                      "assigned_agent": agent["name"], "_score": score, "actual_duration": actual,
                      "completion_status": "completed" if rng.random() > 0.07 * complexity / 3 else "failed",
                      "is_synthetic": True})
        rows += tasks
        graphs.append({"project_id": pid, "task_ids": [t["task_id"] for t in tasks],
                       "edges": [[tasks[s]["task_id"], tasks[d]["task_id"]] for s, d in edges]})

    df = pd.DataFrame(rows)
    qs = df["_score"].quantile([0.30, 0.65, 0.90]).values
    df["task_priority"] = pd.cut(df["_score"], [-np.inf, *qs, np.inf], labels=PRIORITIES).astype(str)
    df = df.drop(columns="_score")
    # 3% label noise on category to mimic annotation errors
    flip = rng.random(len(df)) < 0.03
    df.loc[flip, "task_category"] = rng.choice(list(TEMPLATES), size=int(flip.sum()))
    # realistic dirt: missing values + exact duplicates (cleaned by preprocessing)
    miss = rng.random(len(df)) < 0.04
    df.loc[miss, "estimated_complexity"] = np.nan
    df.loc[rng.random(len(df)) < 0.01, "task_description"] = None
    dup = df.sample(frac=0.02, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)
    return df, graphs


def write_synthetic(out_dir: Path | None = None, n_projects: int = 300, seed: int = 42) -> tuple[Path, Path]:
    out_dir = out_dir or settings.data_dir / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    df, graphs = generate_projects(n_projects, seed)
    csv = out_dir / "synthetic_tasks.csv"
    gj = out_dir / "synthetic_graphs.json"
    df.to_csv(csv, index=False)
    gj.write_text(json.dumps(graphs), encoding="utf-8")
    return csv, gj
