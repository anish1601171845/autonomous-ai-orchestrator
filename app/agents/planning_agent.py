"""Agent 2 — Planning & Dependency: semantic candidates + GNN dependency prediction + DAG validation."""
from __future__ import annotations

import networkx as nx

from app.ml import gnn_model, semantic_mapping

NAME = "planning_agent"
PRIO_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, None: 4}


def run(tasks: list[dict]) -> dict:
    gnn_input = [{"task_id": t["task_id"], "description": t["description"], "category": t.get("category"),
                  "feature": t.get("feature"), "complexity": t.get("complexity", 3)} for t in tasks]
    candidates = semantic_mapping.rank_candidates(gnn_input)
    edges, method = gnn_model.predict_dependencies(gnn_input)
    dag, rejected = gnn_model.build_dag([t["task_id"] for t in tasks], edges)
    deps = {t["task_id"]: sorted(dag.predecessors(t["task_id"])) for t in tasks}
    for t in tasks:
        t["dependency_ids"] = deps[t["task_id"]]
    return {
        "edges": [{"source": u, "target": v, "score": d.get("score", 1.0), "method": d.get("method", method)} for u, v, d in dag.edges(data=True)],
        "rejected_edges": rejected,
        "semantic_candidates": candidates[:25],
        "method": method,
        "encoder_backend": semantic_mapping.get_encoder().backend,
        "is_dag": nx.is_directed_acyclic_graph(dag),
    }


def execution_order(tasks: list[dict]) -> list[str]:
    g = nx.DiGraph()
    by_id = {t["task_id"]: t for t in tasks}
    g.add_nodes_from(by_id)
    for t in tasks:
        g.add_edges_from((d, t["task_id"]) for d in t.get("dependency_ids", []) if d in by_id)
    return list(nx.lexicographical_topological_sort(g, key=lambda n: (PRIO_RANK.get(by_id[n].get("priority"), 4), n)))
