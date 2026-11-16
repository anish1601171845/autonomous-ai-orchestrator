import networkx as nx
import pandas as pd

from app.agents import planning_agent
from app.ml import classification, gnn_model, priority_prediction, semantic_mapping
from app.ml.preprocessing import clean, group_split, validate
from app.ml.synthetic_data import generate_projects
from tests.conftest import needs_models


def test_synthetic_and_preprocessing_pipeline():
    df, graphs = generate_projects(n_projects=20, seed=1)
    assert df["is_synthetic"].all() and len(graphs) == 20
    clean_df, rep = clean(df)
    assert rep.duplicates_removed > 0 and clean_df["task_id"].is_unique
    assert clean_df["estimated_complexity"].notna().all()
    assert validate(clean_df) == []
    tr, va, te = group_split(clean_df)
    assert not set(tr.project_id) & set(te.project_id)
    assert not set(va.project_id) & set(te.project_id)


def test_validate_reports_problems():
    assert validate(pd.DataFrame({"task_id": [1]}))[0].startswith("missing column")


@needs_models
def test_classifier_inference():
    preds = classification.predict(["Write unit tests for login", "Deploy billing to production", "Document the search API"])
    assert [p[0] for p in preds] == ["testing", "deployment", "documentation"]
    assert all(0 < p[1] <= 1 for p in preds)


@needs_models
def test_priority_inference_monotonic_in_urgency():
    base = {"description": "Implement payment processing backend service", "category": "development", "complexity": 4,
            "business_impact": 5, "num_dependents": 3}
    lo, hi = priority_prediction.predict([{**base, "urgency": 1, "business_impact": 1, "num_dependents": 0}, {**base, "urgency": 5}])
    order = ["low", "medium", "high", "critical"]
    assert order.index(hi[0]) > order.index(lo[0])


@needs_models
def test_semantic_candidates_are_unordered_suggestions():
    tasks = [{"task_id": "A", "description": "Implement login backend"}, {"task_id": "B", "description": "Write unit tests for login"},
             {"task_id": "C", "description": "Containerise service with Docker"}]
    cands = semantic_mapping.rank_candidates(tasks, min_sim=0.0)
    assert cands and all({"a", "b", "similarity"} <= c.keys() for c in cands)


def test_build_dag_rejects_cycles():
    edges = [{"source": "A", "target": "B", "score": 0.9, "method": "t"}, {"source": "B", "target": "C", "score": 0.8, "method": "t"},
             {"source": "C", "target": "A", "score": 0.5, "method": "t"}]
    dag, rejected = gnn_model.build_dag(["A", "B", "C"], edges)
    assert nx.is_directed_acyclic_graph(dag)
    assert rejected and rejected[0]["source"] == "C"
    assert gnn_model.find_cycles([("A", "B"), ("B", "A")])


@needs_models
def test_gnn_dependency_prediction_and_order():
    tasks = [
        {"task_id": "T1", "description": "Set up project repository, configuration and dependency management", "category": "development", "feature": "project", "complexity": 2},
        {"task_id": "T2", "description": "Implement login backend service with input validation", "category": "development", "feature": "login", "complexity": 4},
        {"task_id": "T3", "description": "Write unit tests for login", "category": "testing", "feature": "login", "complexity": 3},
        {"task_id": "T4", "description": "Run end-to-end integration tests across all modules", "category": "testing", "feature": "project", "complexity": 3},
    ]
    plan = planning_agent.run(tasks)
    assert plan["is_dag"]
    deps = {t["task_id"]: t["dependency_ids"] for t in tasks}
    assert "T2" in deps["T3"]  # tests depend on the implementation, not the reverse
    assert "T3" not in deps["T2"]
    order = planning_agent.execution_order(tasks)
    assert order.index("T2") < order.index("T3")


def test_rule_baseline_direction():
    import torch

    tasks = [{"task_id": "a", "category": "development", "feature": "x"}, {"task_id": "b", "category": "testing", "feature": "x"}]
    s = gnn_model.rule_baseline_scores(tasks, torch.tensor([[0, 1], [1, 0]]))
    assert list(s) == [1.0, 0.0]
