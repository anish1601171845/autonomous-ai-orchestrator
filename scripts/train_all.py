"""Reproducible training pipeline: data -> cleaning -> EDA figures -> classifier -> priority -> semantic -> GNN -> (RL).

Usage:
    python scripts/train_all.py                 # everything except RL
    python scripts/train_all.py --with-rl --rl-timesteps 300000
    python scripts/train_all.py --data data/raw/my_tasks.csv   # use your own labelled CSV (same columns)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.ml import classification, gnn_model, priority_prediction, semantic_mapping  # noqa: E402
from app.ml.preprocessing import clean, group_split, validate  # noqa: E402
from app.ml.synthetic_data import write_synthetic  # noqa: E402

FIG = settings.reports_dir / "figures"


def eda(df: pd.DataFrame) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    df["task_category"].value_counts().plot.bar(ax=ax[0], color="#4c78a8", title="Category distribution")
    df["task_priority"].value_counts().reindex(["low", "medium", "high", "critical"]).plot.bar(ax=ax[1], color="#f58518", title="Priority distribution")
    pd.crosstab(df["estimated_complexity"], df["task_priority"])[["low", "medium", "high", "critical"]].plot.bar(stacked=True, ax=ax[2], title="Priority by complexity")
    fig.tight_layout()
    fig.savefig(FIG / "eda_distributions.png", dpi=120)
    plt.close(fig)
    return {"rows": int(len(df)), "projects": int(df["project_id"].nunique()),
            "category_counts": df["task_category"].value_counts().to_dict(), "priority_counts": df["task_priority"].value_counts().to_dict(),
            "mean_tasks_per_project": round(float(df.groupby("project_id").size().mean()), 2),
            "corr_priority_rank": df.assign(pr=df["task_priority"].map({"low": 0, "medium": 1, "high": 2, "critical": 3}))
            [["pr", "urgency", "business_impact", "estimated_complexity", "num_dependents"]].corr()["pr"].drop("pr").round(3).to_dict()}


def confusion_png(m: dict, name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(m["confusion_matrix"], cmap="Blues")
    ax.set_xticks(range(len(m["labels"])), m["labels"], rotation=45, ha="right")
    ax.set_yticks(range(len(m["labels"])), m["labels"])
    for i, row in enumerate(m["confusion_matrix"]):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", fontsize=8)
    ax.set_title(name)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    fig.tight_layout()
    fig.savefig(FIG / f"cm_{name.replace(' ', '_').lower()}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--graphs", type=Path, default=None)
    ap.add_argument("--projects", type=int, default=300)
    ap.add_argument("--gnn-epochs", type=int, default=30)
    ap.add_argument("--with-rl", action="store_true")
    ap.add_argument("--rl-timesteps", type=int, default=100_000)
    a = ap.parse_args()

    if a.data is None:
        csv, gj = write_synthetic(n_projects=a.projects)
        print(f"[data] synthetic dataset written: {csv}")
    else:
        csv, gj = a.data, a.graphs
    raw = pd.read_csv(csv)
    df, report = clean(raw)
    problems = validate(df)
    if problems:
        raise SystemExit(f"dataset validation failed: {problems}")
    df.to_csv(settings.data_dir / "processed" / "tasks_clean.csv", index=False)
    tr, va, te = group_split(df)
    leak = set(tr["project_id"]) & set(te["project_id"])
    assert not leak, "project leakage between train and test"
    ds = {"source": "SYNTHETIC" if a.data is None else str(a.data), "cleaning": asdict(report),
          "split": {"train": len(tr), "val": len(va), "test": len(te), "grouped_by": "project_id", "project_overlap": 0}, "eda": eda(df)}
    (settings.reports_dir / "dataset_report.json").write_text(json.dumps(ds, indent=2, default=str), encoding="utf-8")
    print(f"[data] {report} split={ds['split']}")

    cm = classification.train(tr, va, te)
    confusion_png(cm, "Task classification")
    print(f"[classifier] selected={cm['selected']} test_acc={cm['test_accuracy']} macro_f1={cm['test_f1_macro']}")
    pm = priority_prediction.train(tr, va, te)
    confusion_png(pm, "Priority prediction")
    print(f"[priority] selected={pm['selected']} test_acc={pm['test_accuracy']} macro_f1={pm['test_f1_macro']}")

    semantic_mapping.get_encoder.cache_clear()
    enc = semantic_mapping.get_encoder()
    enc.fit(tr["task_description"].fillna("").tolist())  # fitted on training split only
    graphs = json.loads(Path(gj).read_text(encoding="utf-8")) if gj else []
    test_projects = set(te["project_id"])
    sm = semantic_mapping.evaluate_similarity_as_dependency(df, [g for g in graphs if g["project_id"] in test_projects], enc)
    semantic_mapping.save_metrics(sm)
    print(f"[semantic] backend={enc.backend} similarity-as-dependency AUC={sm['roc_auc']} AP={sm['average_precision']}")

    gm = gnn_model.train(df, graphs, epochs=a.gnn_epochs, test_projects=test_projects)
    if gm["status"] == "trained":
        print(f"[gnn] GraphSAGE {gm['gnn_graphsage']} | rule baseline {gm['rule_baseline_networkx']}")
    else:
        print(f"[gnn] {gm['note']}")

    if a.with_rl:
        from app.rl.train import train as train_rl

        rm = train_rl(a.rl_timesteps)
        print(f"[rl] {rm['verdict']}")


if __name__ == "__main__":
    main()
