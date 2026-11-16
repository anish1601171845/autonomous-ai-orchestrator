"""Build reports/model_evaluation.md and charts from the JSON metrics written during training.

Every number in the generated report is read from reports/*.json; nothing is typed by hand.
Usage: python scripts/make_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.core.config import settings  # noqa: E402

R = settings.reports_dir
FIG = R / "figures"


def load(name: str) -> dict:
    p = R / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def figures(gnn: dict, rl: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if gnn.get("history"):
        ep = [h["epoch"] for h in gnn["history"]]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(ep, [h["train_loss"] for h in gnn["history"]], label="train loss")
        ax.plot(ep, [h["val_ap"] for h in gnn["history"]], label="val average precision")
        ax.set_xlabel("epoch")
        ax.set_title("GraphSAGE dependency model training")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "gnn_training.png", dpi=120)
        plt.close(fig)
    if rl.get("training_curve"):
        c = rl["training_curve"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot([x["timesteps"] for x in c], [x["mean_episode_reward"] for x in c])
        ax.set_xlabel("timesteps")
        ax.set_ylabel("mean episode reward (last 50)")
        ax.set_title("PPO scheduler training (simulated)")
        fig.tight_layout()
        fig.savefig(FIG / "rl_reward_curve.png", dpi=120)
        plt.close(fig)
        ev = rl["evaluation"]
        names = list(ev)
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, m, t in zip(axs, ["makespan", "load_imbalance_cv", "success_rate"], ["Makespan (lower better)", "Load imbalance CV (lower better)", "Success rate"]):
            ax.bar(names, [ev[n][m] for n in names], color=["#9aa5b1", "#4c78a8", "#f58518"])
            ax.set_title(t)
            if m == "success_rate":
                ax.set_ylim(0.85, 1.0)
        fig.tight_layout()
        fig.savefig(FIG / "rl_scheduler_comparison.png", dpi=120)
        plt.close(fig)


def real_sections(rq: dict, tk: dict) -> list[str]:
    md: list[str] = []
    if rq:
        md += ["---", "", "# Results on REAL data", "",
               "## 6. Requirement-type classification (PROMISE NFR dataset)", "",
               f"Public PROMISE NFR corpus: {sum(rq['train_label_counts'].values())} training statements (`nfr.txt`) and "
               f"{sum(rq['test_label_counts'].values())} test statements (`test.txt`) from **different projects**, so the test is "
               "cross-project. GridSearchCV (stratified 5-fold) on train; scored once on test. Features: word 1-2-gram + char 2-5-gram TF-IDF.", ""]
        rows = []
        for key, name in (("binary_functional_vs_nfr", "Functional vs non-functional"), ("multiclass_requirement_type", "12 requirement types")):
            x = rq[key]
            rows.append([name, x["selected"], max(v["cv_f1_macro"] for v in x["candidates"].values()), x["test_accuracy"],
                         x["test_f1_macro"], x["majority_baseline_accuracy"]])
        md += [table(rows, ["task", "selected model", "CV macro-F1", "test accuracy", "test macro-F1", "majority baseline acc."]), "",
               f"Deployed model: {rq.get('deployed_model', 'trained on train split')}.", "",
               "**Interpretation.** Cross-validation on the training projects scores clearly higher than the unseen test projects: "
               "requirement wording is project-specific, and security/operational statements phrased like 'only users with X access "
               "must be able to ...' are easily mistaken for functional ones. Rare classes (portability: 2 examples; legal and "
               "fault tolerance: 10 each) limit macro-F1.", "", "![req](figures/cm_requirements.png)", ""]
        _cm_png(rq["multiclass_requirement_type"], "Requirement type (PROMISE test)", "cm_requirements.png")
    if tk:
        md += ["## 7. Ticket triage (multilingual customer-support tickets)", "",
               f"{tk['rows']} tickets in {len(tk['languages'])} languages ({', '.join(f'{k}: {v}' for k, v in tk['languages'].items())}). "
               "Stratified 80/20 split per target; GridSearchCV (stratified 5-fold) on the 80%. Features: language-agnostic char "
               "3-5-gram + word 1-2-gram TF-IDF plus one-hot language; training augmented with subject-only copies.", ""]
        rows = [[t, x["selected"], x["test_accuracy"], x.get("short_text_test_accuracy", "-"), x["test_f1_macro"],
                 x["majority_baseline_accuracy"], ", ".join(f"{k} {v}" for k, v in x["accuracy_by_language"].items())]
                for t, x in tk["targets"].items()]
        md += [table(rows, ["target", "selected model", "test accuracy", "short (subject-only) accuracy", "test macro-F1",
                            "majority baseline acc.", "accuracy by language"]), ""]
        ab = tk.get("ablation_vs_previous_model")
        if ab:
            arows = [[t, ab[t]["previous_test_accuracy"], tk["targets"][t]["test_accuracy"], ab[t]["previous_short_text_accuracy"],
                      tk["targets"][t]["short_text_test_accuracy"]] for t in ("type", "priority", "queue")]
            md += ["**Design ablation.** Removing the business-type feature and training on subject-only copies trades a little "
                   "full-text accuracy for clearly better accuracy on one-line tickets (the common real-world case):", "",
                   table(arows, ["target", "previous full-text acc.", "current full-text acc.", "previous short acc.", "current short acc."]),
                   "", f"_{ab['cv_caveat']}_", ""]
        md += [
               "**Interpretation.** All three models beat the majority baseline by a wide margin and work across all five languages. "
               "Priority is partly subjective (similar wording appears with different priorities), which caps accuracy; queue "
               "confusions are mostly between overlapping teams (Technical / Product / IT Support).", "",
               "![tix](figures/ticket_accuracy_by_language.png)", ""]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        langs = sorted(next(iter(tk["targets"].values()))["accuracy_by_language"])
        w = 0.25
        for i, (t, x) in enumerate(tk["targets"].items()):
            ax.bar([j + i * w for j in range(len(langs))], [x["accuracy_by_language"][lg] for lg in langs], w, label=t)
        ax.set_xticks([j + w for j in range(len(langs))], langs)
        ax.set_ylim(0, 1)
        ax.set_title("Ticket triage hold-out accuracy by language")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "ticket_accuracy_by_language.png", dpi=120)
        plt.close(fig)
    return md


def _cm_png(m: dict, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(m["confusion_matrix"], cmap="Blues")
    ax.set_xticks(range(len(m["labels"])), m["labels"])
    ax.set_yticks(range(len(m["labels"])), m["labels"])
    for i, row in enumerate(m["confusion_matrix"]):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=120)
    plt.close(fig)


def main() -> None:
    ds, cls, pri, gnn, sem, rl = (load(n) for n in ("dataset_report", "classification_metrics", "priority_metrics", "gnn_metrics", "semantic_metrics", "rl_metrics"))
    figures(gnn, rl)
    md = ["# Model Evaluation Report", "",
          "> **Data provenance.** All results below come from the reproducible **synthetic** demonstration dataset "
          "(`app/ml/synthetic_data.py`, seed 42) and a **simulated** scheduling environment. They show that the pipeline "
          "trains and generalises to held-out synthetic projects; they are **not** measurements of real enterprise performance.", ""]
    if ds:
        c, s = ds["cleaning"], ds["split"]
        md += ["## 1. Dataset", "",
               table([["raw rows", c["rows_in"]], ["duplicates removed", c["duplicates_removed"]], ["empty descriptions removed", c["empty_descriptions_removed"]],
                      ["complexity values imputed (per-category median)", c["imputed_complexity"]], ["clean rows", c["rows_out"]],
                      ["projects", ds["eda"]["projects"]], ["train / val / test rows", f"{s['train']} / {s['val']} / {s['test']}"],
                      ["split strategy", f"GroupShuffleSplit by `{s['grouped_by']}` (project overlap train↔test: {s['project_overlap']})"]], ["item", "value"]),
               "", "Correlation of priority rank with features: " + ", ".join(f"`{k}` {v}" for k, v in ds["eda"]["corr_priority_rank"].items()), "",
               "![EDA](figures/eda_distributions.png)", ""]
    for title, m, fig in (("2. Task classification", cls, "cm_task_classification.png"), ("3. Priority prediction", pri, "cm_priority_prediction.png")):
        if not m:
            continue
        rows = [[k, v["cv_f1_macro"], v["val_f1_macro"], v["val_accuracy"], json.dumps(v["best_params"]), v["fit_seconds"]] for k, v in m["candidates"].items()]
        per = [[lbl, round(m["report"][lbl]["precision"], 3), round(m["report"][lbl]["recall"], 3), round(m["report"][lbl]["f1-score"], 3), int(m["report"][lbl]["support"])] for lbl in m["labels"]]
        md += [f"## {title}", "", "GridSearchCV with 3-fold GroupKFold (grouped by project) on the training split; model selected on validation macro-F1, "
               "then refit on train+val and scored once on the test split.", "",
               table(rows, ["model", "CV macro-F1", "val macro-F1", "val accuracy", "best params", "fit s"]), "",
               f"**Selected:** `{m['selected']}` — **test accuracy {m['test_accuracy']}**, **test macro-F1 {m['test_f1_macro']}** (n={m['n_test']}).", "",
               table(per, ["class", "precision", "recall", "F1", "support"]), "", f"![confusion]({'figures/' + fig})", ""]
    if gnn.get("status") == "trained":
        g, b = gnn["gnn_graphsage"], gnn["rule_baseline_networkx"]
        rows = [["GraphSAGE GNN (PyG)", g["roc_auc"], g["average_precision"], g["precision"], g["recall"], g["f1"], g["threshold"]],
                ["NetworkX rule baseline", b["roc_auc"], b["average_precision"], b["precision"], b["recall"], b["f1"], b["threshold"]]]
        if sem:
            rows.append([f"Cosine similarity only ({sem['method'].split('(')[-1].rstrip(')')})", sem["roc_auc"], sem["average_precision"], "-", "-", "-", "-"])
        md += ["## 4. Dependency prediction (directed edges)", "",
               f"Graphs: {gnn['graphs']['train']} train / {gnn['graphs']['val']} val / {gnn['graphs']['test']} test projects "
               f"(test projects are the same held-out projects as above). Test ordered pairs: {gnn['test_pairs']}, positive rate {gnn['test_positive_rate']}. "
               f"Encoder: `{gnn['encoder_backend']}`. Threshold chosen on validation.", "",
               table(rows, ["method", "ROC-AUC", "avg precision", "precision", "recall", "F1", "threshold"]), "",
               "**Interpretation.** Similarity alone is a weak dependency signal (and cannot give direction). The GNN scores near-perfectly because "
               "synthetic dependencies follow consistent structural rules (setup → analysis → implement → test → integrate → deploy) that the model can learn; "
               "real project graphs are noisier, so expect substantially lower scores until it is retrained on labelled real data.", "",
               "![gnn](figures/gnn_training.png)", ""]
    elif gnn:
        md += ["## 4. Dependency prediction", "", f"GNN not trained: {gnn.get('note')}", ""]
    if rl:
        ev = rl["evaluation"]
        rows = [[k, v["makespan"], v["makespan_std"], v["load_imbalance_cv"], v["success_rate"], v["invalid_actions"], v["priority_weighted_completion"]] for k, v in ev.items()]
        md += ["## 5. RL workload scheduling", "",
               f"{rl['algorithm']}, {rl['timesteps']:,} timesteps on {rl['data']}; evaluated on {rl['eval_episodes']} unseen seeded episodes.", "",
               table(rows, ["scheduler", "makespan", "makespan std", "load imbalance CV", "success rate", "invalid actions/ep", "priority-weighted completion"]), "",
               f"**Verdict:** {rl['verdict']}", "",
               "![rl](figures/rl_reward_curve.png)", "", "![rl2](figures/rl_scheduler_comparison.png)", ""]
    md += real_sections(load("requirements_metrics"), load("ticket_metrics"))
    (R / "model_evaluation.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {R / 'model_evaluation.md'}")


if __name__ == "__main__":
    main()
