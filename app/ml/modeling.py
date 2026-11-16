"""Shared training utilities: model comparison with GridSearchCV + grouped CV, evaluation, persistence."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline


def compare_models(
    build_pipeline, candidates: dict[str, tuple[Any, dict]], X_tr: pd.DataFrame, y_tr: pd.Series,
    groups: pd.Series, X_va: pd.DataFrame, y_va: pd.Series, labels: list[str],
) -> tuple[Pipeline, dict]:
    """Tune every candidate with GridSearchCV (GroupKFold), pick best on validation macro-F1."""
    results = {}
    best_name, best_est, best_f1 = None, None, -1.0
    for name, (estimator, grid) in candidates.items():
        pipe = build_pipeline(estimator)
        param_grid = {f"clf__{k}": v for k, v in grid.items()}
        gs = GridSearchCV(pipe, param_grid, cv=GroupKFold(n_splits=3), scoring="f1_macro", n_jobs=-1)
        t0 = time.time()
        gs.fit(X_tr, y_tr, groups=groups)
        va_pred = gs.best_estimator_.predict(X_va)
        f1 = f1_score(y_va, va_pred, average="macro")
        results[name] = {
            "cv_f1_macro": round(float(gs.best_score_), 4),
            "val_f1_macro": round(float(f1), 4),
            "val_accuracy": round(float(accuracy_score(y_va, va_pred)), 4),
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "fit_seconds": round(time.time() - t0, 2),
        }
        if f1 > best_f1:
            best_name, best_est, best_f1 = name, gs.best_estimator_, f1
    return best_est, {"candidates": results, "selected": best_name}


def refit_on_train_val(est: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    final = clone(est)
    final.fit(X, y)
    return final


def evaluate(est: Pipeline, X_te: pd.DataFrame, y_te: pd.Series, labels: list[str]) -> dict:
    pred = est.predict(X_te)
    return {
        "test_accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "test_f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_te, pred, labels=labels).tolist(),
        "report": classification_report(y_te, pred, labels=labels, output_dict=True, zero_division=0),
        "n_test": int(len(y_te)),
    }


def save(est: Pipeline, metrics: dict, model_path: Path, metrics_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(est, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)), encoding="utf-8")


def predict_with_confidence(est: Pipeline, X: pd.DataFrame) -> list[tuple[str, float | None]]:
    preds = est.predict(X)
    if hasattr(est, "predict_proba"):
        probs = est.predict_proba(X).max(axis=1)
        return [(str(p), round(float(c), 4)) for p, c in zip(preds, probs)]
    return [(str(p), None) for p in preds]
