"""Requirement-type classifier trained on the REAL PROMISE NFR dataset.

Two models:
  * binary  : functional vs non-functional
  * multiclass: F + 11 NFR types (security, performance, usability, ...)
Both use word + character n-gram TF-IDF and are tuned with GridSearchCV (stratified CV on the
training file); the separate PROMISE test file is used once for final scoring.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline

from app.core.config import settings
from app.ml.real_data import PROMISE_TEST, PROMISE_TRAIN, REQUIREMENT_TYPES, clean_requirement, load_promise

MODEL_PATH = settings.models_dir / "requirements_classifier.joblib"
METRICS_PATH = settings.reports_dir / "requirements_metrics.json"


def _pipeline(est) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("feats", feats), ("clf", est)])


CANDIDATES = {
    "logistic_regression": (LogisticRegression(max_iter=4000, class_weight="balanced"), {"C": [1.0, 5.0, 20.0]}),
    "linear_svm_sgd": (SGDClassifier(loss="modified_huber", class_weight="balanced", random_state=42, max_iter=3000),
                       {"alpha": [1e-5, 1e-4, 1e-3]}),
    "complement_nb": (ComplementNB(), {"alpha": [0.1, 0.3, 1.0]}),
}


def _fit_task(X_tr, y_tr, X_te, y_te, labels: list[str]) -> tuple[Pipeline, dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, best, best_score = {}, None, -1.0
    for name, (est, grid) in CANDIDATES.items():
        gs = GridSearchCV(_pipeline(est), {f"clf__{k}": v for k, v in grid.items()}, cv=cv, scoring="f1_macro", n_jobs=-1)
        t0 = time.time()
        gs.fit(X_tr, y_tr)
        results[name] = {"cv_f1_macro": round(float(gs.best_score_), 4),
                         "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                         "fit_seconds": round(time.time() - t0, 2)}
        if gs.best_score_ > best_score:
            best, best_score, best_name = gs.best_estimator_, gs.best_score_, name
    pred = best.predict(X_te)
    present = [lbl for lbl in labels if lbl in set(y_te) | set(pred)]
    majority = pd.Series(y_tr).mode()[0]
    return best, {
        "candidates": results, "selected": best_name,
        "test_accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "test_f1_macro": round(float(f1_score(y_te, pred, average="macro", labels=present, zero_division=0)), 4),
        "test_f1_weighted": round(float(f1_score(y_te, pred, average="weighted", zero_division=0)), 4),
        "majority_baseline_accuracy": round(float((np.asarray(y_te) == majority).mean()), 4),
        "labels": present, "confusion_matrix": confusion_matrix(y_te, pred, labels=present).tolist(),
        "report": classification_report(y_te, pred, labels=present, output_dict=True, zero_division=0),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    }


def train() -> dict:
    tr, te = load_promise(PROMISE_TRAIN), load_promise(PROMISE_TEST)
    te = te[~te["text"].isin(set(tr["text"]))]  # guard against train/test overlap
    bin_model, bin_m = _fit_task(tr["text"], tr["is_functional"].map({1: "functional", 0: "non_functional"}),
                                 te["text"], te["is_functional"].map({1: "functional", 0: "non_functional"}),
                                 ["functional", "non_functional"])
    multi_model, multi_m = _fit_task(tr["text"], tr["label"], te["text"], te["label"], list(REQUIREMENT_TYPES))
    # Evaluation above uses the untouched PROMISE test file. The *deployed* models are then refit on
    # train + test (same tuned hyper-parameters) so inference benefits from every labelled statement.
    allx = pd.concat([tr, te], ignore_index=True)
    bin_model.fit(allx["text"], allx["is_functional"].map({1: "functional", 0: "non_functional"}))
    multi_model.fit(allx["text"], allx["label"])
    joblib.dump({"binary": bin_model, "multiclass": multi_model}, MODEL_PATH)
    metrics = {"dataset": "PROMISE NFR (real, public)", "train_label_counts": tr["label"].value_counts().to_dict(),
               "test_label_counts": te["label"].value_counts().to_dict(), "label_names": REQUIREMENT_TYPES,
               "binary_functional_vs_nfr": bin_m, "multiclass_requirement_type": multi_m,
               "deployed_model": f"refit on train+test ({len(tr) + len(te)} statements) after evaluation"}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    load_model.cache_clear()
    return metrics


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def predict(sentences: list[str]) -> list[dict]:
    m = load_model()
    if m is None:
        raise FileNotFoundError("requirements classifier not trained; run `python scripts/train_real.py`")
    texts = [clean_requirement(s) for s in sentences]
    out = []
    multi_p = m["multiclass"].predict_proba(texts)
    bin_p = m["binary"].predict_proba(texts)
    mc, bc = list(m["multiclass"].classes_), list(m["binary"].classes_)
    for i, t in enumerate(texts):
        label = mc[int(np.argmax(multi_p[i]))]
        is_func = bc[int(np.argmax(bin_p[i]))] == "functional"
        # keep the two heads consistent: the binary head (more data per class) decides F vs NFR
        if is_func and label != "F":
            label = "F"
        elif not is_func and label == "F":
            order = np.argsort(-multi_p[i])
            label = next(mc[j] for j in order if mc[j] != "F")
        out.append({"text": t, "label": label, "label_name": REQUIREMENT_TYPES[label], "is_functional": label == "F",
                    "confidence": round(float(multi_p[i][mc.index(label)]), 4),
                    "functional_probability": round(float(bin_p[i][bc.index("functional")]), 4)})
    return out
