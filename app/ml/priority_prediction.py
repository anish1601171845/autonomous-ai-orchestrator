"""Task priority predictor combining text, urgency, complexity, impact and dependency features."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import CATEGORIES, PRIORITIES, settings
from app.ml import modeling
from app.ml.preprocessing import add_features, clean_text

MODEL_PATH = settings.models_dir / "priority_model.joblib"
METRICS_PATH = settings.reports_dir / "priority_metrics.json"
NUMERIC = ["estimated_complexity", "urgency", "business_impact", "num_dependents", "urgent_cue", "low_cue"]


def _pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([
        ("text", TfidfVectorizer(min_df=3, max_features=500, sublinear_tf=True), "clean_text"),
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(categories=[CATEGORIES], handle_unknown="ignore"), ["task_category"]),
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])


CANDIDATES = {
    "logistic_regression": (LogisticRegression(max_iter=3000), {"C": [0.3, 1.0, 3.0]}),
    "random_forest": (RandomForestClassifier(random_state=42, n_jobs=1), {"n_estimators": [300], "max_depth": [8, 16], "min_samples_leaf": [2]}),
    "gradient_boosting": (GradientBoostingClassifier(random_state=42), {"n_estimators": [150, 300], "learning_rate": [0.05, 0.1], "max_depth": [3]}),
}


def _X(df: pd.DataFrame) -> pd.DataFrame:
    return df[["clean_text", "task_category", *NUMERIC]]


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    tr, va, te = (add_features(d) for d in (train_df, val_df, test_df))
    best, comparison = modeling.compare_models(
        _pipeline, CANDIDATES, _X(tr), tr["task_priority"], tr["project_id"], _X(va), va["task_priority"], PRIORITIES)
    trval = pd.concat([tr, va])
    final = modeling.refit_on_train_val(best, _X(trval), trval["task_priority"])
    metrics = {"task": "task_priority_prediction", "data": "SYNTHETIC", **comparison,
               **modeling.evaluate(final, _X(te), te["task_priority"], PRIORITIES)}
    modeling.save(final, metrics, MODEL_PATH, METRICS_PATH)
    load_model.cache_clear()
    return metrics


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def predict(records: list[dict]) -> list[tuple[str, float | None]]:
    """records: dicts with description, category, complexity, urgency, business_impact, num_dependents."""
    model = load_model()
    if model is None:
        raise FileNotFoundError("priority model not trained; run `python scripts/train_all.py`")
    df = pd.DataFrame({
        "clean_text": [clean_text(r.get("description", "")) for r in records],
        "task_category": [r.get("category") or "development" for r in records],
        "estimated_complexity": [r.get("complexity", 3) for r in records],
        "urgency": [r.get("urgency", 3) for r in records],
        "business_impact": [r.get("business_impact", 3) for r in records],
        "num_dependents": [r.get("num_dependents", 0) for r in records],
    })
    return modeling.predict_with_confidence(model, _X(add_features(df)))
