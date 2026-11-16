"""Task category classifier (TF-IDF + linear/tree ensembles, tuned with GridSearchCV)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.config import CATEGORIES, settings
from app.ml import modeling
from app.ml.preprocessing import clean_text

MODEL_PATH = settings.models_dir / "task_classifier.joblib"
METRICS_PATH = settings.reports_dir / "classification_metrics.json"


def _pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000, sublinear_tf=True)),
        ("clf", estimator),
    ])


CANDIDATES = {
    "logistic_regression": (LogisticRegression(max_iter=2000), {"C": [0.5, 2.0, 8.0]}),
    "random_forest": (RandomForestClassifier(random_state=42, n_jobs=1), {"n_estimators": [200], "max_depth": [None, 30]}),
    "gradient_boosting": (GradientBoostingClassifier(random_state=42), {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]}),
}


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    X = lambda d: d["clean_text"]  # noqa: E731
    best, comparison = modeling.compare_models(
        _pipeline, CANDIDATES, X(train_df), train_df["task_category"], train_df["project_id"],
        X(val_df), val_df["task_category"], CATEGORIES)
    trval = pd.concat([train_df, val_df])
    final = modeling.refit_on_train_val(best, X(trval), trval["task_category"])
    metrics = {"task": "task_category_classification", "data": "SYNTHETIC", **comparison,
               **modeling.evaluate(final, X(test_df), test_df["task_category"], CATEGORIES)}
    modeling.save(final, metrics, MODEL_PATH, METRICS_PATH)
    load_model.cache_clear()
    return metrics


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def predict(descriptions: list[str]) -> list[tuple[str, float | None]]:
    model = load_model()
    if model is None:
        raise FileNotFoundError("task classifier not trained; run `python scripts/train_all.py`")
    return modeling.predict_with_confidence(model, pd.Series([clean_text(d) for d in descriptions]))
