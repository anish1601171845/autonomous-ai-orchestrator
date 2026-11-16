"""Semantic task representations (BERT / RoBERTa / Sentence-Transformers, TF-IDF+LSA fallback)
and similarity-based dependency *candidate* ranking.

Semantic similarity is NOT dependency: two similar tasks ("write unit tests for login" and
"write unit tests for billing") are usually independent, and similarity is symmetric so it cannot
tell which task comes first. Similarity is therefore only used to propose candidates and as a GNN
input feature; final direction and acceptance are decided by the GNN + graph validation.
"""
from __future__ import annotations

import json
from functools import lru_cache
from itertools import permutations

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.core.security import get_logger
from app.ml.preprocessing import clean_text

log = get_logger(__name__)
TFIDF_PATH = settings.models_dir / "semantic_tfidf_lsa.joblib"


class SemanticEncoder:
    """Encodes text to L2-normalised vectors. `backend` reports what is actually in use."""

    def __init__(self, backend: str | None = None, model_name: str | None = None):
        self.model_name = model_name or settings.semantic_model
        wanted = backend or settings.semantic_backend
        self.backend = "tfidf_lsa"
        self._st = self._hf = self._tok = None
        if wanted in ("auto", "transformer"):
            try:
                self._load_transformer()
            except Exception as e:  # offline / model not cached
                if wanted == "transformer":
                    raise
                log.warning("Transformer encoder unavailable (%s); using TF-IDF+LSA fallback", type(e).__name__)
        if self.backend == "tfidf_lsa":
            self._lsa: Pipeline | None = joblib.load(TFIDF_PATH) if TFIDF_PATH.exists() else None

    def _load_transformer(self) -> None:
        name = self.model_name
        if "sentence-transformers" in name or "MiniLM" in name or "mpnet" in name:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(name)
            self.backend = f"sentence-transformers:{name}"
        else:  # bert-base-uncased / roberta-base etc: mean pooled token embeddings
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(name)
            self._hf = AutoModel.from_pretrained(name).eval()
            self.backend = f"transformers-meanpool:{name}"

    def fit(self, corpus: list[str]) -> SemanticEncoder:
        if self.backend == "tfidf_lsa":
            self._lsa = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("svd", TruncatedSVD(n_components=64, random_state=42)),
            ]).fit([clean_text(t) for t in corpus])
            joblib.dump(self._lsa, TFIDF_PATH)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = [clean_text(t) for t in texts]
        if self._st is not None:
            v = np.asarray(self._st.encode(texts, normalize_embeddings=True))
        elif self._hf is not None:
            import torch

            with torch.no_grad():
                enc = self._tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
                out = self._hf(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                v = ((out * mask).sum(1) / mask.sum(1)).numpy()
        else:
            if self._lsa is None:
                raise FileNotFoundError("TF-IDF/LSA encoder not fitted; run `python scripts/train_all.py`")
            v = self._lsa.transform(texts)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(n, 1e-9, None)

    @property
    def dim(self) -> int:
        return int(self.encode(["probe"]).shape[1])


@lru_cache(maxsize=1)
def get_encoder() -> SemanticEncoder:
    return SemanticEncoder()


def similarity_matrix(texts: list[str], encoder: SemanticEncoder | None = None) -> np.ndarray:
    e = (encoder or get_encoder()).encode(texts)
    return e @ e.T


def rank_candidates(tasks: list[dict], top_k: int = 3, min_sim: float = 0.25) -> list[dict]:
    """Rank (unordered) candidate relationships by semantic similarity. Not a dependency verdict."""
    if len(tasks) < 2:
        return []
    S = similarity_matrix([t["description"] for t in tasks])
    out = []
    for i in range(len(tasks)):
        order = np.argsort(-S[i])
        for j in [j for j in order if j != i][:top_k]:
            if S[i, j] >= min_sim and i < j:
                out.append({"a": tasks[i]["task_id"], "b": tasks[j]["task_id"], "similarity": round(float(S[i, j]), 4)})
    return sorted(out, key=lambda r: -r["similarity"])


def evaluate_similarity_as_dependency(df, graphs: list[dict], encoder: SemanticEncoder) -> dict:
    """Quantify how well raw similarity predicts *directed* dependency on labelled (synthetic) graphs."""
    text = dict(zip(df["task_id"], df["task_description"].fillna("")))
    y, s = [], []
    for g in graphs:
        ids = [t for t in g["task_ids"] if t in text]
        if len(ids) < 2:
            continue
        S = similarity_matrix([text[t] for t in ids], encoder)
        E = {tuple(e) for e in g["edges"]}
        idx = {t: k for k, t in enumerate(ids)}
        for a, b in permutations(ids, 2):
            y.append(int((a, b) in E))
            s.append(float(S[idx[a], idx[b]]))
    y, s = np.array(y), np.array(s)
    return {"method": f"cosine similarity ({encoder.backend})", "pairs": int(len(y)), "positive_rate": round(float(y.mean()), 4),
            "roc_auc": round(float(roc_auc_score(y, s)), 4), "average_precision": round(float(average_precision_score(y, s)), 4),
            "note": "Similarity is symmetric, so it scores A->B and B->A identically; it cannot recover direction."}


def save_metrics(m: dict) -> None:
    (settings.reports_dir / "semantic_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
