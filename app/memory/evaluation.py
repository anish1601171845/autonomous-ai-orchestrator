"""Labelled retrieval benchmark for agent memory: no memory vs TF-IDF vs sentence embeddings.

Corpus: one verified "memory" per (feature, category) pair, written with template A from the synthetic
task generator. Queries: the SAME (feature, category) written with different templates B and C.
Exactly one memory is relevant per query. Hard negatives are the same feature in other categories
(e.g. "Write unit tests for login" vs "Implement login backend"), so lexical overlap alone is not enough.

Two query sets:
* lexical   - the feature name is repeated verbatim (easy for TF-IDF; Recall@5 is near-trivial because
              the 5 category variants of a feature share its name, so Recall@1 and MRR matter more);
* paraphrase - the feature is described with different words ("sign-in and identity verification"
              instead of "user authentication"), which tests semantic rather than lexical matching.
"""
from __future__ import annotations

import tempfile
import time

import numpy as np

from app.memory.vector_store import TfidfEmbedder, VectorStore, get_embedder
from app.ml.synthetic_data import FEATURES, TEMPLATES

CATEGORIES = ["development", "testing", "documentation", "deployment", "analysis"]
SOLUTION = {
    "development": "Implemented {f} as a validated service module with unit tests; QA passed.",
    "testing": "Added pytest unit and edge-case tests for {f}; all tests pass in the sandbox.",
    "documentation": "Wrote {f} usage guide and API reference in docs/.",
    "deployment": "Added Dockerfile and CI pipeline to ship {f}; rollback plan documented.",
    "analysis": "Documented {f} requirements, constraints and acceptance criteria.",
}


PARAPHRASE = {
    "user authentication": "sign-in and identity verification", "user registration": "new account sign-up",
    "payment processing": "charging customer cards", "invoice generation": "producing billing statements",
    "reporting dashboard": "KPI overview screen", "inventory tracking": "stock level monitoring",
    "notification service": "alerts and reminder messages", "search engine": "full-text lookup",
    "order management": "purchase lifecycle handling", "customer profile": "client account details",
    "audit logging": "change history trail", "file upload": "document attachments",
    "role based access control": "permission levels per user type", "data export": "CSV download of records",
    "recommendation module": "personalised suggestions", "chat support": "live messaging with agents",
    "analytics pipeline": "metrics aggregation jobs", "shipping integration": "courier delivery tracking",
    "subscription billing": "recurring monthly charges", "admin console": "back-office management panel",
    "student records": "learner information", "attendance tracking": "presence register",
    "course catalog": "list of available classes", "employee onboarding": "new hire setup",
    "leave management": "vacation requests", "product catalog": "item listings",
    "shopping cart": "basket before checkout", "feedback survey": "customer satisfaction questionnaire",
}


def build_benchmark(paraphrase: bool = False) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    corpus, queries = [], []
    for f in FEATURES:
        for c in CATEGORIES:
            t = TEMPLATES[c]
            mid = f"{c}:{f}"
            corpus.append((mid, f"[{c}] {t[0].format(f=f)}\n{SOLUTION[c].format(f=f)}"))
            qf = PARAPHRASE.get(f, f) if paraphrase else f
            for q in (t[1], t[2]):
                queries.append((mid, q.format(f=qf)))
    return corpus, queries


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 characters per token (rough, model-independent)


def evaluate(backend: str, k: int = 5, paraphrase: bool = False) -> dict:
    corpus, queries = build_benchmark(paraphrase)
    if backend == "none":
        return {"backend": "no memory", "query_set": "paraphrase" if paraphrase else "lexical",
                "recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "mean_latency_ms": 0.0,
                "context_tokens_per_task": 0, "queries": len(queries), "memories": len(corpus)}
    emb = TfidfEmbedder() if backend == "tfidf" else get_embedder("embedding")
    with tempfile.TemporaryDirectory() as d:
        vs = VectorStore(d, emb)
        vs.build(corpus)
        docs = dict(corpus)
        hits, hits1, rr, lat, toks = 0, 0, [], [], []
        for rel, q in queries:
            t0 = time.perf_counter()
            res = vs.search(q, k)
            lat.append((time.perf_counter() - t0) * 1000)
            ids = [r[0] for r in res]
            hits += rel in ids
            hits1 += bool(ids) and ids[0] == rel
            rr.append(1 / (ids.index(rel) + 1) if rel in ids else 0.0)
            toks.append(sum(_tokens(docs[i]) for i in ids[:3]))
    return {"backend": vs.backend_name, "query_set": "paraphrase" if paraphrase else "lexical",
            "recall_at_1": round(hits1 / len(queries), 4), "recall_at_5": round(hits / len(queries), 4), "mrr": round(float(np.mean(rr)), 4),
            "mean_latency_ms": round(float(np.mean(lat)), 3), "p95_latency_ms": round(float(np.percentile(lat, 95)), 3),
            "context_tokens_per_task": int(np.mean(toks)), "queries": len(queries), "memories": len(corpus)}
