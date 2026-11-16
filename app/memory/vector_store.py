"""Vector index for agent memory: Sentence-Transformers embeddings (preferred) or a TF-IDF fallback,
searched with FAISS (inner product on L2-normalised vectors = cosine similarity), numpy if FAISS is absent.

The SQL database is the source of truth; this index is a derived, persisted cache that is rebuilt
whenever its stored id list or embedding version no longer matches the database.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from app.core.config import settings
from app.core.security import get_logger

log = get_logger(__name__)

try:  # optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - exercised only when faiss is missing
    faiss = None


class TfidfEmbedder:
    """Offline fallback: word + char n-gram TF-IDF, refit on the current corpus at every rebuild."""

    version = "tfidf-word12-char35-v1"
    backend = "tfidf"

    def __init__(self) -> None:
        self._vec: FeatureUnion | None = None

    def fit(self, corpus: list[str]) -> None:
        self._vec = FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        self._vec.fit(corpus or ["memory index placeholder document"])

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._vec is None:
            raise RuntimeError("TF-IDF embedder not fitted")
        m = self._vec.transform(texts).toarray().astype("float32")
        return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)

    @property
    def needs_refit(self) -> bool:
        return True


class SentenceEmbedder:
    backend = "embedding"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model_name)
        self.version = f"st:{model_name}"

    def fit(self, corpus: list[str]) -> None:  # pretrained: nothing to fit
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")

    @property
    def needs_refit(self) -> bool:
        return False


_EMBEDDERS: dict[str, object] = {}


def get_embedder(backend: str | None = None):
    """auto -> sentence embeddings if the model loads (cached or downloadable), else TF-IDF."""
    backend = backend or settings.memory_backend
    if backend in _EMBEDDERS:
        return _EMBEDDERS[backend]
    emb = None
    if backend in ("auto", "embedding"):
        try:
            emb = SentenceEmbedder(settings.memory_embedding_model)
        except Exception as e:
            if backend == "embedding":
                raise
            log.warning("embedding model unavailable (%s); memory uses the TF-IDF fallback", type(e).__name__)
    emb = emb or TfidfEmbedder()
    _EMBEDDERS[backend] = emb
    return emb


class VectorStore:
    def __init__(self, directory: Path, embedder) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.ids: list[str] = []
        self._index = None
        self._matrix: np.ndarray | None = None

    # -------------------------------------------------------------- persistence
    @property
    def _meta_path(self) -> Path:
        return self.dir / "index_meta.json"

    def _save(self) -> None:
        meta = {"ids": self.ids, "version": self.embedder.version, "faiss": faiss is not None}
        if faiss is not None and self._index is not None and self.embedder.backend == "embedding":
            faiss.write_index(self._index, str(self.dir / "memory.faiss"))
        self._meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def try_load(self, expected_ids: list[str]) -> bool:
        """Load a persisted embedding index if it matches the DB. TF-IDF indexes are always rebuilt (refit)."""
        if self.embedder.backend != "embedding" or faiss is None or not self._meta_path.exists():
            return False
        meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != self.embedder.version or meta.get("ids") != expected_ids:
            return False
        path = self.dir / "memory.faiss"
        if not path.exists():
            return False
        self._index, self.ids = faiss.read_index(str(path)), list(expected_ids)
        return True

    # -------------------------------------------------------------- build / search
    def build(self, records: list[tuple[str, str]]) -> None:
        self.ids = [r[0] for r in records]
        texts = [r[1] for r in records]
        self.embedder.fit(texts)
        if not texts:
            self._index, self._matrix = None, None
            self._save()
            return
        vecs = self.embedder.encode(texts)
        if faiss is not None:
            self._index = faiss.IndexFlatIP(vecs.shape[1])
            self._index.add(vecs)
            self._matrix = None
        else:
            self._matrix = vecs
        self._save()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = self.embedder.encode([query])
        k = min(k, len(self.ids))
        if self._index is not None:
            scores, idx = self._index.search(q, k)
            pairs = zip(idx[0], scores[0])
        else:
            sims = (self._matrix @ q.T).ravel()
            top = np.argsort(-sims)[:k]
            pairs = ((i, sims[i]) for i in top)
        return [(self.ids[int(i)], float(s)) for i, s in pairs if int(i) >= 0]

    @property
    def backend_name(self) -> str:
        return f"{self.embedder.version}+{'faiss' if faiss is not None else 'numpy'}"
