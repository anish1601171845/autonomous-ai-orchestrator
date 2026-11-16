import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="orch_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["WORKSPACE_DIR"] = str(_TMP / "workspace")
os.environ["LLM_API_KEY"] = ""
os.environ["GITHUB_TOKEN"] = ""
os.environ["GITHUB_REPO"] = ""
os.environ["SCHEDULER"] = "auto"
os.environ["MEMORY_INDEX_DIR"] = str(_TMP / "memory_index")
os.environ["MEMORY_BACKEND"] = "tfidf"          # offline + deterministic in CI
os.environ["RETRY_BASE_DELAY_S"] = "0.01"
os.environ["RETRY_MAX_DELAY_S"] = "0.05"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app.ml import classification, gnn_model, priority_prediction, semantic_mapping  # noqa: E402

MODELS_READY = all(p.exists() for p in (classification.MODEL_PATH, priority_prediction.MODEL_PATH, semantic_mapping.TFIDF_PATH))
needs_models = pytest.mark.skipif(not MODELS_READY, reason="trained models missing: run `python scripts/train_all.py`")

EXAMPLE = ("Build a student management system with registration, login, database integration, "
           "student dashboard, and automated testing.")


@pytest.fixture(autouse=True)
def _clear_caches():
    gnn_model.load_model.cache_clear()
    yield
