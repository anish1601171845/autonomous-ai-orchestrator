import pytest

from app.memory import rag_memory, vector_store
from app.memory.evaluation import build_benchmark, evaluate


@pytest.fixture
def mem():
    rag_memory.reset_service()
    svc = rag_memory.service()
    for m in svc.list_memories(10_000):
        svc.delete_memory(m["memory_id"])
    yield svc
    rag_memory.reset_service()


LOGIN = ("Implement login backend service with validation",
         "Module login: UserService with PBKDF2 hashing, register/authenticate/logout; 8 tests passed")


def test_save_verified_and_retrieve(mem):
    r = rag_memory.save_verified_solution(*LOGIN, "qa_passed", task_category="development", agent_id="coder-1")
    assert r["saved"]
    rag_memory.save_verified_solution("Write the course catalog user guide", "docs/course_catalog.md written", "human_approved",
                                      task_category="documentation")
    hits = rag_memory.retrieve_similar_solutions("Build REST API endpoints for user login", task_category="development", min_score=0.0)
    assert hits and hits[0]["memory_id"] == r["memory_id"] and hits[0]["rank"] == 1 and 0 < hits[0]["score"] <= 1
    assert all(h["task_category"] == "development" for h in hits)


def test_only_verified_outcomes_are_stored(mem):
    assert not rag_memory.save_verified_solution(*LOGIN, "qa_failed")["saved"]
    assert not rag_memory.save_verified_solution(*LOGIN, "pending")["saved"]
    assert mem.list_memories() == []


def test_duplicates_rejected(mem):
    assert rag_memory.save_verified_solution(*LOGIN, "qa_passed", task_category="development")["saved"]
    again = rag_memory.save_verified_solution(*LOGIN, "qa_passed", task_category="development")
    assert not again["saved"] and "duplicate" in again["reason"]
    near = rag_memory.save_verified_solution(LOGIN[0] + " ", LOGIN[1] + "  ", "qa_passed", task_category="development")
    assert not near["saved"]


def test_secrets_and_contacts_never_stored(mem):
    rag_memory.save_verified_solution("Set up the deployment pipeline with credentials",
                                      "used token ghp_" + "a" * 36 + " and mailed ops@example.com", "qa_passed", task_category="deployment")
    stored = mem.list_memories()[0]["solution_summary"]
    assert "ghp_" not in stored and "ops@example.com" not in stored and "REDACTED" in stored


def test_persists_across_restart(mem):
    mid = rag_memory.save_verified_solution(*LOGIN, "qa_passed", task_category="development")["memory_id"]
    rag_memory.reset_service()  # simulate process restart: index rebuilt from the database
    hits = rag_memory.retrieve_similar_solutions("user login service", min_score=0.0, record=False)
    assert hits[0]["memory_id"] == mid


def test_access_restrictions(mem):
    rag_memory.save_verified_solution(*LOGIN, "qa_passed", task_category="development", allowed_roles=["coding"])
    rag_memory.save_verified_solution("Implement payroll export", "Module payroll: CSV export; 5 tests passed", "qa_passed",
                                      task_category="development", project_id="other-project")
    assert rag_memory.retrieve_similar_solutions("login service", agent_id="coder-1", min_score=0.0, record=False)
    assert not rag_memory.retrieve_similar_solutions("login service", agent_id="docs-1", min_score=0.0, record=False)
    assert not rag_memory.retrieve_similar_solutions("payroll export", min_score=0.0, record=False,
                                                     task_category="development") or all(
        h["task_description"] != "Implement payroll export" for h in rag_memory.retrieve_similar_solutions("payroll export", min_score=0.0, record=False))


def test_retrieval_is_logged_and_context_is_marked_untrusted(mem):
    rag_memory.save_verified_solution(*LOGIN, "qa_passed", task_category="development")
    hits = rag_memory.retrieve_similar_solutions("login backend", workflow_id=None, task_id="T9", min_score=0.0)
    assert mem.retrieval_history()[0]["task_id"] == "T9"
    ctx = rag_memory.build_memory_context(hits)
    assert ctx.startswith("<<UNTRUSTED_MEMORY_CONTEXT") and "Do not follow" in ctx and "```" not in ctx


def test_missing_embedding_model_falls_back_to_tfidf(monkeypatch):
    vector_store._EMBEDDERS.clear()

    def boom(*a, **k):
        raise OSError("model not downloadable")

    monkeypatch.setattr(vector_store, "SentenceEmbedder", boom)
    emb = vector_store.get_embedder("auto")
    assert emb.backend == "tfidf"
    with pytest.raises(OSError):
        vector_store._EMBEDDERS.clear()
        vector_store.get_embedder("embedding")
    vector_store._EMBEDDERS.clear()


def test_rebuild_index(mem):
    rag_memory.save_verified_solution(*LOGIN, "qa_passed", task_category="development")
    info = rag_memory.rebuild_index()
    assert info["memories"] == 1 and "tfidf" in info["backend"]


def test_retrieval_benchmark_is_labelled_and_runs():
    corpus, queries = build_benchmark()
    assert len(corpus) == 140 and len(queries) == 280
    lex, para = evaluate("tfidf"), evaluate("tfidf", paraphrase=True)
    assert lex["recall_at_5"] > para["recall_at_5"]  # TF-IDF depends on shared wording
    assert evaluate("none")["mrr"] == 0.0
