import pytest

from app.agents import coding_agent
from app.agents import debate_engine as de
from app.database import repository

TASK = {"task_id": "T01", "title": "Implement login", "feature": "login",
        "description": "Implement login backend service with input validation and error handling",
        "acceptance_criteria": ["Inputs are validated and errors raise clear exceptions"]}


@pytest.fixture
def wf():
    wid = "wf_debate_" + __import__("uuid").uuid4().hex[:6]
    repository.create_workflow(wid, "debate test")
    coding_agent.generate_code(wid, TASK)
    return wid


def _break(wid):
    ws = coding_agent.workspace(wid)
    src = (ws / "login.py").read_text()
    src = src.replace("raise AuthError", "return None  # ")
    src += "\n\ndef export_users():\n    pass\n"
    (ws / "login.py").write_text(src)


def test_structured_messages_and_history(wf):
    res = de.start_debate(wf, TASK)
    assert isinstance(res.messages[0], de.DebateMessage)
    roles = [m.role for m in res.messages]
    assert roles[:3] == ["system", "developer", "critic"] and "judge" in roles
    h = de.get_debate_history(res.debate_id)
    assert h and len(h["messages"]) == len(res.messages) and h["task_context"]["task_id"] == "T01"


def test_critic_finds_issues_and_revision_fixes_them(wf):
    _break(wf)
    res = de.start_debate(wf, TASK)
    review = next(m for m in res.messages if m.role == "critic")
    codes = {i.code for i in review.issues}
    assert {"placeholder", "untested_api", "missing_validation"} <= codes
    assert any(m.kind == "revision" for m in res.messages) and res.revisions >= 1
    assert res.judge_verdict == "accept" and res.stop_reason == "accepted_by_judge"


def test_static_critic_flags_unsafe_code():
    issues = de.static_review("import subprocess\n\ndef run():\n    return subprocess.run(['ls'])\n", "def test_x():\n    run()\n", [])
    assert any(i.severity == "blocking" and i.code == "unsafe_or_invalid" for i in issues)


def test_max_round_limit_and_judge_reject(wf, monkeypatch):
    # revision that never fixes anything -> must stop at the round limit with a reject
    monkeypatch.setattr(de.DebateEngine, "revise_solution", lambda self, issues, r: self._add(
        de.DebateMessage(role="developer", kind="revision", round=r, content="no change")))
    _break(wf)
    res = de.DebateEngine(wf, TASK, max_rounds=3).run()
    assert res.rounds == 3 and res.stop_reason == "max_rounds_reached" and res.judge_verdict == "reject"
    assert de.DebateEngine(wf, TASK, max_rounds=9).max_rounds == 3  # hard cap


def test_budget_stops_debate(wf):
    _break(wf)
    res = de.DebateEngine(wf, TASK, token_budget=1).run()
    assert res.stop_reason == "token_budget_exhausted"


def test_qa_rejection_recorded_after_judge(wf):
    res = de.start_debate(wf, TASK)
    de.record_qa_result(res.debate_id, False, "1 failed")
    h = de.get_debate_history(res.debate_id)
    assert h["qa_passed"] is False and h["status"] == "qa_failed" and h["messages"][-1]["role"] == "qa"


def test_llm_timeout_falls_back_to_static_critic(wf, monkeypatch):
    monkeypatch.setattr(de.llm, "enabled", lambda: True)

    def timeout(*a, **k):
        raise TimeoutError("LLM timed out")

    monkeypatch.setattr(de, "_llm_review", timeout)
    monkeypatch.setattr(de.coding_agent, "generate_code", lambda *a, **k: {"backend": "template", "files": []})
    res = de.DebateEngine(wf, TASK).run()
    review = next(m for m in res.messages if m.role == "critic")
    assert "LLM reviewer failed (TimeoutError)" in review.content
