import pytest

from app.agents import code_templates, coding_agent, qa_agent

CASES = [
    ("login", "Implement login backend service with input validation", "auth"),
    ("project", "Design and implement the relational database schema for the app", "database"),
    ("student dashboard", "Implement student dashboard backend service", "dashboard"),
    ("project", "Set up project repository, configuration and dependency management", "config"),
    ("course catalog", "Implement course catalog backend service", "crud"),
]


@pytest.mark.parametrize("feature,desc,kind", CASES)
def test_generated_code_passes_sandboxed_qa(feature, desc, kind):
    assert code_templates.archetype(feature, desc) == kind
    task = {"task_id": "T1", "title": desc, "description": desc, "feature": feature, "acceptance_criteria": []}
    out = coding_agent.generate_code(f"wf_test_{kind}", task)
    assert out["backend"] == f"template:{kind}" and out["explanation"]
    report = qa_agent.review(out["files"])
    assert report["passed"], report
    assert report["tests"]["passed"] >= 3 and report["tests"]["failed"] == 0


def test_qa_blocks_insecure_code_without_running_it():
    ws = coding_agent.workspace("wf_test_insecure")
    (ws / "bad.py").write_text("import subprocess\n\ndef run():\n    return subprocess.run(['ls'])\n", encoding="utf-8")
    (ws / "test_bad.py").write_text("from bad import run\n\ndef test_run():\n    run()\n", encoding="utf-8")
    rel = [f"wf_test_insecure/{n}" for n in ("bad.py", "test_bad.py")]
    r = qa_agent.review(rel)
    assert not r["passed"] and r["tests"] is None and any("subprocess" in f for f in r["feedback"])


def test_qa_reports_failing_tests():
    ws = coding_agent.workspace("wf_test_fail")
    (ws / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (ws / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n", encoding="utf-8")
    r = qa_agent.review(["wf_test_fail/calc.py", "wf_test_fail/test_calc.py"])
    assert not r["passed"] and r["tests"]["failed"] == 1 and any("test_add" in f for f in r["feedback"])


def test_workspace_path_traversal_blocked():
    with pytest.raises(ValueError):
        coding_agent.workspace("../escape")
    with pytest.raises(PermissionError):
        coding_agent._write(coding_agent.workspace("wf_ok"), "../../evil.py", "x")
