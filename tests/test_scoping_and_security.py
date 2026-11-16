import pytest

from app.agents import scoping_agent
from app.core import security
from app.core.config import settings
from tests.conftest import EXAMPLE


def test_scoping_decomposes_example():
    reqs, tasks = scoping_agent.run(EXAMPLE)
    titles = [t.title for t in tasks]
    assert reqs["subject"] == "student management system"
    assert set(reqs["functional"]) == {"registration", "login", "student dashboard"}
    assert reqs["flags"]["database"] and reqs["flags"]["testing"]
    assert "Database schema" in titles and "Integration tests" in titles
    assert {"Implement login", "Test login"} <= set(titles)
    assert len({t.task_id for t in tasks}) == len(tasks)
    assert all(t.acceptance_criteria and 1 <= t.complexity <= 5 for t in tasks)
    assert "security" in reqs["non_functional"]


def test_scoping_rejects_empty():
    with pytest.raises(ValueError):
        scoping_agent.run("hi")


def test_deployment_detected():
    _, tasks = scoping_agent.run("Create an inventory tracking app with product catalog and deploy it to production with docker")
    assert any(t.title == "Deployment" for t in tasks)


@pytest.mark.parametrize("src,needle", [
    ("import subprocess\n", "subprocess"), ("import os\nos.system('ls')\n", "system"),
    ("eval('1')\n", "eval"), ("from socket import socket\n", "socket"), ("def broken(:\n", "syntax"),
])
def test_scan_code_flags_violations(src, needle):
    assert any(needle in i for i in security.scan_code(src))


def test_scan_code_allows_safe_code():
    assert security.scan_code("import hashlib\nimport sqlite3\n\ndef f(x):\n    return x * 2\n") == []


def test_redaction():
    text = "token ghp_" + "a" * 36 + " and Authorization: Bearer sk-abcdefghijklmnopqrstu"
    out = security.redact(text)
    assert "ghp_" not in out and "sk-abc" not in out and "REDACTED" in out


def test_sandbox_runs_and_limits(tmp_path):
    (tmp_path / "ok.py").write_text("print('hello')\n")
    r = security.run_in_sandbox(tmp_path, ["ok.py"])
    assert r.passed and "hello" in r.stdout
    (tmp_path / "env.py").write_text("import os\nprint(sorted(k for k in os.environ if 'TOKEN' in k or 'KEY' in k))\n")
    assert security.run_in_sandbox(tmp_path, ["env.py"]).stdout.strip() == "[]"


def test_sandbox_timeout(tmp_path, monkeypatch):
    object.__setattr__(settings, "sandbox_timeout_s", 2)
    try:
        (tmp_path / "loop.py").write_text("while True:\n    pass\n")
        r = security.run_in_sandbox(tmp_path, ["loop.py"])
        assert not r.passed
    finally:
        object.__setattr__(settings, "sandbox_timeout_s", 60)


def test_sandbox_env_windows_keeps_system_vars_but_no_secrets(monkeypatch, tmp_path):
    monkeypatch.setattr(security.os, "name", "nt")
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    env = security._sandbox_env(tmp_path)
    assert env["SYSTEMROOT"] == r"C:\Windows"  # Windows Python cannot start without it
    assert not any("TOKEN" in k or "KEY" in k for k in env)
