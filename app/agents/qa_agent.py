"""Agent 6 — Quality Assurance: static security scan, lint, sandboxed pytest, structured feedback."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from app.core.config import settings
from app.core.security import run_in_sandbox, scan_code

NAME = "qa_agent"


def _ruff(paths: list[Path]) -> list[str]:
    exe = shutil.which("ruff")
    if not exe or not paths:
        return []
    try:
        r = subprocess.run([exe, "check", "--select", "E9,F", "--output-format", "concise", *map(str, paths)],
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return ["ruff timed out"]
    return [ln.split("/")[-1] for ln in r.stdout.splitlines() if re.search(r":\d+:\d+:", ln)]


def _summary(stdout: str) -> dict:
    m = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", stdout)}
    return {"passed": m.get("passed", 0), "failed": m.get("failed", 0), "errors": m.get("error", 0) + m.get("errors", 0)}


def review(files: list[str], test_targets: list[str] | None = None) -> dict:
    """Review code files (paths relative to the workspace root) and run their tests in the sandbox."""
    paths = [settings.workspace_dir / f for f in files]
    py = [p for p in paths if p.suffix == ".py"]
    if not py:
        return {"passed": True, "kind": "non_code", "security_issues": [], "lint_issues": [], "tests": None, "feedback": []}
    workdir = py[0].parent
    security = [f"{p.name}: {i}" for p in py for i in scan_code(p.read_text(encoding="utf-8"))]
    lint = _ruff(py)
    feedback = [*security, *lint]
    tests = None
    if security:
        feedback.append("tests not executed: security policy violations must be fixed first")
    else:
        targets = test_targets or [p.name for p in py if p.name.startswith("test_")]
        res = run_in_sandbox(workdir, ["-m", "pytest", "-q", "-p", "no:cacheprovider", *targets])
        tests = {**_summary(res.stdout), "returncode": res.returncode, "timed_out": res.timed_out,
                 "output_tail": res.stdout[-1500:] + res.stderr[-500:]}
        if not res.passed:
            fails = re.findall(r"^(?:FAILED|ERROR) (.+)$", res.stdout, re.M)
            feedback += [f"test failure: {f}" for f in fails[:10]] or [f"pytest exited with {res.returncode}: {res.stderr[-300:]}"]
    passed = not security and not lint and tests is not None and tests["returncode"] == 0
    return {"passed": passed, "kind": "code", "security_issues": security, "lint_issues": lint, "tests": tests, "feedback": feedback}


def run_suite(workflow_dir: Path, targets: list[str] | None = None) -> dict:
    files = [str(p.relative_to(settings.workspace_dir)) for p in sorted(workflow_dir.glob("*.py"))]
    tests = targets or [Path(f).name for f in files if Path(f).name.startswith("test_")]
    return review(files, tests)
