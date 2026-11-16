"""Agent 5 — Coding (plus documentation/analysis/deployment artifact producers).

Writes proposed changes into an isolated per-workflow workspace. Generated code is never imported
or executed by this process; execution happens only in the QA agent's sandbox.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from app.agents import code_templates as tpl
from app.core import llm
from app.core.config import settings
from app.core.security import get_logger

log = get_logger(__name__)
NAME = "coding_agent"


def workspace(workflow_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", workflow_id):
        raise ValueError("invalid workflow id")
    p = (settings.workspace_dir / workflow_id).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write(ws: Path, rel: str, content: str) -> str:
    target = (ws / rel).resolve()
    if ws not in target.parents:
        raise PermissionError(f"refusing to write outside workspace: {rel}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target.relative_to(settings.workspace_dir))


def _functions(source: str) -> list[dict]:
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            out.append({"name": node.name, "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                        "doc": (ast.get_docstring(node) or "").split("\n")[0]})
    return out


def _llm_code(task: dict, mod: str, feedback: list[str] | None, memory_context: str | None = None) -> tuple[str, str]:
    system = ("You are a senior Python engineer. Produce a self-contained module using only the Python standard library "
              "(no subprocess, sockets, network, eval/exec, file deletion) and a pytest test file. Reply with exactly two blocks: "
              f"```python {mod}.py``` and ```python test_{mod}.py```. Tests must import from {mod}.")
    user = json.dumps({"task": task["title"], "description": task["description"], "acceptance_criteria": task.get("acceptance_criteria", []),
                       "previous_qa_feedback": feedback or []})
    if memory_context:  # retrieved past solutions, explicitly marked untrusted inside the text
        user += "\n\n" + memory_context
    blocks = llm.extract_code_blocks(llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}]))
    if f"{mod}.py" not in blocks or f"test_{mod}.py" not in blocks:
        raise ValueError("LLM response missing required code blocks")
    return blocks[f"{mod}.py"], blocks[f"test_{mod}.py"]


def generate_code(workflow_id: str, task: dict, feedback: list[str] | None = None, memory_context: str | None = None) -> dict:
    ws = workspace(workflow_id)
    mod = tpl.module_name(task.get("feature", "service"), task["description"])
    backend = "template"
    src = tests = None
    if llm.enabled():
        try:
            src, tests = _llm_code(task, mod, feedback, memory_context)
            backend = "llm"
        except Exception as e:
            log.warning("LLM code generation failed for %s, using template: %s", task["task_id"], e)
    if src is None:
        kind = tpl.archetype(task.get("feature", ""), task["description"])
        src, tests = tpl.render(kind, mod, task.get("feature", ""), task["title"])
        backend = f"template:{kind}"
    files = [_write(ws, f"{mod}.py", src), _write(ws, f"test_{mod}.py", tests)]
    return {"files": files, "module": mod, "backend": backend, "explanation": _functions(src),
            "memory_context_used": bool(memory_context) and backend == "llm",
            "memory_context_supplied": bool(memory_context)}


def generate_analysis(workflow_id: str, task: dict, requirements: dict, tasks: list[dict]) -> dict:
    lines = [f"# Requirements analysis — {requirements.get('subject', '')}", "",
             "## Functional requirements", *[f"- {f}" for f in requirements.get("functional", [])], "",
             "## Non-functional requirements", *([f"- {n}" for n in requirements.get("non_functional", [])] or ["- none stated"]), "",
             "## Acceptance criteria by task"]
    for t in tasks:
        lines += [f"### {t['task_id']} {t['title']}", *[f"- [ ] {c}" for c in t.get("acceptance_criteria", [])], ""]
    return {"files": [_write(workspace(workflow_id), "docs/requirements_analysis.md", "\n".join(lines))], "backend": "template:analysis"}


def generate_docs(workflow_id: str, task: dict, requirements: dict, tasks: list[dict]) -> dict:
    ws = workspace(workflow_id)
    modules = sorted(p.stem for p in ws.glob("*.py") if not p.name.startswith("test_"))
    api = []
    for m in modules:
        api.append(f"### `{m}`")
        api += [f"- **{f['name']}** ({f['kind']}){': ' + f['doc'] if f['doc'] else ''}" for f in _functions((ws / f"{m}.py").read_text(encoding="utf-8"))]
        api.append("")
    text = "\n".join([
        f"# {requirements.get('subject', 'System').title()} — generated documentation", "",
        "Produced by the Autonomous AI Multi-Agent Workflow Orchestrator.", "",
        "## Setup", "```bash", "pip install pytest", "python -m pytest -q", "```", "",
        "## Modules", *(api or ["(no modules generated yet)"]),
        "## Task plan", "| Task | Category | Priority | Depends on |", "|---|---|---|---|",
        *[f"| {t['task_id']} {t['title']} | {t.get('category')} | {t.get('priority')} | {', '.join(t.get('dependency_ids', [])) or '-'} |" for t in tasks],
    ])
    return {"files": [_write(ws, "docs/README.md", text)], "backend": "template:documentation"}


def generate_deployment(workflow_id: str, task: dict) -> dict:
    ws = workspace(workflow_id)
    docker = "FROM python:3.11-slim\nWORKDIR /app\nCOPY . /app\nRUN pip install --no-cache-dir pytest\nRUN python -m pytest -q\nCMD [\"python\", \"-c\", \"print('service image built')\"]\n"
    ci = ("name: ci\non: [push, pull_request]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n"
          "      - uses: actions/setup-python@v5\n        with:\n          python-version: '3.11'\n      - run: pip install pytest\n      - run: python -m pytest -q\n")
    return {"files": [_write(ws, "Dockerfile", docker), _write(ws, ".github/workflows/ci.yml", ci)], "backend": "template:deployment"}


def generate_integration_test(workflow_id: str) -> dict:
    ws = workspace(workflow_id)
    modules = sorted(p.stem for p in ws.glob("*.py") if not p.name.startswith("test_"))
    body = "import importlib\n\nimport pytest\n\nMODULES = " + repr(modules) + "\n\n\n" \
           "@pytest.mark.parametrize('name', MODULES)\ndef test_module_imports_and_exposes_api(name):\n" \
           "    mod = importlib.import_module(name)\n    public = [n for n in dir(mod) if not n.startswith('_')]\n    assert public, f'{name} exposes no public API'\n"
    if not modules:
        body = "def test_no_modules_generated():\n    assert True, 'no modules to integrate'\n"
    return {"files": [_write(ws, "test_integration.py", body)], "backend": "template:integration"}
