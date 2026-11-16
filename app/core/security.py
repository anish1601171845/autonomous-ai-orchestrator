"""Security utilities: secret redaction, static code safety scanning, sandboxed execution."""
from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings

_SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(authorization:\s*(bearer|token)\s+)\S+"),
]


def redact(text: str) -> str:
    """Remove anything that looks like a credential from text bound for logs/reports."""
    if not text:
        return text
    for secret in (settings.github_token, settings.llm_api_key):
        if secret:
            text = text.replace(secret, "***REDACTED***")
    for pat in _SECRET_PATTERNS:
        text = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "***REDACTED***", text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.msg))
        if record.args:
            record.args = tuple(redact(str(a)) for a in record.args)
        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        h.addFilter(RedactingFilter())
        logger.addHandler(h)
        logger.setLevel(settings.log_level)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------- static scan
FORBIDDEN_MODULES = {"subprocess", "socket", "shutil", "ctypes", "multiprocessing", "requests",
                     "urllib", "http", "httpx", "ftplib", "telnetlib", "smtplib", "pickle", "marshal"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
FORBIDDEN_ATTRS = {"system", "popen", "spawn", "fork", "kill", "remove", "rmdir", "unlink", "rmtree", "execv"}


def scan_code(source: str) -> list[str]:
    """Return a list of policy violations found in generated Python source."""
    issues: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_MODULES:
                    issues.append(f"forbidden import '{a.name}' (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_MODULES:
                issues.append(f"forbidden import '{node.module}' (line {node.lineno})")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                issues.append(f"forbidden call '{f.id}()' (line {node.lineno})")
            if isinstance(f, ast.Attribute) and f.attr in FORBIDDEN_ATTRS:
                issues.append(f"forbidden call '.{f.attr}()' (line {node.lineno})")
    return issues


# ---------------------------------------------------------------- sandbox
@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _limits() -> None:  # runs in the child before exec (POSIX only)
    import resource

    mem = settings.sandbox_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_CPU, (settings.sandbox_timeout_s, settings.sandbox_timeout_s))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
    os.setsid()


_WINDOWS_REQUIRED_ENV = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
                         "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")


def _sandbox_env(workdir: Path) -> dict[str, str]:
    """Minimal, secret-free environment. Windows Python cannot start without SYSTEMROOT & co,
    so those (non-secret) variables are passed through; tokens/keys never are."""
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(workdir), "NO_PROXY": "*",
           "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
           "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9"}
    if os.name == "nt":
        env.update({k: os.environ[k] for k in _WINDOWS_REQUIRED_ENV if k in os.environ})
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("SYSTEMROOT", r"C:\Windows") + r"\System32"
        env["USERPROFILE"] = str(workdir)
    else:
        env["PATH"] = "/usr/bin:/bin"
        env["HOME"] = str(workdir)
    return env


def run_in_sandbox(workdir: Path, args: list[str]) -> SandboxResult:
    """Run `python -I <args>` inside workdir with a scrubbed env, rlimits and a timeout.

    This is process-level isolation (no secrets, CPU/memory/file-size limits, isolated
    interpreter mode, separate session). For production use run it inside a container
    without network access — see docs/ARCHITECTURE.md.
    """
    if not settings.allow_sandbox_execution:
        return SandboxResult(1, "", "sandbox execution disabled by ALLOW_SANDBOX_EXECUTION=false", False)
    env = _sandbox_env(workdir)
    kwargs: dict = {}
    if os.name == "posix":
        kwargs["preexec_fn"] = _limits
    else:  # Windows: own process group so a timeout can't take the parent down; no rlimits available
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-B", "-X", "utf8", *args], cwd=workdir, env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=settings.sandbox_timeout_s, **kwargs,
        )
        return SandboxResult(proc.returncode, redact(proc.stdout[-8000:]), redact(proc.stderr[-4000:]), False)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return SandboxResult(124, redact(out[-4000:]), "timeout", True)
