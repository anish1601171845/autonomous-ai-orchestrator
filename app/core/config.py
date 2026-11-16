"""Central configuration loaded from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"
    reports_dir: Path = ROOT_DIR / "reports"
    workspace_dir: Path = Path(os.getenv("WORKSPACE_DIR") or str(ROOT_DIR / "workspace")).resolve()

    database_url: str = os.getenv("DATABASE_URL") or f"sqlite:///{ROOT_DIR / 'orchestrator.db'}"

    # LLM (OpenAI-compatible). Empty key => deterministic template backend.
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "60"))

    # Semantic encoder: a local path or HF model id. Falls back to TF-IDF/LSA if unavailable.
    semantic_model: str = os.getenv("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    semantic_backend: str = os.getenv("SEMANTIC_BACKEND", "tfidf")  # tfidf | auto | transformer (retrain GNN after switching)

    # GitHub (read-only by default)
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    github_repo: str = os.getenv("GITHUB_REPO", "")  # owner/name
    github_api_url: str = os.getenv("GITHUB_API_URL", "https://api.github.com")
    github_write_enabled: bool = _bool("GITHUB_WRITE_ENABLED", False)

    # Sandbox
    allow_sandbox_execution: bool = _bool("ALLOW_SANDBOX_EXECUTION", True)
    sandbox_timeout_s: int = int(os.getenv("SANDBOX_TIMEOUT_S", "60"))
    sandbox_memory_mb: int = int(os.getenv("SANDBOX_MEMORY_MB", "1024"))
    max_qa_retries: int = int(os.getenv("MAX_QA_RETRIES", "2"))

    # ---- AegisFlow enhancements (each independently configurable)
    memory_enabled: bool = _bool("MEMORY_ENABLED", True)
    memory_backend: str = os.getenv("MEMORY_BACKEND", "auto")  # auto | embedding | tfidf
    memory_embedding_model: str = os.getenv("MEMORY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    memory_top_k: int = int(os.getenv("MEMORY_TOP_K", "3"))
    memory_min_score: float = float(os.getenv("MEMORY_MIN_SCORE", "0.25"))
    memory_project: str = os.getenv("MEMORY_PROJECT", "default")

    risk_enabled: bool = _bool("RISK_ENABLED", True)
    risk_medium_threshold: float = float(os.getenv("RISK_MEDIUM_THRESHOLD", "0.15"))
    risk_high_threshold: float = float(os.getenv("RISK_HIGH_THRESHOLD", "0.35"))
    task_deadline_s_per_complexity: float = float(os.getenv("TASK_DEADLINE_S_PER_COMPLEXITY", "8"))

    debate_enabled: bool = _bool("DEBATE_ENABLED", True)
    debate_max_rounds: int = min(3, int(os.getenv("DEBATE_MAX_ROUNDS", "3")))
    debate_time_budget_s: float = float(os.getenv("DEBATE_TIME_BUDGET_S", "60"))
    debate_token_budget: int = int(os.getenv("DEBATE_TOKEN_BUDGET", "12000"))
    llm_cost_per_1k_tokens: float = float(os.getenv("LLM_COST_PER_1K_TOKENS", "0.0006"))

    recovery_mode: str = os.getenv("RECOVERY_MODE", "full")  # none | fixed_retry | checkpoint | full
    retry_max_attempts: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    retry_base_delay_s: float = float(os.getenv("RETRY_BASE_DELAY_S", "0.5"))
    retry_max_delay_s: float = float(os.getenv("RETRY_MAX_DELAY_S", "8"))
    max_workflow_resumes: int = int(os.getenv("MAX_WORKFLOW_RESUMES", "2"))

    scheduler: str = os.getenv("SCHEDULER", "auto")  # auto | rl | least_loaded | round_robin
    api_url: str = os.getenv("API_URL", "http://localhost:8000")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    agent_pool: list[dict] = field(
        default_factory=lambda: [
            {"name": "coder-1", "role": "coding", "capabilities": ["development", "testing"], "skill": 1.2},
            {"name": "coder-2", "role": "coding", "capabilities": ["development", "deployment"], "skill": 1.0},
            {"name": "qa-1", "role": "qa", "capabilities": ["testing", "analysis"], "skill": 1.1},
            {"name": "docs-1", "role": "documentation", "capabilities": ["documentation", "analysis"], "skill": 1.0},
            {"name": "devops-1", "role": "devops", "capabilities": ["deployment", "development"], "skill": 0.9},
        ]
    )


settings = Settings()
for _d in (settings.models_dir, settings.reports_dir, settings.workspace_dir, settings.data_dir / "synthetic",
           settings.data_dir / "processed", settings.data_dir / "raw"):
    _d.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["development", "testing", "documentation", "deployment", "analysis"]
PRIORITIES = ["low", "medium", "high", "critical"]
