"""Agent 8 — GitHub Integration: repo insight, reviewer recommendation, approval-gated branch + PR."""
from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.integrations.github_client import GitHubClient, GitHubError, WriteNotPermitted

NAME = "github_agent"


def status() -> dict:
    return {"configured": bool(settings.github_repo), "authenticated": bool(settings.github_token),
            "repo": settings.github_repo or None, "write_enabled": settings.github_write_enabled}


def overview(client: GitHubClient | None = None) -> dict:
    if not settings.github_repo and client is None:
        return {"available": False, "reason": "GITHUB_REPO not configured", **status()}
    c = client or GitHubClient()
    try:
        return {"available": True, "repository": c.repository(), "issues": c.issues(max_items=30),
                "pull_requests": c.pull_requests(max_items=30), "recent_commits": c.commits(max_items=30)}
    except GitHubError as e:
        return {"available": False, "reason": str(e), **status()}


def publish(workflow_id: str, files: list[str], summary: str, approved: bool, client: GitHubClient | None = None) -> dict:
    """Create a branch, commit generated artifacts, open a PR. Never merges. Requires approval."""
    if not approved:
        return {"performed": False, "reason": "not approved"}
    if not settings.github_repo and client is None:
        return {"performed": False, "reason": "GITHUB_REPO not configured; artifacts kept locally"}
    c = client or GitHubClient()
    branch = f"orchestrator/{workflow_id}"
    try:
        keys = sorted({Path(f).stem.replace("test_", "") for f in files})
        reviewers = [r["login"] for r in c.recommend_reviewers(keys, sample_commits=15)]
        c.create_branch(branch, approved=True)
        for f in files:
            local = settings.workspace_dir / f
            c.put_file(branch, f"generated/{Path(f).relative_to(workflow_id)}", local.read_text(encoding="utf-8"),
                       f"orchestrator: add {Path(f).name}", approved=True)
        pr = c.create_pull_request(branch, f"[orchestrator] {workflow_id}", summary, approved=True, reviewers=reviewers)
        return {"performed": True, "branch": branch, "pull_request": pr, "reviewers_requested": reviewers}
    except (WriteNotPermitted, GitHubError) as e:
        return {"performed": False, "reason": str(e)}
