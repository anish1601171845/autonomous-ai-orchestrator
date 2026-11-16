"""GitHub REST API client: authenticated, paginated, rate-limit aware, retrying, read-only by default."""
from __future__ import annotations

import base64
import time
from collections import Counter
from typing import Any

import httpx

from app.core.config import settings
from app.core.security import get_logger, redact

log = get_logger(__name__)


class GitHubError(RuntimeError):
    pass


class WriteNotPermitted(GitHubError):
    pass


class GitHubClient:
    def __init__(self, token: str | None = None, repo: str | None = None, base_url: str | None = None,
                 transport: httpx.BaseTransport | None = None, write_enabled: bool | None = None,
                 max_retries: int = 3, max_rate_wait_s: int = 60):
        self.repo = repo or settings.github_repo
        self.write_enabled = settings.github_write_enabled if write_enabled is None else write_enabled
        self.max_retries, self.max_rate_wait_s = max_retries, max_rate_wait_s
        token = token if token is not None else settings.github_token
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "autonomous-ai-orchestrator"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.authenticated = bool(token)
        self._http = httpx.Client(base_url=base_url or settings.github_api_url, headers=headers, timeout=20, transport=transport)

    # ------------------------------------------------------------ core
    def _request(self, method: str, url: str, **kw) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                r = self._http.request(method, url, **kw)
            except httpx.TimeoutException as e:
                if attempt == self.max_retries:
                    raise GitHubError(f"timeout calling GitHub: {url}") from e
                time.sleep(min(2 ** attempt, 10))
                continue
            if r.status_code in (403, 429) and (r.headers.get("X-RateLimit-Remaining") == "0" or "retry-after" in r.headers):
                wait = int(r.headers.get("Retry-After") or max(0, int(r.headers.get("X-RateLimit-Reset", "0")) - int(time.time())) or 1)
                if wait > self.max_rate_wait_s or attempt == self.max_retries:
                    raise GitHubError(f"GitHub rate limit exceeded; resets in {wait}s")
                log.warning("GitHub rate limited, sleeping %ss", wait)
                time.sleep(wait)
                continue
            if r.status_code >= 500 and attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 10))
                continue
            if r.status_code >= 400:
                raise GitHubError(f"GitHub {r.status_code} for {method} {url}: {redact(r.text[:300])}")
            return r
        raise GitHubError("unreachable")

    def _paginate(self, url: str, params: dict | None = None, max_items: int = 300) -> list[dict]:
        items: list[dict] = []
        params = {"per_page": 100, **(params or {})}
        next_url: str | None = url
        while next_url and len(items) < max_items:
            r = self._request("GET", next_url, params=params)
            data = r.json()
            items += data if isinstance(data, list) else data.get("items", [])
            next_url = r.links.get("next", {}).get("url")
            params = None  # the next link already carries query params
        return items[:max_items]

    def _repo(self, repo: str | None) -> str:
        repo = repo or self.repo
        if not repo or "/" not in repo:
            raise GitHubError("repository must be configured as 'owner/name' (GITHUB_REPO)")
        return repo

    # ------------------------------------------------------------ read
    def repository(self, repo: str | None = None) -> dict:
        d = self._request("GET", f"/repos/{self._repo(repo)}").json()
        return {k: d.get(k) for k in ("full_name", "description", "default_branch", "stargazers_count", "forks_count",
                                       "open_issues_count", "language", "html_url", "pushed_at")}

    def commits(self, repo: str | None = None, max_items: int = 100, since: str | None = None) -> list[dict]:
        raw = self._paginate(f"/repos/{self._repo(repo)}/commits", {"since": since} if since else None, max_items)
        return [{"sha": c["sha"], "author": (c.get("author") or {}).get("login") or c["commit"]["author"]["name"],
                 "date": c["commit"]["author"]["date"], "message": c["commit"]["message"].split("\n")[0]} for c in raw]

    def commit_files(self, sha: str, repo: str | None = None) -> list[str]:
        return [f["filename"] for f in self._request("GET", f"/repos/{self._repo(repo)}/commits/{sha}").json().get("files", [])]

    def issues(self, repo: str | None = None, state: str = "open", max_items: int = 100) -> list[dict]:
        raw = self._paginate(f"/repos/{self._repo(repo)}/issues", {"state": state}, max_items)
        return [{"number": i["number"], "title": i["title"], "state": i["state"], "labels": [lb["name"] for lb in i.get("labels", [])],
                 "user": i["user"]["login"]} for i in raw if "pull_request" not in i]

    def pull_requests(self, repo: str | None = None, state: str = "open", max_items: int = 100) -> list[dict]:
        raw = self._paginate(f"/repos/{self._repo(repo)}/pulls", {"state": state}, max_items)
        return [{"number": p["number"], "title": p["title"], "state": p["state"], "user": p["user"]["login"],
                 "head": p["head"]["ref"], "html_url": p["html_url"]} for p in raw]

    def recommend_reviewers(self, paths_or_keywords: list[str], repo: str | None = None, sample_commits: int = 30,
                            exclude: list[str] | None = None, top_k: int = 3) -> list[dict]:
        """Rank reviewers by how often they recently touched files matching the given paths/keywords,
        falling back to overall recent commit activity."""
        commits = self.commits(repo, max_items=sample_commits)
        score: Counter = Counter()
        activity: Counter = Counter(c["author"] for c in commits)
        keys = [k.lower() for k in paths_or_keywords if k]
        for c in commits[:sample_commits]:
            files = self.commit_files(c["sha"], repo)
            hits = sum(1 for f in files for k in keys if k in f.lower())
            if hits:
                score[c["author"]] += hits
        excluded = set(exclude or [])
        ranked = [(a, s, "file ownership") for a, s in score.most_common() if a not in excluded]
        ranked += [(a, 0, "recent activity") for a, _ in activity.most_common() if a not in excluded and a not in score]
        return [{"login": a, "score": s, "commits_in_sample": activity[a], "reason": why} for a, s, why in ranked[:top_k]]

    # ------------------------------------------------------------ write (approval gated)
    def _require_write(self, approved: bool) -> None:
        if not approved:
            raise WriteNotPermitted("repository modification requires explicit user approval")
        if not self.write_enabled:
            raise WriteNotPermitted("GITHUB_WRITE_ENABLED is false")
        if not self.authenticated:
            raise WriteNotPermitted("GITHUB_TOKEN not configured")

    def create_branch(self, branch: str, approved: bool, repo: str | None = None, base: str | None = None) -> dict:
        self._require_write(approved)
        repo = self._repo(repo)
        base = base or self.repository(repo)["default_branch"]
        sha = self._request("GET", f"/repos/{repo}/git/ref/heads/{base}").json()["object"]["sha"]
        return self._request("POST", f"/repos/{repo}/git/refs", json={"ref": f"refs/heads/{branch}", "sha": sha}).json()

    def put_file(self, branch: str, path: str, content: str, message: str, approved: bool, repo: str | None = None) -> dict:
        self._require_write(approved)
        body: dict[str, Any] = {"message": message, "branch": branch, "content": base64.b64encode(content.encode()).decode()}
        return self._request("PUT", f"/repos/{self._repo(repo)}/contents/{path}", json=body).json()

    def create_pull_request(self, branch: str, title: str, body: str, approved: bool, repo: str | None = None,
                            base: str | None = None, reviewers: list[str] | None = None) -> dict:
        self._require_write(approved)
        repo = self._repo(repo)
        base = base or self.repository(repo)["default_branch"]
        pr = self._request("POST", f"/repos/{repo}/pulls", json={"title": title, "head": branch, "base": base, "body": body}).json()
        if reviewers:
            try:
                self._request("POST", f"/repos/{repo}/pulls/{pr['number']}/requested_reviewers", json={"reviewers": reviewers})
            except GitHubError as e:
                log.warning("could not request reviewers: %s", e)
        return {"number": pr["number"], "html_url": pr["html_url"]}
    # Merging is deliberately not implemented: merges stay a human decision.
