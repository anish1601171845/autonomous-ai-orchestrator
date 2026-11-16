import httpx
import pytest

from app.agents import github_agent
from app.integrations.github_client import GitHubClient, GitHubError, WriteNotPermitted

COMMITS = [{"sha": f"s{i}", "author": {"login": "alice" if i % 3 else "bob"},
            "commit": {"author": {"name": "x", "date": "2026-01-01T00:00:00Z"}, "message": f"change {i}\n\nbody"}} for i in range(5)]


def make_client(handler, **kw) -> GitHubClient:
    return GitHubClient(token="ghp_test", repo="acme/app", base_url="https://api.test", transport=httpx.MockTransport(handler), **kw)


def test_pagination_follows_link_header():
    def h(req: httpx.Request):
        page = int(req.url.params.get("page", "1"))
        headers = {"Link": '<https://api.test/repos/acme/app/commits?page=2>; rel="next"'} if page == 1 else {}
        return httpx.Response(200, json=COMMITS[:3] if page == 1 else COMMITS[3:], headers=headers)
    commits = make_client(h).commits()
    assert len(commits) == 5 and commits[0]["message"] == "change 0" and commits[1]["author"] == "alice"


def test_rate_limit_retry_then_success():
    calls = {"n": 0}

    def h(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(403, headers={"X-RateLimit-Remaining": "0", "Retry-After": "1"}, json={})
        return httpx.Response(200, json={"full_name": "acme/app", "default_branch": "main"})
    assert make_client(h).repository()["default_branch"] == "main" and calls["n"] == 2


def test_rate_limit_too_long_raises():
    h = lambda req: httpx.Response(429, headers={"Retry-After": "9999"}, json={})  # noqa: E731
    with pytest.raises(GitHubError, match="rate limit"):
        make_client(h).repository()


def test_server_errors_retried_and_4xx_raise():
    seq = iter([httpx.Response(502), httpx.Response(200, json=[])])
    assert make_client(lambda r: next(seq)).issues() == []
    with pytest.raises(GitHubError, match="404"):
        make_client(lambda r: httpx.Response(404, text="nope")).repository()


def test_issues_exclude_pull_requests():
    data = [{"number": 1, "title": "bug", "state": "open", "labels": [{"name": "bug"}], "user": {"login": "u"}},
            {"number": 2, "title": "pr", "state": "open", "labels": [], "user": {"login": "u"}, "pull_request": {}}]
    assert [i["number"] for i in make_client(lambda r: httpx.Response(200, json=data)).issues()] == [1]


def test_reviewer_recommendation_uses_file_ownership():
    def h(req):
        if req.url.path.endswith("/commits"):
            return httpx.Response(200, json=COMMITS)
        sha = req.url.path.rsplit("/", 1)[-1]
        files = ["src/login.py"] if sha in ("s0", "s3") else ["README.md"]
        return httpx.Response(200, json={"files": [{"filename": f} for f in files]})
    rec = make_client(h).recommend_reviewers(["login"])
    assert rec[0]["login"] == "bob" and rec[0]["reason"] == "file ownership"


def test_writes_require_approval_and_flag():
    c = make_client(lambda r: httpx.Response(200, json={}), write_enabled=False)
    with pytest.raises(WriteNotPermitted, match="approval"):
        c.create_branch("b", approved=False)
    with pytest.raises(WriteNotPermitted, match="GITHUB_WRITE_ENABLED"):
        c.create_pull_request("b", "t", "x", approved=True)
    no_token = GitHubClient(token="", repo="acme/app", transport=httpx.MockTransport(lambda r: httpx.Response(200)), write_enabled=True)
    with pytest.raises(WriteNotPermitted, match="TOKEN"):
        no_token.put_file("b", "f", "c", "m", approved=True)


def test_approved_pull_request_flow():
    seen = []

    def h(req):
        seen.append((req.method, req.url.path))
        p = req.url.path
        if p == "/repos/acme/app":
            return httpx.Response(200, json={"default_branch": "main"})
        if p.endswith("/git/ref/heads/main"):
            return httpx.Response(200, json={"object": {"sha": "abc"}})
        if p.endswith("/pulls"):
            return httpx.Response(201, json={"number": 7, "html_url": "https://x/pr/7"})
        return httpx.Response(201, json={})
    c = make_client(h, write_enabled=True)
    c.create_branch("feat", approved=True)
    pr = c.create_pull_request("feat", "title", "body", approved=True, reviewers=["bob"])
    assert pr["number"] == 7
    assert ("POST", "/repos/acme/app/git/refs") in seen and ("POST", "/repos/acme/app/pulls/7/requested_reviewers") in seen
    assert not any("merge" in p for _, p in seen)


def test_github_agent_publish_refuses_without_approval():
    assert github_agent.publish("wf", [], "s", approved=False)["performed"] is False
    assert "not configured" in github_agent.publish("wf", [], "s", approved=True)["reason"]
