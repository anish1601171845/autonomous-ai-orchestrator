"""Collect real GitHub data (commits, issues, PRs) into data/raw/ for analysis or labelling.

Usage: GITHUB_TOKEN=... python scripts/collect_github.py owner/repo [--max 500]
Handles pagination and rate limits via app.integrations.github_client.
Issues are exported with empty task_category/task_priority columns: label them before training
with `python scripts/train_all.py --data <labelled.csv>`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.integrations.github_client import GitHubClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--max", type=int, default=500)
    a = ap.parse_args()
    c = GitHubClient(repo=a.repo)
    out = settings.data_dir / "raw"
    slug = a.repo.replace("/", "__")
    commits = pd.DataFrame(c.commits(max_items=a.max))
    commits.to_csv(out / f"{slug}_commits.csv", index=False)
    issues = pd.DataFrame(c.issues(state="all", max_items=a.max))
    if not issues.empty:
        issues["task_description"] = issues["title"]
        issues["task_category"] = ""
        issues["task_priority"] = ""
    issues.to_csv(out / f"{slug}_issues_unlabelled.csv", index=False)
    pd.DataFrame(c.pull_requests(state="all", max_items=a.max)).to_csv(out / f"{slug}_pulls.csv", index=False)
    print(f"saved {len(commits)} commits, {len(issues)} issues to {out}")


if __name__ == "__main__":
    main()
