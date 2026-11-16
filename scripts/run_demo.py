"""Run one end-to-end workflow from the command line and print a summary.

Usage: python scripts/run_demo.py ["your requirement"] [--approve]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.orchestrator import approve, run_workflow  # noqa: E402

DEFAULT = ("Build a student management system with registration, login, database integration, "
           "student dashboard, and automated testing.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("requirement", nargs="?", default=DEFAULT)
    ap.add_argument("--approve", action="store_true", help="approve artifacts after QA (no GitHub write unless configured)")
    a = ap.parse_args()
    s = run_workflow(a.requirement)
    print(f"\nWorkflow {s['workflow_id']} -> {s['status']}")
    print(f"scheduler={s['summary'].get('scheduler')} dependencies={s['summary'].get('dependency_method')}")
    print(f"{'task':5} {'title':38} {'category':14} {'priority':9} {'agent':9} {'status':10} deps")
    for t in s["tasks"]:
        print(f"{t['task_id']:5} {t['title'][:38]:38} {t['category']:14} {t['priority']:9} {str(t['assigned_agent']):9} {t['status']:10} {','.join(t['dependency_ids'])}")
    print("final test suite:", json.dumps({k: v for k, v in (s["summary"].get("final_test_suite") or {}).items() if k != "output_tail"}))
    print("workspace:", s["summary"].get("workspace"))
    if a.approve and s["status"] == "awaiting_approval":
        print("approval:", approve(s["workflow_id"], True, publish_to_github=False))


if __name__ == "__main__":
    main()
