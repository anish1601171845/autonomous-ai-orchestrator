"""Train the models that use the REAL datasets in data/real/.

    python scripts/train_real.py                 # both (tickets take ~5-10 min on a laptop CPU)
    python scripts/train_real.py --only requirements
    python scripts/train_real.py --only tickets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml import real_data, requirements_classifier, ticket_models  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["requirements", "tickets"])
    a = ap.parse_args()
    avail = real_data.available()
    if a.only in (None, "requirements"):
        if not avail["promise"]:
            print("[requirements] skipped: data/real/promise_nfr_train.txt / promise_nfr_test.txt not found")
        else:
            m = requirements_classifier.train()
            for k in ("binary_functional_vs_nfr", "multiclass_requirement_type"):
                x = m[k]
                print(f"[requirements:{k}] {x['selected']} test_acc={x['test_accuracy']} macro_f1={x['test_f1_macro']} "
                      f"(majority baseline {x['majority_baseline_accuracy']})")
    if a.only in (None, "tickets"):
        if not avail["tickets"]:
            print("[tickets] skipped: data/real/tickets_multilang_4k.csv not found")
        else:
            m = ticket_models.train()
            for t, x in m["targets"].items():
                print(f"[tickets:{t}] {x['selected']} test_acc={x['test_accuracy']} macro_f1={x['test_f1_macro']} "
                      f"(majority baseline {x['majority_baseline_accuracy']}) by_language={x['accuracy_by_language']}")


if __name__ == "__main__":
    main()
