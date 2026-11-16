"""Convenience entry point.

    python main.py train      # generate synthetic data and train ML + GNN (+ RL with --with-rl)
    python main.py train-real # train on the real datasets in data/real/ (requirements + tickets)
    python main.py evaluate   # AegisFlow baseline experiments (memory, risk, debate, recovery)
    python main.py demo       # run one end-to-end workflow
    python main.py api        # start FastAPI on :8000
    python main.py dashboard  # start Streamlit on :8501
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    cmd, rest = (sys.argv[1] if len(sys.argv) > 1 else "help"), sys.argv[2:]
    py = sys.executable
    targets = {
        "train": [py, str(ROOT / "scripts/train_all.py"), *rest],
        "train-real": [py, str(ROOT / "scripts/train_real.py"), *rest],
        "evaluate": [py, str(ROOT / "scripts/evaluate_enhancements.py"), *rest],
        "demo": [py, str(ROOT / "scripts/run_demo.py"), *rest],
        "api": [py, "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", *rest],
        "dashboard": [py, "-m", "streamlit", "run", str(ROOT / "dashboard/streamlit_app.py"), *rest],
        "test": [py, "-m", "pytest", "-q", *rest],
    }
    if cmd not in targets:
        print(__doc__)
        sys.exit(0 if cmd == "help" else 2)
    sys.exit(subprocess.call(targets[cmd], cwd=ROOT))


if __name__ == "__main__":
    main()
