#!/bin/sh
# Container entrypoint: picks the role and makes sure trained models exist.
set -e
cd /app

ensure_models() {
  if [ ! -f models/task_classifier.joblib ] || [ ! -f models/priority_model.joblib ] || [ ! -f models/semantic_tfidf_lsa.joblib ]; then
    echo "[entrypoint] trained models not found - training on the synthetic dataset (takes a few minutes)..."
    python scripts/train_all.py
    python scripts/make_report.py
  fi
}

role="${1:-api}"
[ $# -gt 0 ] && shift

case "$role" in
  api)
    ensure_models
    exec uvicorn app.api.main:app --host 0.0.0.0 --port "${API_PORT:-8000}" --workers "${API_WORKERS:-1}" "$@"
    ;;
  dashboard)
    exec streamlit run dashboard/streamlit_app.py --server.port "${DASHBOARD_PORT:-8501}" \
         --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false "$@"
    ;;
  train)
    python scripts/train_all.py "$@"
    exec python scripts/make_report.py
    ;;
  train-real)
    exec python scripts/train_real.py "$@"
    ;;
  evaluate)
    exec python scripts/evaluate_enhancements.py "$@"
    ;;
  train-rl)
    exec python -m app.rl.train "$@"
    ;;
  demo)
    ensure_models
    exec python scripts/run_demo.py "$@"
    ;;
  test)
    exec python -m pytest -q -p no:cacheprovider "$@"
    ;;
  shell)
    exec /bin/sh "$@"
    ;;
  *)
    exec "$role" "$@"
    ;;
esac
