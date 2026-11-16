"""Container healthcheck (no curl in python:slim). Checks the API or the dashboard depending on the role."""
import os
import sys
import urllib.request

role = os.environ.get("HEALTHCHECK_ROLE", "api")
url = (f"http://127.0.0.1:{os.environ.get('DASHBOARD_PORT', '8501')}/_stcore/health" if role == "dashboard"
       else f"http://127.0.0.1:{os.environ.get('API_PORT', '8000')}/api/health")
try:
    with urllib.request.urlopen(url, timeout=4) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
