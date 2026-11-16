# ---------------------------------------------------------------------------
# Autonomous AI Multi-Agent Workflow Orchestrator — CPU image
# One image, several roles (api | dashboard | train | demo | test), chosen by
# the container command. See docs/DOCKER.md.
# ---------------------------------------------------------------------------

############################ builder ############################
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /build
COPY requirements.txt requirements-docker.txt ./

# 1) CPU-only PyTorch first (the default PyPI wheel bundles ~2 GB of CUDA libraries)
# 2) everything else; torch is already satisfied so it is not reinstalled
RUN pip install --upgrade pip wheel \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-docker.txt --extra-index-url https://download.pytorch.org/whl/cpu

############################ runtime ############################
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="autonomous-ai-orchestrator" \
      org.opencontainers.image.description="LangGraph multi-agent orchestrator with ML triage, GNN dependencies and RL scheduling"

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/var/hf-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    DATABASE_URL=sqlite:////app/var/orchestrator.db \
    WORKSPACE_DIR=/app/workspace \
    API_URL=http://localhost:8000

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --chown=app:app . .

# Directories that compose mounts as named volumes. Creating them here (owned by
# the non-root user) lets Docker seed the volumes with the shipped models/reports
# on first start and keeps them writable.
RUN mkdir -p /app/var /app/workspace /app/models /app/reports /app/data/raw /app/data/processed /app/data/synthetic \
 && chown -R app:app /app \
 && chmod +x /app/docker/entrypoint.sh

USER app
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
  CMD ["python", "/app/docker/healthcheck.py"]

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["api"]
