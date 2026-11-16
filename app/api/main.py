"""FastAPI application. Run: uvicorn app.api.main:app --reload  (docs at /docs)."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import enhancements, ml, real, workflows
from app.core.security import get_logger, redact

log = get_logger("api")

app = FastAPI(title="AegisFlow AI — Self-Healing Multi-Agent Workflow Orchestrator", version="2.0.0",
              description="Scopes, plans, prioritises, schedules, codes and QA-checks enterprise tasks with cooperating agents, "
                          "plus RAG memory, predictive risk, agent debate and self-healing recovery.")
app.include_router(workflows.router)
app.include_router(ml.router)
app.include_router(real.router)
app.include_router(enhancements.router)


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": "http_error", "detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "validation_error",
                                                  "detail": [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]})


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.error("unhandled error: %s", redact(repr(exc)))
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": type(exc).__name__})


@app.middleware("http")
async def access_log(request: Request, call_next):
    resp = await call_next(request)
    log.info("%s %s -> %s", request.method, request.url.path, resp.status_code)
    return resp


@app.get("/", include_in_schema=False)
def root():
    return {"service": app.title, "docs": "/docs", "health": "/api/health"}
