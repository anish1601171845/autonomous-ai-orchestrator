"""Minimal, versioned, idempotent schema migrations (works on SQLite and PostgreSQL).

The project originally used `create_all`, which never alters existing tables. Databases created by
earlier versions (e.g. the Docker `orchestrator-var` volume) therefore need explicit migrations.
Each migration runs once and is recorded in `schema_version`.
"""
from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.core.security import get_logger

log = get_logger(__name__)


def _create_all(engine: Engine) -> None:
    from app.database.models import Base

    Base.metadata.create_all(engine)


def _add_task_risk_columns(engine: Engine) -> None:
    cols = {c["name"] for c in inspect(engine).get_columns("tasks")}
    with engine.begin() as conn:
        if "risk_probability" not in cols:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN risk_probability FLOAT"))
        if "risk_category" not in cols:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN risk_category VARCHAR(10)"))


MIGRATIONS = [
    (1, "create core + AegisFlow enhancement tables", _create_all),
    (2, "add tasks.risk_probability / tasks.risk_category", _add_task_risk_columns),
]


def current_version(engine: Engine) -> int:
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, description VARCHAR(200))"))
        v = conn.execute(text("SELECT MAX(version) FROM schema_version")).scalar()
    return int(v or 0)


def migrate(engine: Engine) -> int:
    v = current_version(engine)
    if v and "tasks" in inspect(engine).get_table_names():
        _create_all(engine)  # cheap, idempotent: picks up tables added without a version bump
    for version, desc, fn in MIGRATIONS:
        if version > v:
            fn(engine)
            with engine.begin() as conn:
                conn.execute(text("INSERT INTO schema_version (version, description) VALUES (:v, :d)"), {"v": version, "d": desc})
            log.info("applied migration %s: %s", version, desc)
            v = version
    return v
