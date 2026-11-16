"""Agent 1 — Requirement Scoping: requirement text -> structured, estimable tasks with acceptance criteria."""
from __future__ import annotations

import re

from app.core import llm
from app.core.security import get_logger
from app.core.state import Task

log = get_logger(__name__)
NAME = "scoping_agent"

NFR_KEYWORDS = {
    "security": ["secure", "security", "encrypt", "authentication", "login", "password", "rbac", "role"],
    "performance": ["fast", "performance", "latency", "real-time", "realtime", "scalab", "high load"],
    "reliability": ["reliab", "availability", "uptime", "fault", "recover"],
    "testability": ["test", "testing", "qa", "coverage"],
    "usability": ["dashboard", "ui", "user friendly", "responsive", "accessib"],
    "compliance": ["gdpr", "hipaa", "audit", "compliance"],
}
HIGH_COMPLEXITY = ["auth", "login", "payment", "billing", "integration", "security", "recommend", "search", "real-time", "machine learning", "analytics"]
LOW_COMPLEXITY = ["export", "profile", "feedback", "notification", "catalog"]
CORE_IMPACT = ["login", "auth", "payment", "registration", "order", "billing", "database", "security"]
URGENT = ["urgent", "asap", "deadline", "critical", "immediately", "blocking"]
DB_WORDS = ["database", "db", "persistence", "storage", "sql"]
TEST_WORDS = ["automated testing", "testing", "tests", "test suite", "qa"]
DEPLOY_WORDS = ["deploy", "deployment", "production", "docker", "ci/cd", "cloud", "kubernetes", "hosting"]
DOC_WORDS = ["documentation", "docs", "readme", "guide"]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _split_features(req: str) -> tuple[str, list[str]]:
    text = re.sub(r"\s+", " ", req.strip().rstrip("."))
    m = re.search(r"\b(?:with|including|that (?:has|supports)|featuring|supporting)\b|:", text, re.I)
    subject, body = (text[: m.start()], text[m.end():]) if m else (text, "")
    subject = re.sub(r"^(please\s+)?(build|create|develop|make|implement|design)\s+(an?|the)?\s*", "", subject, flags=re.I).strip()
    parts = re.split(r",|;|\band\b|\bplus\b|\balso\b", body, flags=re.I)
    feats = []
    for p in parts:
        p = re.sub(r"^\s*(a|an|the|support for|ability to)\s+", "", p.strip(), flags=re.I).strip(" .")
        if 2 < len(p) < 80:
            feats.append(p.lower())
    return subject or "application", feats


def _complexity(text: str) -> int:
    t = text.lower()
    c = 3 + any(k in t for k in HIGH_COMPLEXITY) - any(k in t for k in LOW_COMPLEXITY)
    return max(1, min(5, c + (1 if len(t.split()) > 6 else 0)))


def _criteria(kind: str, feature: str) -> list[str]:
    return {
        "dev": [f"{feature} logic is implemented as an importable module", "Inputs are validated and errors raise clear exceptions",
                f"Unit tests for {feature} pass in the sandbox"],
        "test": [f"Automated tests cover the main and edge cases of {feature}", "Test suite exits with status 0"],
        "analysis": [f"Functional and non-functional requirements for {feature} are written down", "Acceptance criteria agreed"],
        "doc": ["Documentation describes setup, usage and API", "Documentation references every implemented module"],
        "deploy": ["Container build file and CI pipeline definition exist", "Deployment steps are reproducible"],
    }[kind]


def rule_based_scope(requirement: str) -> tuple[dict, list[Task]]:
    req_l = requirement.lower()
    subject, feats = _split_features(requirement)
    nfr = sorted({k for k, words in NFR_KEYWORDS.items() if any(w in req_l for w in words)})
    wants_db = any(re.search(rf"\b{re.escape(w)}\b", req_l) for w in DB_WORDS)
    wants_tests = any(w in req_l for w in TEST_WORDS)
    wants_deploy = any(w in req_l for w in DEPLOY_WORDS)
    wants_docs = any(w in req_l for w in DOC_WORDS)
    urgency = 5 if any(w in req_l for w in URGENT) else 3

    # remove infrastructural phrases from the feature list (they become project-level tasks)
    def infra(f: str) -> bool:
        return any(w in f for w in DB_WORDS + TEST_WORDS + DEPLOY_WORDS + DOC_WORDS)
    features = [f for f in feats if not infra(f)] or [subject]
    tasks: list[Task] = []

    def add(title: str, desc: str, feature: str, kind: str, complexity: int, impact: int) -> Task:
        t = Task(task_id=f"T{len(tasks) + 1:02d}", title=title, description=desc, feature=feature, complexity=complexity,
                 urgency=urgency, business_impact=impact, estimated_duration_h=round(complexity * 2.5, 1),
                 acceptance_criteria=_criteria(kind, feature))
        tasks.append(t)
        return t

    add("Project setup", f"Set up project repository, configuration and dependency management for the {subject}", "project", "dev", 2, 4)
    if wants_db:
        add("Database schema", f"Design and implement the relational database schema for the {subject}", "project", "dev", 3, 5)
    add("Requirements analysis", f"Gather and analyse requirements and acceptance criteria for the {subject}", "project", "analysis", 2, 3)
    for f in features:
        cx = _complexity(f)
        impact = 5 if any(k in f for k in CORE_IMPACT) else 3
        add(f"Implement {f}", f"Implement {f} backend service with input validation and error handling", f, "dev", cx, impact)
        add(f"Test {f}", f"Write unit tests for {f}", f, "test", max(1, cx - 1), impact)
    add("Integration tests", "Run end-to-end integration tests across all modules", "project", "test", 3, 4 if wants_tests else 3)
    add("System documentation", f"Write the overall system documentation and user guide for the {subject}", "project", "doc", 2, 3 if wants_docs else 2)
    if wants_deploy:
        add("Deployment", f"Deploy the complete {subject} to production with Docker and a CI/CD pipeline", "project", "deploy", 3, 4)
    requirements = {"subject": subject, "functional": features, "non_functional": nfr,
                    "flags": {"database": wants_db, "testing": wants_tests, "deployment": wants_deploy, "documentation": wants_docs},
                    "backend": "rule_based"}
    return requirements, tasks


def llm_scope(requirement: str) -> tuple[dict, list[Task]]:
    prompt = (
        "Decompose the software requirement into 6-20 engineering tasks. Return ONLY JSON: "
        '{"subject": str, "functional": [str], "non_functional": [str], "tasks": [{"title": str, "description": str, '
        '"feature": str (use "project" for cross-cutting tasks), "complexity": 1-5, "urgency": 1-5, "business_impact": 1-5, '
        '"acceptance_criteria": [str]}]}. Descriptions must start with a verb, e.g. "Implement ...", "Write unit tests for ...".'
    )
    data = llm.extract_json(llm.chat([{"role": "system", "content": prompt}, {"role": "user", "content": requirement}]))
    tasks = []
    for i, t in enumerate(data["tasks"], 1):
        cx = int(t.get("complexity", 3))
        tasks.append(Task(task_id=f"T{i:02d}", title=t["title"][:120], description=t["description"], feature=t.get("feature", "project").lower(),
                          complexity=cx, urgency=int(t.get("urgency", 3)), business_impact=int(t.get("business_impact", 3)),
                          estimated_duration_h=round(cx * 2.5, 1), acceptance_criteria=t.get("acceptance_criteria", [])))
    req = {k: data.get(k) for k in ("subject", "functional", "non_functional")} | {"backend": "llm"}
    return req, tasks


def run(requirement: str) -> tuple[dict, list[Task]]:
    if not requirement or len(requirement.strip()) < 10:
        raise ValueError("requirement must be at least 10 characters")
    if is_requirements_document(requirement):
        try:
            from app.agents import requirements_scoping

            return requirements_scoping.scope_document(requirement)
        except FileNotFoundError as e:  # classifier not trained -> fall back
            log.warning("requirements classifier unavailable (%s); using rule-based scoping", e)
    if llm.enabled():
        try:
            return llm_scope(requirement)
        except Exception as e:  # recover: fall back to deterministic scoping
            log.warning("LLM scoping failed, falling back to rule-based: %s", e)
    return rule_based_scope(requirement)


_REQ_SENTENCE = re.compile(r"\b(shall|must|should|will be able to|is able to)\b", re.I)


def split_requirements(text: str) -> list[str]:
    """Split a requirements document into individual requirement statements.
    Accepts one requirement per line (optionally prefixed like 'SE:' as in PROMISE) or running prose."""
    lines = [ln.strip(" -*\t") for ln in text.replace("\r", "").split("\n") if ln.strip()]
    if len(lines) < 2:
        lines = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    out = []
    for ln in lines:
        ln = re.sub(r"^\s*(?:[A-Z]{1,2}|\d+[.)])\s*:\s*", "", ln).strip()  # drop 'SE:' or '12:' prefixes
        ln = re.sub(r"^\s*\d+[.)]\s+", "", ln).strip()
        if len(ln) > 12:
            out.append(ln)
    return out


def is_requirements_document(text: str) -> bool:
    """A requirements document = 3+ statements that use requirement modal verbs (shall/must/...)."""
    stmts = split_requirements(text)
    return len(stmts) >= 3 and sum(bool(_REQ_SENTENCE.search(s)) for s in stmts) >= 3
