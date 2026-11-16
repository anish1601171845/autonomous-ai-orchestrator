"""Scoping for full requirements documents (Agent 1, document mode).

1. Split the document into requirement statements.
2. Classify each with the PROMISE-trained requirements classifier (functional vs 11 NFR types).
3. Cluster functional requirements into features (TF-IDF + KMeans) -> implement + test task per feature,
   with the original requirement sentences as acceptance criteria.
4. Group non-functional requirements by type into cross-cutting tasks (security controls, performance
   tests, usability review, availability set-up, compliance review, ...).
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.agents.scoping_agent import DB_WORDS, DEPLOY_WORDS, URGENT, split_requirements
from app.core.state import Task
from app.ml import requirements_classifier
from app.ml.real_data import REQUIREMENT_TYPES

MAX_FEATURES = 8
DOMAIN_STOP = {"shall", "system", "able", "user", "users", "allow", "provide", "new", "existing", "include", "including",
               "limited", "information", "given", "ability", "application", "product", "data", "display", "displayed",
               "add", "added", "modify", "relating", "specify", "member", "members", "staff", "program", "administrators",
               "administrator", "use", "used", "must", "will", "need", "needs", "based", "following"}
STOP = list(ENGLISH_STOP_WORDS | DOMAIN_STOP)

# NFR type -> (task title, description, task kind used for acceptance criteria, feature, complexity)
NFR_TASKS = {
    "security": (("SE",), "Security controls",
                 "Implement security controls: authentication, role based access control and audit logging", "dev", "access control", 4),
    "operational": (("O",), "External integrations",
                    "Implement integrations with external operational systems and service interfaces", "dev", "external integrations", 4),
    "performance": (("PE", "SC"), "Performance and scalability tests",
                    "Run performance and scalability tests against the stated response time and load targets", "test", "project", 3),
    "usability": (("US", "LF"), "Usability and UI review",
                  "Review the user interface against usability and look and feel requirements and define acceptance criteria", "analysis", "project", 2),
    "availability": (("A", "FT"), "Availability and fault tolerance",
                     "Deploy the complete system to production with monitoring, failover and backups for availability", "deploy", "project", 3),
    "legal": (("L",), "Compliance review",
              "Analyse legal and regulatory compliance requirements for all business rules", "analysis", "project", 2),
    "maintainability": (("MN", "PO"), "Maintainability and portability guide",
                        "Write documentation describing maintainability, configuration and portability guidelines", "doc", "project", 2),
}


def _feature_names(sentences: list[str], k: int) -> tuple[list[int], list[str]]:
    if len(sentences) == 1:
        return [0], [_fallback_name(sentences[0])]
    vec = TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b", sublinear_tf=True)
    X = vec.fit_transform(sentences)
    if X.shape[1] == 0:
        return list(range(len(sentences))), [_fallback_name(s) for s in sentences]
    k = max(1, min(k, len(sentences)))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    terms = np.array(vec.get_feature_names_out())
    names, used = [], set()
    for c in range(k):
        order = np.argsort(-km.cluster_centers_[c])
        cand = [t for t in terms[order[:12]] if len(t) > 2]
        # prefer a bigram of two different words; otherwise join the two strongest distinct unigrams
        bigram = next((t for t in cand if " " in t and len(set(t.split())) == 2), None)
        uni: list[str] = []
        for t in cand:
            if " " not in t and not any(t in u or u in t for u in uni):
                uni.append(t)
        name = bigram or " ".join(uni[:2]) or f"feature {c + 1}"
        base, n = name, 2
        while name in used:
            name = f"{base} {n}"
            n += 1
        used.add(name)
        names.append(name)
    return list(km.labels_), names


def _fallback_name(sentence: str) -> str:
    words = [w for w in re.findall(r"[a-zA-Z]{3,}", sentence.lower()) if w not in STOP]
    return " ".join(words[:2]) or "core feature"


def scope_document(text: str) -> tuple[dict, list[Task]]:
    statements = split_requirements(text)
    preds = requirements_classifier.predict(statements)
    functional = [p["text"] for p in preds if p["is_functional"]]
    nfr: dict[str, list[str]] = defaultdict(list)
    for p in preds:
        if not p["is_functional"]:
            nfr[p["label"]].append(p["text"])
    low = text.lower()
    urgency = 5 if any(w in low for w in URGENT) else 3
    tasks: list[Task] = []

    def add(title: str, desc: str, feature: str, complexity: int, impact: int, criteria: list[str]) -> Task:
        t = Task(task_id=f"T{len(tasks) + 1:02d}", title=title[:120], description=desc, feature=feature,
                 complexity=max(1, min(5, complexity)), urgency=urgency, business_impact=max(1, min(5, impact)),
                 estimated_duration_h=round(complexity * 2.5, 1), acceptance_criteria=criteria)
        tasks.append(t)
        return t

    add("Project setup", "Set up project repository, configuration and dependency management for the system", "project", 2, 4,
        ["Repository, configuration and dependencies are reproducible"])
    if any(re.search(rf"\b{re.escape(w)}\b", low) for w in DB_WORDS) or functional:
        add("Database schema", "Design and implement the relational database schema for the system", "project", 3, 5,
            ["Schema covers every entity referenced by the functional requirements"])
    add("Requirements analysis", "Gather and analyse requirements and acceptance criteria for the system", "project", 2, 3,
        [f"{len(functional)} functional and {sum(map(len, nfr.values()))} non-functional requirements reviewed"])

    features: list[dict] = []
    if functional:
        k = min(MAX_FEATURES, max(1, round(len(functional) / 3)))
        labels, names = _feature_names(functional, k)
        for c, name in enumerate(names):
            members = [s for s, lbl in zip(functional, labels) if lbl == c]
            if not members:
                continue
            cx = min(5, 2 + len(members) // 3 + (1 if any(w in " ".join(members).lower() for w in ("report", "notify", "conflict", "schedule")) else 0))
            impact = 4 if len(members) >= 3 else 3
            add(f"Implement {name}", f"Implement {name} backend service with input validation and error handling", name, cx, impact, members[:8])
            add(f"Test {name}", f"Write unit tests for {name}", name, max(1, cx - 1), impact,
                [f"Automated tests cover: {m[:100]}" for m in members[:5]])
            features.append({"feature": name, "requirements": members})

    for key, (codes, title, desc, _kind, feature, cx) in NFR_TASKS.items():
        items = [s for c in codes for s in nfr.get(c, [])]
        if items:
            add(title, desc, feature, cx + (1 if len(items) > 5 else 0), 4 if key in ("security", "legal", "availability") else 3, items[:8])

    add("Integration tests", "Run end-to-end integration tests across all modules", "project", 3, 4,
        ["All feature modules import and pass their tests together"])
    add("System documentation", "Write the overall system documentation and user guide for the system", "project", 2, 3,
        ["Documentation lists every feature and its source requirements"])
    if any(w in low for w in DEPLOY_WORDS) and not any(nfr.get(c) for c in ("A", "FT")):
        add("Deployment", "Deploy the complete system to production with Docker and a CI/CD pipeline", "project", 3, 4,
            ["Container build file and CI pipeline definition exist"])

    counts = {REQUIREMENT_TYPES[c]: len(v) for c, v in sorted(nfr.items())}
    requirements = {
        "subject": "requirements document", "functional": [f["feature"] for f in features],
        "non_functional": sorted(counts), "flags": {"document_mode": True},
        "backend": "requirements_classifier+kmeans",
        "statements": len(statements), "functional_count": len(functional), "nfr_counts": counts,
        "classified": [{k: p[k] for k in ("text", "label", "label_name", "confidence")} for p in preds],
        "feature_groups": features,
    }
    return requirements, tasks
