"""Loaders for the two REAL labelled datasets shipped in data/real/.

1. PROMISE NFR requirements dataset (promise_nfr_train.txt / promise_nfr_test.txt)
   Lines look like "SE:Only users with Update access must be able to initiate dispute requests."
   Labels: F = functional, plus 11 non-functional requirement (NFR) types.
2. Multilingual customer-support tickets (tickets_multilang_4k.csv)
   4,000 tickets in en/de/es/fr/pt with type, priority, queue, business_type and tags.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.core.config import settings

REAL_DIR = settings.data_dir / "real"
PROMISE_TRAIN = REAL_DIR / "promise_nfr_train.txt"
PROMISE_TEST = REAL_DIR / "promise_nfr_test.txt"
TICKETS_CSV = REAL_DIR / "tickets_multilang_4k.csv"

REQUIREMENT_TYPES = {
    "F": "Functional",
    "A": "Availability",
    "FT": "Fault tolerance",
    "L": "Legal",
    "LF": "Look and feel",
    "MN": "Maintainability",
    "O": "Operational",
    "PE": "Performance",
    "PO": "Portability",
    "SC": "Scalability",
    "SE": "Security",
    "US": "Usability",
}
_LINE = re.compile(r"^\s*([A-Z]{1,2})\s*:\s*(.+?)\s*$")


def clean_requirement(text: str) -> str:
    text = text.replace("�", "'").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def load_promise(path: Path) -> pd.DataFrame:
    rows = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LINE.match(raw)
        if m and m.group(1) in REQUIREMENT_TYPES:
            rows.append({"label": m.group(1), "text": clean_requirement(m.group(2))})
    df = pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)
    df["is_functional"] = (df["label"] == "F").astype(int)
    return df


def ticket_text(subject, body) -> str:
    s = "" if subject is None or (isinstance(subject, float) and pd.isna(subject)) else str(subject)
    b = "" if body is None or (isinstance(body, float) and pd.isna(body)) else str(body)
    text = f"{s}\n{b}".strip()
    return re.sub(r"<(name|acc_num|tel_num|email)>", " ", text)


def load_tickets(path: Path = TICKETS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = [ticket_text(s, b) for s, b in zip(df["subject"], df["body"])]
    df = df[df["text"].str.len() > 10].drop_duplicates("text").reset_index(drop=True)
    df["business_type"] = df["business_type"].fillna("unknown").str.strip("_ ").replace({"Pit Services": "IT Services"})
    df["language"] = df["language"].fillna("unknown")
    return df


def available() -> dict[str, bool]:
    return {"promise": PROMISE_TRAIN.exists() and PROMISE_TEST.exists(), "tickets": TICKETS_CSV.exists()}
