"""Dataset cleaning, validation, feature engineering and leakage-safe splitting."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from app.core.config import CATEGORIES, PRIORITIES

REQUIRED_COLUMNS = ["task_id", "task_description", "task_category", "task_priority", "estimated_complexity",
                    "estimated_duration", "dependency_ids", "assigned_agent", "completion_status",
                    "actual_duration"]
NUMERIC_FEATURES = ["estimated_complexity", "urgency", "business_impact", "num_dependents"]

_URGENT = re.compile(r"\b(?:urgent|blocking|escalation|critical|asap|high-visibility|production|security)\b")
_LOW = re.compile(r"\b(?:nice to have|when time permits|low risk|optional)\b")


def clean_text(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s:/\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validate(df: pd.DataFrame) -> list[str]:
    problems = [f"missing column: {c}" for c in REQUIRED_COLUMNS if c not in df.columns]
    if problems:
        return problems
    bad_cat = set(df["task_category"].dropna()) - set(CATEGORIES)
    bad_pri = set(df["task_priority"].dropna()) - set(PRIORITIES)
    if bad_cat:
        problems.append(f"unknown categories: {sorted(bad_cat)}")
    if bad_pri:
        problems.append(f"unknown priorities: {sorted(bad_pri)}")
    if df["task_id"].duplicated().any():
        problems.append("duplicate task_id values after cleaning")
    return problems


@dataclass
class CleanReport:
    rows_in: int
    duplicates_removed: int
    empty_descriptions_removed: int
    imputed_complexity: int
    rows_out: int


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanReport]:
    n0 = len(df)
    df = df.drop_duplicates().copy()
    dups = n0 - len(df)
    df = df.drop_duplicates(subset="task_id", keep="first")
    dups = n0 - len(df)
    df["clean_text"] = df["task_description"].map(clean_text)
    empty = int((df["clean_text"] == "").sum())
    df = df[df["clean_text"] != ""].copy()
    imputed = int(df["estimated_complexity"].isna().sum())
    # impute complexity with the per-category median (category-level stat, not target leakage for priority;
    # computed on the full set only for this categorical feature that is available at inference time)
    df["estimated_complexity"] = df.groupby("task_category")["estimated_complexity"].transform(
        lambda s: s.fillna(s.median()))
    df["estimated_complexity"] = df["estimated_complexity"].fillna(3).round().astype(int)
    for c in ("urgency", "business_impact", "num_dependents"):
        if c not in df:
            df[c] = 3 if c != "num_dependents" else 0
        df[c] = df[c].fillna(df[c].median())
    return df.reset_index(drop=True), CleanReport(n0, dups, empty, imputed, len(df))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["urgent_cue"] = df["clean_text"].str.contains(_URGENT).astype(int)
    df["low_cue"] = df["clean_text"].str.contains(_LOW).astype(int)
    df["text_len"] = df["clean_text"].str.split().map(len)
    return df


def group_split(df: pd.DataFrame, group_col: str = "project_id", seed: int = 42,
                test_size: float = 0.2, val_size: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by project so near-identical tasks of one project never straddle train/test (leakage control)."""
    groups = df[group_col] if group_col in df else pd.Series(np.arange(len(df)))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trv_idx, te_idx = next(gss.split(df, groups=groups))
    trv, te = df.iloc[trv_idx], df.iloc[te_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    tr_idx, va_idx = next(gss2.split(trv, groups=groups.iloc[trv_idx]))
    return trv.iloc[tr_idx].reset_index(drop=True), trv.iloc[va_idx].reset_index(drop=True), te.reset_index(drop=True)
