"""GraphSAGE directed link-prediction model for task dependencies (PyTorch Geometric),
a NetworkX rule baseline, and DAG validation.

Message-passing graph at inference = semantic kNN edges + same-feature edges (both known before any
dependency exists). The model then scores every ordered pair (u -> v, "u must finish before v").
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from itertools import permutations

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from app.core.config import CATEGORIES, settings
from app.ml.semantic_mapping import SemanticEncoder, get_encoder

CKPT = settings.models_dir / "gnn_dependency.pt"
METRICS_PATH = settings.reports_dir / "gnn_metrics.json"
STAGE = {"analysis": 0, "development": 1, "testing": 2, "documentation": 3, "deployment": 4}


# ------------------------------------------------------------------ features
def node_features(tasks: list[dict], emb: np.ndarray) -> torch.Tensor:
    cat = np.zeros((len(tasks), len(CATEGORIES)), dtype=np.float32)
    for i, t in enumerate(tasks):
        if t.get("category") in CATEGORIES:
            cat[i, CATEGORIES.index(t["category"])] = 1
    proj = np.array([[1.0 if t.get("feature") == "project" else 0.0] for t in tasks], dtype=np.float32)
    cx = np.array([[float(t.get("complexity", 3)) / 5] for t in tasks], dtype=np.float32)
    return torch.tensor(np.hstack([emb.astype(np.float32), cat, proj, cx]))


def message_edges(tasks: list[dict], emb: np.ndarray, k: int = 3) -> torch.Tensor:
    n = len(tasks)
    pairs = set()
    if n > 1:
        S = emb @ emb.T
        np.fill_diagonal(S, -1)
        for i in range(n):
            for j in np.argsort(-S[i])[: min(k, n - 1)]:
                pairs.add((i, int(j)))
                pairs.add((int(j), i))
    for i in range(n):
        for j in range(n):
            if i != j and tasks[i].get("feature") == tasks[j].get("feature"):
                pairs.add((i, j))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()


class DependencyGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.c1, self.c2 = SAGEConv(in_dim, hidden), SAGEConv(hidden, hidden)
        self.skip = nn.Linear(in_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 3 + 2, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def encode(self, x, ei):
        h = F.relu(self.c1(x, ei)) if ei.numel() else F.relu(self.skip(x))
        h = self.c2(h, ei) if ei.numel() else h
        return h + self.skip(x)

    def forward(self, x, ei, pairs, pair_feats):
        h = self.encode(x, ei)
        hu, hv = h[pairs[0]], h[pairs[1]]
        return self.head(torch.cat([hu, hv, hu * hv, pair_feats], dim=1)).squeeze(-1)


def _pair_feats(emb: np.ndarray, pairs: torch.Tensor, tasks: list[dict]) -> torch.Tensor:
    u, v = pairs[0].numpy(), pairs[1].numpy()
    sim = (emb[u] * emb[v]).sum(1)
    same = np.array([tasks[a].get("feature") == tasks[b].get("feature") for a, b in zip(u, v)], dtype=np.float32)
    return torch.tensor(np.stack([sim, same], 1), dtype=torch.float32)


# ------------------------------------------------------------------ datasets
@dataclass
class GraphSample:
    data: Data
    pairs: torch.Tensor
    pair_feats: torch.Tensor
    labels: torch.Tensor
    tasks: list[dict]
    project_id: str = ""


def build_sample(tasks: list[dict], edges: set[tuple[str, str]] | None, encoder: SemanticEncoder) -> GraphSample:
    emb = encoder.encode([t["description"] for t in tasks])
    x, ei = node_features(tasks, emb), message_edges(tasks, emb)
    idx = range(len(tasks))
    pairs = torch.tensor(list(permutations(idx, 2)), dtype=torch.long).t() if len(tasks) > 1 else torch.empty((2, 0), dtype=torch.long)
    ids = [t["task_id"] for t in tasks]
    labels = torch.tensor([float((ids[a], ids[b]) in (edges or set())) for a, b in pairs.t().tolist()])
    return GraphSample(Data(x=x, edge_index=ei), pairs, _pair_feats(emb, pairs, tasks), labels, tasks)


def samples_from_synthetic(df, graphs: list[dict], encoder: SemanticEncoder) -> list[GraphSample]:
    rows = df.set_index("task_id")
    out = []
    for g in graphs:
        ids = [t for t in g["task_ids"] if t in rows.index]
        if len(ids) < 3:
            continue
        tasks = [{"task_id": t, "description": str(rows.at[t, "task_description"] or ""),
                  "category": rows.at[t, "task_category"], "feature": rows.at[t, "feature"],
                  "complexity": rows.at[t, "estimated_complexity"]} for t in ids]
        smp = build_sample(tasks, {tuple(e) for e in g["edges"]}, encoder)
        smp.project_id = g["project_id"]
        out.append(smp)
    return out


# ------------------------------------------------------------------ baseline
def rule_baseline_scores(tasks: list[dict], pairs: torch.Tensor) -> np.ndarray:
    """NetworkX-style heuristic: u->v if same feature (or v is project-level) and stage(u) < stage(v)."""
    s = []
    for a, b in pairs.t().tolist():
        u, v = tasks[a], tasks[b]
        su, sv = STAGE.get(u.get("category"), 1), STAGE.get(v.get("category"), 1)
        related = u.get("feature") == v.get("feature") or u.get("feature") == "project" or v.get("feature") == "project"
        s.append(1.0 if related and su < sv else 0.0)
    return np.array(s)


# ------------------------------------------------------------------ training
def _metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4), "average_precision": round(float(average_precision_score(y, p)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4), "threshold": round(float(thr), 3)}


def _predict(model, s: GraphSample) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(s.data.x, s.data.edge_index, s.pairs, s.pair_feats)).numpy()


def train(df, graphs: list[dict], epochs: int = 40, seed: int = 42, min_graphs: int = 30,
          test_projects: set[str] | None = None) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    encoder = get_encoder()
    samples = samples_from_synthetic(df, graphs, encoder)
    if len(samples) < min_graphs:
        m = {"status": "insufficient_data", "graphs": len(samples), "required": min_graphs,
             "note": "GNN not trained: too few labelled dependency graphs. Rule baseline will be used."}
        METRICS_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")
        return m
    random.shuffle(samples)
    if test_projects:  # reuse the tabular models' project-level test split (no project in both train and test)
        te = [s for s in samples if s.project_id in test_projects]
        rest = [s for s in samples if s.project_id not in test_projects]
        k = max(1, int(0.125 * len(rest)))
        va, tr = rest[:k], rest[k:]
    else:
        n = len(samples)
        tr, va, te = samples[: int(0.7 * n)], samples[int(0.7 * n): int(0.8 * n)], samples[int(0.8 * n):]
    model = DependencyGNN(samples[0].data.x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    history = []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for s in tr:
            pos = (s.labels == 1).nonzero().squeeze(-1)
            neg = (s.labels == 0).nonzero().squeeze(-1)
            neg = neg[torch.randperm(len(neg))[: max(3 * len(pos), 1)]]  # 1:3 negative sampling
            sel = torch.cat([pos, neg])
            logits = model(s.data.x, s.data.edge_index, s.pairs[:, sel], s.pair_feats[sel])
            loss = F.binary_cross_entropy_with_logits(logits, s.labels[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        yv = np.concatenate([s.labels.numpy() for s in va])
        pv = np.concatenate([_predict(model, s) for s in va])
        history.append({"epoch": ep + 1, "train_loss": round(tot / len(tr), 4), "val_ap": round(float(average_precision_score(yv, pv)), 4)})
    # threshold chosen on validation (max F1), never on test
    thr_grid = np.linspace(0.1, 0.9, 33)
    thr = float(max(thr_grid, key=lambda t: f1_score(yv, (pv >= t).astype(int), zero_division=0)))
    yt = np.concatenate([s.labels.numpy() for s in te])
    pt = np.concatenate([_predict(model, s) for s in te])
    pb = np.concatenate([rule_baseline_scores(s.tasks, s.pairs) for s in te])
    torch.save({"state_dict": model.state_dict(), "in_dim": samples[0].data.x.shape[1], "threshold": thr,
                "encoder_backend": encoder.backend}, CKPT)
    m = {"status": "trained", "data": "SYNTHETIC", "encoder_backend": encoder.backend, "graphs": {"train": len(tr), "val": len(va), "test": len(te)},
         "test_pairs": int(len(yt)), "test_positive_rate": round(float(yt.mean()), 4),
         "gnn_graphsage": _metrics(yt, pt, thr), "rule_baseline_networkx": _metrics(yt, pb, 0.5), "history": history}
    METRICS_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")
    load_model.cache_clear()
    return m


# ------------------------------------------------------------------ inference
_cache: dict = {}


def load_model():
    if "m" not in _cache:
        if not CKPT.exists():
            _cache["m"] = None
        else:
            ck = torch.load(CKPT, weights_only=False)
            m = DependencyGNN(ck["in_dim"])
            m.load_state_dict(ck["state_dict"])
            m.eval()
            _cache["m"] = (m, ck["threshold"], ck["encoder_backend"])
    return _cache["m"]


load_model.cache_clear = lambda: _cache.clear()  # type: ignore[attr-defined]


def predict_dependencies(tasks: list[dict]) -> tuple[list[dict], str]:
    """Return scored directed edges (prob >= threshold) and the method used."""
    if len(tasks) < 2:
        return [], "none"
    loaded = load_model()
    encoder = get_encoder()
    s = build_sample(tasks, None, encoder)
    if loaded is not None and loaded[2] == encoder.backend:
        model, thr, _ = loaded
        probs, method = _predict(model, s), "gnn_graphsage"
    else:
        probs, thr, method = rule_baseline_scores(tasks, s.pairs), 0.5, "rule_baseline_networkx"
    ids = [t["task_id"] for t in tasks]
    edges = [{"source": ids[a], "target": ids[b], "score": round(float(p), 4), "method": method}
             for (a, b), p in zip(s.pairs.t().tolist(), probs) if p >= thr]
    return edges, method


def build_dag(task_ids: list[str], edges: list[dict]) -> tuple[nx.DiGraph, list[dict]]:
    """Insert edges by descending confidence, rejecting any that would close a cycle."""
    g = nx.DiGraph()
    g.add_nodes_from(task_ids)
    rejected = []
    for e in sorted(edges, key=lambda e: -e["score"]):
        g.add_edge(e["source"], e["target"], **e)
        if not nx.is_directed_acyclic_graph(g):
            g.remove_edge(e["source"], e["target"])
            rejected.append({**e, "reason": "would create cycle"})
    # drop redundant transitive edges to keep the plan readable
    red = nx.transitive_reduction(g)
    red.add_nodes_from(g.nodes)
    for u, v in red.edges:
        red[u][v].update(g[u][v])
    return red, rejected


def find_cycles(edges: list[tuple[str, str]]) -> list[list[str]]:
    return list(nx.simple_cycles(nx.DiGraph(edges)))
