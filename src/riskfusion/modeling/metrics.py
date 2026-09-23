"""Metric definitions — exactly as fixed in docs/METRICS.md."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 15) -> list[dict[str, float]]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [
        {
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "n": int((idx == b).sum()),
            "mean_pred": float(p[idx == b].mean()),
            "frac_pos": float(y[idx == b].mean()),
        }
        for b in range(bins)
        if (idx == b).any()
    ]


def rates(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, float]:
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": tp / max(1, tp + fn),
        "fpr": fp / max(1, fp + tn),
        "precision": tp / max(1, tp + fp),
        "flag_rate": float(pred.mean()),
    }


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, budget: float) -> float:
    """Lowest threshold whose FPR on clean sessions is <= budget."""
    neg = np.sort(s[y == 0])[::-1]
    k = int(np.floor(budget * len(neg)))
    if k >= len(neg):
        return float(neg[-1])
    return float(np.nextafter(neg[k], np.inf))


def summary(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "pr_auc": pr_auc(y, p),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "ece": ece(y, p),
        "brier": float(brier_score_loss(y, p)),
    }
    out.update(rates(y, p, thr))
    return out


def bootstrap_ci(
    fn: Any, *arrays: np.ndarray, n: int = 1000, seed: int = 0, stratify: np.ndarray | None = None
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = len(arrays[0])
    vals = []
    for _ in range(n):
        if stratify is not None:
            idx = np.concatenate(
                [
                    rng.choice(np.flatnonzero(stratify == g), size=int((stratify == g).sum()))
                    for g in np.unique(stratify)
                ]
            )
        else:
            idx = rng.integers(0, N, N)
        v = fn(*(a[idx] for a in arrays))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
