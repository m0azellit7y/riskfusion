"""FR-31 noise robustness and FR-41 drift monitoring.

Drift monitor: compares each incoming batch with a reference (the validation split the operating point was
tuned for) on three things, each with an alert threshold:
  * input distribution — PSI of the 10 most important features (alert PSI > 0.2 on any feature)
  * prediction rate    — share of sessions sent to human review (alert if > 2x or < 0.5x the reference)
  * per-slice FPR      — FPR of clean sessions per lighting level, when labels are available
                         (alert if any slice exceeds 2x the FPR budget)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskfusion.features.engine import FEATURE_VERSION

from . import metrics as M
from .model import RiskModel
from .train import FPR_BUDGET, load_dataset

PSI_ALERT = 0.2
RATE_ALERT = 2.0


def wilson_low(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    ph = k / n
    return float((ph + z * z / (2 * n) - z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / (1 + z * z / n))


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:  # (near-)constant reference: compare the share of non-reference values instead
        edges = np.array([-np.inf, edges[0] + 1e-12, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / ref.size
    c = np.histogram(cur, edges)[0] / cur.size
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _load_eval(data_root: Path, dv: str, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fdir = data_root / "features" / FEATURE_VERSION / dv
    X = pd.read_parquet(fdir / "features.parquet").set_index("session_id")
    meta = pd.read_parquet(data_root / "synthetic" / dv / "sessions.parquet").set_index("session_id").loc[X.index]
    return X.reindex(columns=features), meta


def noise_robustness(data_root: Path, model: RiskModel, normal_dv: str, noisy_dv: str) -> dict[str, Any]:
    thr = model.meta["tiers"]["HUMAN_REVIEW"]
    out = {}
    for name, dv in (("detector noise as trained (v1.1)", normal_dv), ("detector false positives +50%", noisy_dv)):
        X, meta = _load_eval(data_root, dv, model.features)
        y = meta["violation"].astype(int).to_numpy()
        out[name] = {"dataset_version": dv, **M.summary(y, model.predict(X), thr)}
    a, b = out.values()
    out["change"] = {
        "pr_auc": b["pr_auc"] - a["pr_auc"],
        "fpr": b["fpr"] - a["fpr"],
        "recall": b["recall"] - a["recall"],
    }
    return out


def drift_report(
    data_root: Path, model: RiskModel, ref_dv: str, batches: list[tuple[str, str]], batch_size: int = 200
) -> dict[str, Any]:
    ds = load_dataset(data_root, ref_dv)
    va = ds.split == "validation"
    Xref = ds.X[va].reindex(columns=model.features)
    pref = model.predict(Xref)
    thr = model.meta["tiers"]["HUMAN_REVIEW"]
    ref_rate = float((pref >= thr).mean())
    contrib = np.abs(model.contributions(Xref)[:, :-1]).mean(0)
    top = [model.features[i] for i in np.argsort(-contrib)[:10]]
    rows = []
    for label, dv in batches:
        X, meta = _load_eval(data_root, dv, model.features)
        y = meta["violation"].astype(int).to_numpy()
        p = model.predict(X)
        for b, start in enumerate(range(0, len(X), batch_size)):
            sl = slice(start, start + batch_size)
            psis = {f: psi(Xref[f].to_numpy(float), X[f].to_numpy(float)[sl]) for f in top}
            rate = float((p[sl] >= thr).mean())
            slice_fpr, slice_low = {}, {}
            for lvl in sorted(meta["lighting"].unique()):
                m = (meta["lighting"].to_numpy()[sl] == lvl) & (y[sl] == 0)
                if m.sum() >= 20:
                    k = int((p[sl][m] >= thr).sum())
                    slice_fpr[str(lvl)] = k / int(m.sum())
                    slice_low[str(lvl)] = wilson_low(k, int(m.sum()))
            neg = y[sl] == 0
            k_all = int((p[sl][neg] >= thr).sum())
            fpr_all = k_all / max(1, int(neg.sum()))
            alerts = []
            worst = max(psis, key=lambda k: psis[k] if np.isfinite(psis[k]) else -1)
            if psis[worst] > PSI_ALERT:
                alerts.append(f"input drift: PSI {psis[worst]:.2f} on {worst}")
            if rate > RATE_ALERT * ref_rate or rate < ref_rate / RATE_ALERT:
                alerts.append(f"review rate {rate:.1%} vs reference {ref_rate:.1%}")
            # alert only when the evidence is statistically clear (95% Wilson lower bound), not on small-sample noise
            if wilson_low(k_all, int(neg.sum())) > 1.5 * FPR_BUDGET:
                alerts.append(f"clean-session FPR {fpr_all:.1%} clearly above the {FPR_BUDGET:.0%} budget")
            for lvl, lo in slice_low.items():
                if lo > 2 * FPR_BUDGET:
                    alerts.append(f"FPR {slice_fpr[lvl]:.1%} in {lvl} light clearly exceeds 2x the budget")
            rows.append(
                {
                    "source": label,
                    "dataset_version": dv,
                    "batch": b + 1,
                    "sessions": int(len(p[sl])),
                    "max_psi": psis[worst],
                    "max_psi_feature": worst,
                    "psi": psis,
                    "review_rate": rate,
                    "clean_fpr": fpr_all,
                    "slice_fpr_lighting": slice_fpr,
                    "alerts": alerts,
                }
            )
    return {
        "reference": {
            "dataset_version": ref_dv,
            "split": "validation",
            "review_rate": ref_rate,
            "features_monitored": top,
        },
        "thresholds": {"psi": PSI_ALERT, "review_rate_ratio": RATE_ALERT, "slice_fpr": 2 * FPR_BUDGET},
        "batches": rows,
    }


def run_monitoring(data_root: Path, model_dir: Path, out_dir: Path) -> dict[str, Any]:
    model = RiskModel.load(model_dir)
    nr = noise_robustness(data_root, model, "sim-v1.1-s777-n800", "sim-v1.1-noise150-s777-n800")
    batches = [
        ("new sessions, same conditions", "sim-v1.1-s777-n800"),
        ("detectors degraded (+50% false positives)", "sim-v1.1-noise150-s777-n800"),
    ]
    v1 = data_root / "features" / FEATURE_VERSION / "sim-v1-s20260923-n5000" / "features.parquet"
    if v1.exists():  # an older simulator = a population that shifted
        batches.append(("population shift (simulator v1)", "sim-v1-s20260923-n5000"))
    dr = drift_report(data_root, model, "sim-v1.1-s20260923-n5000", batches)
    dr["batches"] = [b for b in dr["batches"] if not (b["dataset_version"].startswith("sim-v1-") and b["batch"] > 4)]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "noise_robustness.json").write_text(json.dumps(nr, indent=2, default=float) + "\n")
    (out_dir / "drift.json").write_text(json.dumps(dr, indent=2, default=float) + "\n")
    return {"noise": nr, "drift": dr}
