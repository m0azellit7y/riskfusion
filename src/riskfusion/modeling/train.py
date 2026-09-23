"""Phase 4 training (FR-16, 17, 18, 19, 20, 21, 32, 38).

* Model selection uses the validation split only. The sealed holdout is never touched here.
* Calibration is fitted on the dedicated calibration split; the calibration method is chosen by validation ECE.
* The operating point and review tiers are set on the calibration split from reviewer capacity (FPR budgets).
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from riskfusion.data.splits import load_manifest, load_split_ids
from riskfusion.features.engine import FEATURE_VERSION

from . import metrics as M
from .model import RiskModel, is_model_feature, monotone_direction, prepare

MODEL_VERSION = "fusion-v1.0.0"
SEED = 20260923
FPR_BUDGET = 0.03  # reviewer capacity: at most 3% of clean sessions sent to human review (FR-32)
ROUTINE_FLOOR = 0.005
TIER_FPR = {"ROUTINE_REVIEW": 0.10, "HUMAN_REVIEW": FPR_BUDGET, "PRIORITY_REVIEW": 0.005}
ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------------------------ data
@dataclass
class Dataset:
    X: pd.DataFrame
    T: pd.DataFrame  # temporal (ROCKET) features
    y: pd.Series
    meta: pd.DataFrame
    split: pd.Series


def load_dataset(data_root: Path, dataset_version: str, include_holdout: bool = False) -> Dataset:
    fdir = data_root / "features" / FEATURE_VERSION / dataset_version
    X = pd.read_parquet(fdir / "features.parquet").set_index("session_id")
    X = X[[c for c in X.columns if is_model_feature(c)]]
    T = pd.read_parquet(fdir / "temporal.parquet").set_index("session_id")
    meta = pd.read_parquet(data_root / "synthetic" / dataset_version / "sessions.parquet").set_index("session_id")
    split = pd.Series(index=X.index, dtype=object)
    for s in ("train", "validation", "calibration"):
        split.loc[split.index.intersection(load_split_ids(data_root, dataset_version, s))] = s
    if not include_holdout:
        keep = split.notna()
        X, T, split = X[keep], T.loc[X.index], split[keep]
    meta = meta.loc[X.index]
    return Dataset(X, T.loc[X.index], meta["violation"].astype(int), meta, split)


# ------------------------------------------------------------------------------------ models
class RuleBaseline:
    def __init__(self, path: Path = ROOT / "configs" / "models" / "baseline.yaml"):
        cfg = yaml.safe_load(path.read_text())
        self.version, self.bias, self.weights = cfg["version"], cfg["bias"], cfg["weights"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        z = self.bias + sum(w * X[f].fillna(0).to_numpy() for f, w in self.weights.items())
        return 1 / (1 + np.exp(-z))


def learn_directions(X: pd.DataFrame, y: np.ndarray) -> dict[str, int]:
    """Monotone direction per feature from the TRAINING split: sign of the rank correlation with the label."""
    out: dict[str, int] = {}
    yr = pd.Series(y, index=X.index)
    for c in X.columns:
        d = monotone_direction(c)
        if d == 1 and not c.startswith("unk_"):
            x = X[c].fillna(0.0)
            corr = x.rank().corr(yr.rank()) if x.nunique() > 1 else 0.0
            d = -1 if (corr is not None and np.isfinite(corr) and corr < -0.02) else 1
        out[c] = d
    return out


DIRECTIONS: dict[str, int] = {}


def fit_lgbm(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xva: pd.DataFrame,
    yva: np.ndarray,
    params: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[Any, int]:
    import lightgbm as lgb

    dirs = DIRECTIONS or learn_directions(Xtr, ytr)
    Xtr, Xva = prepare(Xtr, dirs), prepare(Xva, dirs)
    mono = [dirs.get(c, 1) for c in Xtr.columns]  # FR-30 guarantee, see model.mask_channels
    full = {
        "objective": "binary",
        "learning_rate": 0.04,
        "num_leaves": 15,
        "min_child_samples": 25,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "scale_pos_weight": float((ytr == 0).sum() / max(1, ytr.sum())),  # FR-19 class imbalance
        "monotone_constraints": mono,
        "monotone_constraints_method": "advanced",
        "seed": SEED,
        "deterministic": True,
        "num_threads": 1,
        "verbose": -1,
    }
    full.update(params)
    dtr = lgb.Dataset(Xtr.to_numpy(dtype=float), ytr, weight=sample_weight, feature_name=list(Xtr.columns))
    dva = lgb.Dataset(Xva.to_numpy(dtype=float), yva, reference=dtr)
    booster = lgb.train(
        full,
        dtr,
        num_boost_round=1500,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False)],
        feval=_feval_ap,
    )
    return booster, booster.best_iteration


def _feval_ap(preds: np.ndarray, data: Any) -> tuple[str, float, bool]:
    return "pr_auc", M.pr_auc(data.get_label(), preds), True


def fit_temporal(Ttr: pd.DataFrame, ytr: np.ndarray, Tva: pd.DataFrame, yva: np.ndarray) -> tuple[Any, float, float]:
    best = (None, -1.0, 0.0)
    for C in (0.003, 0.01, 0.03, 0.1):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
        m.fit(Ttr.to_numpy(), ytr)
        ap = M.pr_auc(yva, m.predict_proba(Tva.to_numpy())[:, 1])
        if ap > best[1]:
            best = (m, ap, C)
    return best


def fit_calibration(p_cal: np.ndarray, y_cal: np.ndarray, p_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
    grid = np.unique(np.concatenate([[0.0, 1.0], np.quantile(p_cal, np.linspace(0, 1, 400))]))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, y_cal)
    iso_y = iso.predict(grid)
    logit = np.log(np.clip(p_cal, 1e-6, 1 - 1e-6) / (1 - np.clip(p_cal, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit[:, None], y_cal)
    g_logit = np.log(np.clip(grid, 1e-6, 1 - 1e-6) / (1 - np.clip(grid, 1e-6, 1 - 1e-6)))
    platt_y = platt.predict_proba(g_logit[:, None])[:, 1]
    e_iso = M.ece(y_val, np.interp(p_val, grid, iso_y))
    e_platt = M.ece(y_val, np.interp(p_val, grid, platt_y))
    # Platt keeps the ranking strictly monotone (isotonic creates ties that scramble the review queue); use
    # isotonic only if it is clearly better calibrated.
    method, ys = ("isotonic", iso_y) if e_iso + 0.005 < e_platt else ("platt", platt_y)
    ys = np.maximum.accumulate(ys)
    # confidence band: bootstrap the chosen calibrator over the calibration split
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(200):
        i = rng.integers(0, len(p_cal), len(p_cal))
        if method == "isotonic":
            b = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal[i], y_cal[i]).predict(grid)
        else:
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit[i][:, None], y_cal[i])
            b = lr.predict_proba(g_logit[:, None])[:, 1]
        boots.append(b)
    B = np.array(boots)
    return {
        "method": method,
        "ece_validation": {"isotonic": e_iso, "platt": e_platt},
        "x": grid.tolist(),
        "y": ys.tolist(),
        "band": {
            "x": grid.tolist(),
            "lo": np.percentile(B, 5, axis=0).tolist(),
            "hi": np.percentile(B, 95, axis=0).tolist(),
            "method": "bootstrap 90% interval, 200 resamples",
        },
    }


# ------------------------------------------------------------------------------------ tracking (FR-38)
class Run:
    def __init__(self, root: Path, name: str, params: dict[str, Any]):
        self.id = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}-{uuid.uuid4().hex[:6]}"
        self.dir = root / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.params, self.t0 = params, time.time()
        self.metrics: dict[str, Any] = {}
        self.root = root

    def log(self, **metrics: Any) -> None:
        self.metrics.update(metrics)

    def artifact(self, name: str, obj: Any) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=2, default=float) + "\n")

    def close(self) -> dict[str, Any]:
        rec = {
            "run_id": self.id,
            "params": self.params,
            "metrics": self.metrics,
            "seconds": round(time.time() - self.t0, 1),
            "platform": platform.platform(),
        }
        self.artifact("run.json", rec)
        with (self.root / "index.jsonl").open("a") as fh:
            fh.write(
                json.dumps({"run_id": self.id, "name": self.params.get("name"), "metrics": self.metrics}, default=float)
                + "\n"
            )
        return rec


# ------------------------------------------------------------------------------------ main
def train_all(data_root: Path, dataset_version: str, models_root: Path, runs_root: Path) -> dict[str, Any]:
    ds = load_dataset(data_root, dataset_version)
    tr, va, ca = (ds.split == s for s in ("train", "validation", "calibration"))
    Xtr, Xva, Xca = ds.X[tr], ds.X[va], ds.X[ca]
    ytr, yva, yca = ds.y[tr].to_numpy(), ds.y[va].to_numpy(), ds.y[ca].to_numpy()
    feats = list(ds.X.columns)
    DIRECTIONS.clear()
    DIRECTIONS.update(learn_directions(Xtr, ytr))
    results: dict[str, Any] = {
        "dataset_version": dataset_version,
        "feature_version": FEATURE_VERSION,
        "split_version": load_manifest(data_root, dataset_version)["split_version"],
    }

    # FR-16 rule baseline
    base = RuleBaseline()
    run = Run(runs_root, "rule-baseline", {"name": "rule-baseline", "weights": base.weights, "bias": base.bias})
    pb_va, pb_ca = base.predict(Xva), base.predict(Xca)
    thr_b = M.threshold_at_fpr(yca, pb_ca, FPR_BUDGET)
    results["baseline"] = {"version": base.version, "validation": M.summary(yva, pb_va, thr_b), "threshold": thr_b}
    run.log(**{f"val_{k}": v for k, v in results["baseline"]["validation"].items()})
    run.close()

    # FR-17/19 primary gradient-boosted model, small grid selected on validation PR-AUC
    grid = [
        {"num_leaves": nl, "min_child_samples": mc, "learning_rate": lr}
        for nl in (7, 15, 31)
        for mc in (15, 40)
        for lr in (0.03, 0.06)
    ]
    best: dict[str, Any] = {"ap": -1.0}
    trials = []
    for params in grid:
        booster, it = fit_lgbm(Xtr, ytr, Xva, yva, params)
        ap = M.pr_auc(yva, booster.predict(prepare(Xva, DIRECTIONS).to_numpy(dtype=float), num_iteration=it))
        trials.append({**params, "best_iteration": it, "val_pr_auc": ap})
        if ap > best["ap"]:
            best = {"ap": ap, "params": params, "iter": it, "booster": booster}
    run = Run(runs_root, "lgbm-grid", {"name": "lgbm-grid", "grid": grid, "selected": best["params"]})
    run.artifact("trials.json", trials)
    import lightgbm as lgb

    booster = best["booster"]
    booster = lgb.Booster(model_str=booster.model_to_string(num_iteration=best["iter"]))
    p_va = np.asarray(booster.predict(prepare(Xva, DIRECTIONS).to_numpy(dtype=float)), dtype=float)
    p_ca = np.asarray(booster.predict(prepare(Xca, DIRECTIONS).to_numpy(dtype=float)), dtype=float)

    # FR-20 calibration on the calibration split
    cal = fit_calibration(p_ca, yca, p_va, yva)
    pc_va = np.clip(np.interp(p_va, cal["x"], cal["y"]), 0, 1)
    pc_ca = np.clip(np.interp(p_ca, cal["x"], cal["y"]), 0, 1)
    # FR-32 operating point + review tiers from reviewer capacity (calibration split, clean sessions)
    tiers = {k: M.threshold_at_fpr(yca, pc_ca, v) for k, v in TIER_FPR.items()}
    # isotonic calibration maps most clean sessions to exactly 0; a routine-review tier needs real evidence
    tiers["ROUTINE_REVIEW"] = min(max(tiers["ROUTINE_REVIEW"], ROUTINE_FLOOR), tiers["HUMAN_REVIEW"])
    tiers["PRIORITY_REVIEW"] = max(tiers["PRIORITY_REVIEW"], tiers["HUMAN_REVIEW"])
    op = tiers["HUMAN_REVIEW"]
    val = M.summary(yva, pc_va, op)
    val["pr_auc_uncalibrated"] = M.pr_auc(yva, p_va)
    val["ece_uncalibrated"] = M.ece(yva, p_va)
    results["primary"] = {
        "version": MODEL_VERSION,
        "params": best["params"],
        "best_iteration": best["iter"],
        "validation": val,
        "calibration": {k: cal[k] for k in ("method", "ece_validation")},
        "operating_point": op,
        "tiers": tiers,
        "reliability_validation": M.reliability(yva, pc_va),
    }
    rel_gain = (val["pr_auc"] - results["baseline"]["validation"]["pr_auc"]) / results["baseline"]["validation"][
        "pr_auc"
    ]
    results["primary"]["relative_pr_auc_gain_vs_baseline"] = rel_gain
    run.log(
        val_pr_auc=val["pr_auc"],
        val_ece=val["ece"],
        val_recall_at_op=val["recall"],
        val_fpr_at_op=val["fpr"],
        rel_gain_vs_baseline=rel_gain,
    )
    run.close()

    # FR-18 temporal model over the per-10 s grid (ROCKET + logistic regression)
    tm, t_ap, t_C = fit_temporal(ds.T[tr], ytr, ds.T[va], yva)
    pt_va = tm.predict_proba(ds.T[va].to_numpy())[:, 1]
    pt_ca = tm.predict_proba(ds.T[ca].to_numpy())[:, 1]
    iso_t = IsotonicRegression(out_of_bounds="clip").fit(pt_ca, yca)
    ptc_va = iso_t.predict(pt_va)
    thr_t = M.threshold_at_fpr(yca, iso_t.predict(pt_ca), FPR_BUDGET)
    results["temporal"] = {
        "method": "ROCKET (400 kernels) + logistic regression",
        "C": t_C,
        "validation": M.summary(yva, ptc_va, thr_t),
    }
    run = Run(runs_root, "temporal-rocket", {"name": "temporal-rocket", "C": t_C})
    run.log(val_pr_auc=results["temporal"]["validation"]["pr_auc"])
    run.close()

    clean_ref = Xtr[ytr == 0].median(numeric_only=True).fillna(0).to_dict()
    bundle = RiskModel(
        booster,
        feats,
        {
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "features": feats,
            "dataset_version": dataset_version,
            "split_version": results["split_version"],
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "params": best["params"],
            "best_iteration": best["iter"],
            "calibration": {"method": cal["method"], "x": cal["x"], "y": cal["y"]},
            "band": cal["band"],
            "operating_point": {"threshold": op, "fpr_budget": FPR_BUDGET},
            "tiers": tiers,
            "tier_fpr_budgets": TIER_FPR,
            "clean_reference": clean_ref,
            "baseline_threshold": thr_b,
            "headline_metrics_validation": {k: val[k] for k in ("pr_auc", "ece", "recall", "fpr", "brier")},
            "seed": SEED,
            "directions": dict(DIRECTIONS),
        },
    )
    bundle.save(models_root / MODEL_VERSION)
    (models_root / MODEL_VERSION / "training_results.json").write_text(json.dumps(results, indent=2, default=float))
    return results
