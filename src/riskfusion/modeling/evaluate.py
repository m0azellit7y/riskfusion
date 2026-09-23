"""Phases 5-6 evaluation (FR-22, 23, 25, 26, 27, 28, 29, 30, 31, 33) — all on the VALIDATION split.

The sealed holdout is only read through ``holdout_evaluation`` (two permitted accesses, FR-21).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskfusion.contracts import CHANNELS
from riskfusion.data.splits import open_sealed_holdout
from riskfusion.features.engine import FEATURE_VERSION

from . import metrics as M
from .model import RiskModel, feature_channels, prepare
from .train import DIRECTIONS, FPR_BUDGET, RuleBaseline, fit_lgbm, load_dataset

SLICE_FACTORS = ("lighting", "webcam_class", "room_noise", "connection_stability", "eyewear", "head_covering")


def _calibrated_threshold(yca: np.ndarray, p_ca: np.ndarray) -> tuple[Any, float]:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_ca, yca)
    return iso, M.threshold_at_fpr(yca, iso.predict(p_ca), FPR_BUDGET)


# ------------------------------------------------------------------------------------ FR-23 / FR-26
def global_interpretation(model: RiskModel, Xva: pd.DataFrame, top: int = 10) -> dict[str, Any]:
    contrib = model.contributions(Xva)[:, :-1]
    imp = np.abs(contrib).mean(0)
    order = np.argsort(-imp)
    importance = [
        {
            "feature": model.features[i],
            "mean_abs_shap": float(imp[i]),
            "channels": list(feature_channels(model.features[i])),
        }
        for i in order[:25]
    ]
    pdp = []
    sample = Xva.sample(min(300, len(Xva)), random_state=0)
    for i in order[:top]:
        f = model.features[i]
        vals = Xva[f].dropna()
        if vals.nunique() < 2:
            continue
        grid = np.unique(np.quantile(vals, np.linspace(0.02, 0.98, 15)))
        curve = []
        for g in grid:
            Xs = sample.copy()
            Xs[f] = g
            curve.append(float(model.predict(Xs).mean()))
        pdp.append({"feature": f, "grid": grid.tolist(), "mean_risk": curve})
    return {"importance": importance, "partial_dependence": pdp}


# ------------------------------------------------------------------------------------ FR-25
def flag_iou(flags: pd.DataFrame, labels: pd.DataFrame, session_ids: list[str]) -> dict[str, Any]:
    by_type: dict[str, list[float]] = {}
    fl = flags[flags["session_id"].isin(session_ids)]
    fgroups = {k: g for k, g in fl.groupby("session_id")}
    for sid in session_ids:
        ivs = (
            json.loads(labels.at[sid, "intervals"])
            if isinstance(labels.at[sid, "intervals"], str)
            else labels.at[sid, "intervals"]
        )
        f = fgroups.get(sid)
        for iv in ivs:
            best = 0.0
            if f is not None:
                ff = f[f["type"] == iv["type"]]
                for s, e in zip(ff["t_start_ms"], ff["t_end_ms"], strict=True):
                    inter = max(0, min(e, iv["end_ms"]) - max(s, iv["start_ms"]))
                    union = max(e, iv["end_ms"]) - min(s, iv["start_ms"])
                    best = max(best, inter / union if union else 0.0)
            by_type.setdefault(iv["type"], []).append(best)
    all_iou = [v for vs in by_type.values() for v in vs]
    neg_ids = [s for s in session_ids if not json.loads(labels.at[s, "intervals"])]
    fp_sessions = fl[fl["session_id"].isin(neg_ids)]["session_id"].nunique()
    return {
        "episodes": len(all_iou),
        "share_iou_ge_0_5": float(np.mean(np.array(all_iou) >= 0.5)) if all_iou else float("nan"),
        "mean_best_iou": float(np.mean(all_iou)) if all_iou else float("nan"),
        "by_type": {
            k: {"episodes": len(v), "share_iou_ge_0_5": float(np.mean(np.array(v) >= 0.5))}
            for k, v in sorted(by_type.items())
        },
        "clean_sessions_with_any_flag": float(fp_sessions / max(1, len(neg_ids))),
        "target": "IoU >= 0.5 for >= 70% of true episodes (FR-25)",
    }


# ------------------------------------------------------------------------------------ FR-22
def ablations(ds: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
    tr, va, ca = (ds.split == s for s in ("train", "validation", "calibration"))
    ytr, yva, yca = ds.y[tr].to_numpy(), ds.y[va].to_numpy(), ds.y[ca].to_numpy()
    out: list[dict[str, Any]] = []

    def run(name: str, group: str, cols: list[str]) -> None:
        b, it = fit_lgbm(ds.X.loc[tr, cols], ytr, ds.X.loc[va, cols], yva, params)
        p_va = b.predict(prepare(ds.X.loc[va, cols], DIRECTIONS).to_numpy(dtype=float), num_iteration=it)
        p_ca = b.predict(prepare(ds.X.loc[ca, cols], DIRECTIONS).to_numpy(dtype=float), num_iteration=it)
        iso, thr = _calibrated_threshold(yca, p_ca)
        r = M.rates(yva, iso.predict(p_va), thr)
        out.append(
            {
                "group": group,
                "variant": name,
                "n_features": len(cols),
                "val_pr_auc": M.pr_auc(yva, p_va),
                "val_recall_at_op": r["recall"],
                "val_fpr_at_op": r["fpr"],
            }
        )

    allc = list(ds.X.columns)
    run("all features", "reference", allc)
    for ch in CHANNELS:
        cols = [c for c in allc if ch not in feature_channels(c)]
        if len(cols) < len(allc):
            run(f"without {ch}", "channel removed", cols)
    run("baseline normalisation off", "normalisation", [c for c in allc if not c.startswith("bn_")])
    whole = [c for c in allc if "__w" not in c]
    for ws in ((), (10,), (60,), (300,)):
        cols = whole + [c for c in allc if any(f"__w{w}_" in c for w in ws)]
        run("whole-session only" if not ws else f"whole-session + {ws[0]} s", "window set", cols)
    # model families
    base = RuleBaseline()
    p_b = base.predict(ds.X[va])
    thr_b = M.threshold_at_fpr(yca, base.predict(ds.X[ca]), FPR_BUDGET)
    rb = M.rates(yva, p_b, thr_b)
    out.append(
        {
            "group": "model family",
            "variant": "rule baseline",
            "n_features": len(base.weights),
            "val_pr_auc": M.pr_auc(yva, p_b),
            "val_recall_at_op": rb["recall"],
            "val_fpr_at_op": rb["fpr"],
        }
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lr = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000),
    )
    lr.fit(ds.X[tr], ytr)
    p_lr = lr.predict_proba(ds.X[va])[:, 1]
    iso, thr = _calibrated_threshold(yca, lr.predict_proba(ds.X[ca])[:, 1])
    rl = M.rates(yva, iso.predict(p_lr), thr)
    out.append(
        {
            "group": "model family",
            "variant": "logistic regression",
            "n_features": len(allc),
            "val_pr_auc": M.pr_auc(yva, p_lr),
            "val_recall_at_op": rl["recall"],
            "val_fpr_at_op": rl["fpr"],
        }
    )
    return out


# ------------------------------------------------------------------------------------ FR-27 / FR-28
def fpr_disparity(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> float:
    fprs = []
    for g in np.unique(groups):
        m = (groups == g) & (y == 0)
        if m.sum() == 0:
            continue
        fprs.append((pred[m].sum() + 1) / (m.sum() + 2))  # add-one smoothing (METRICS.md)
    return float(max(fprs) / min(fprs)) if len(fprs) > 1 else float("nan")


def sliced_metrics(y: np.ndarray, p: np.ndarray, thr: float, meta: pd.DataFrame) -> dict[str, Any]:
    pred = p >= thr
    out: dict[str, Any] = {}
    for factor in SLICE_FACTORS:
        g = meta[factor].astype(str).to_numpy()
        rows = []
        for level in sorted(np.unique(g)):
            m = g == level
            neg = m & (y == 0)
            lo, hi = M.bootstrap_ci(
                lambda yy, pp: float((pp[yy == 0] >= thr).mean()) if (yy == 0).any() else np.nan, y[m], p[m], n=500
            )
            rows.append(
                {
                    "level": level,
                    "n": int(m.sum()),
                    "positives": int(y[m].sum()),
                    "underpowered": bool(m.sum() < 300),
                    "fpr": float(pred[neg].mean()) if neg.any() else float("nan"),
                    "fpr_ci": [lo, hi],
                    "recall": float(pred[m & (y == 1)].mean()) if (m & (y == 1)).any() else float("nan"),
                    "pr_auc": M.pr_auc(y[m], p[m]),
                    "ece": M.ece(y[m], p[m]),
                }
            )
        disp = fpr_disparity(y, pred, g)
        ci = M.bootstrap_ci(lambda yy, pp, gg: fpr_disparity(yy, pp >= thr, gg), y, p, g, n=500, stratify=g)
        out[factor] = {"levels": rows, "fpr_disparity": disp, "fpr_disparity_ci": list(ci)}
    return out


# ------------------------------------------------------------------------------------ FR-29
def mitigation(ds: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Slice-balanced reweighting: clean sessions are reweighted so each lighting x webcam cell carries equal
    total weight, so the model cannot learn to treat dim/low-quality video as evidence."""
    tr, va, ca = (ds.split == s for s in ("train", "validation", "calibration"))
    ytr, yva, yca = ds.y[tr].to_numpy(), ds.y[va].to_numpy(), ds.y[ca].to_numpy()
    cell = (ds.meta["lighting"].astype(str) + "|" + ds.meta["webcam_class"].astype(str))[tr].to_numpy()
    w = np.ones(len(ytr))
    neg = ytr == 0
    counts = pd.Series(cell[neg]).value_counts()
    target = neg.sum() / len(counts)
    for c, n in counts.items():
        w[neg & (cell == c)] = target / n
    res = {}
    for name, sw in (("before (unweighted)", None), ("after (slice-balanced weights)", w)):
        b, it = fit_lgbm(ds.X[tr], ytr, ds.X[va], yva, params, sample_weight=sw)
        p_va = b.predict(prepare(ds.X[va], DIRECTIONS).to_numpy(dtype=float), num_iteration=it)
        iso, thr = _calibrated_threshold(
            yca, b.predict(prepare(ds.X[ca], DIRECTIONS).to_numpy(dtype=float), num_iteration=it)
        )
        pc = iso.predict(p_va)
        r = M.rates(yva, pc, thr)
        res[name] = {
            "val_pr_auc": M.pr_auc(yva, p_va),
            "recall": r["recall"],
            "fpr": r["fpr"],
            **{
                f"fpr_disparity_{f}": fpr_disparity(yva, pc >= thr, ds.meta.loc[va, f].astype(str).to_numpy())
                for f in ("lighting", "webcam_class", "room_noise", "head_covering")
            },
        }
    return {"method": "slice-balanced reweighting of clean sessions (lighting x webcam)", "results": res}


# ------------------------------------------------------------------------------------ FR-30
def missing_channel_test(model: RiskModel, Xva: pd.DataFrame, yva: np.ndarray) -> list[dict[str, Any]]:
    full = model.predict(Xva)
    thr = model.meta["tiers"]["HUMAN_REVIEW"]
    rows = []
    for ch in CHANNELS:
        miss = model.predict(Xva, missing=[ch])
        inc = miss - full
        rows.append(
            {
                "channel": ch,
                "sessions": int(len(full)),
                "share_increased": float(np.mean(inc > 1e-9)),
                "max_increase": float(max(0.0, inc.max())),
                "mean_change": float(inc.mean()),
                "recall_full": float(((full >= thr) & (yva == 1)).sum() / max(1, yva.sum())),
                "recall_missing": float(((miss >= thr) & (yva == 1)).sum() / max(1, yva.sum())),
            }
        )
    return rows


# ------------------------------------------------------------------------------------ FR-33
def cost_analysis(y_ca: np.ndarray, p_ca: np.ndarray, y_va: np.ndarray, p_va: np.ndarray) -> list[dict[str, Any]]:
    out = []
    cands = np.unique(np.concatenate([p_ca, [1.01]]))
    for ratio in (1, 2, 5, 10, 20, 50):
        # cost of a false accusation (FP) = ratio x cost of a missed violation (FN)
        costs = [ratio * ((p_ca >= t) & (y_ca == 0)).sum() + ((p_ca < t) & (y_ca == 1)).sum() for t in cands]
        t = float(cands[int(np.argmin(costs))])
        r = M.rates(y_va, p_va, t)
        out.append(
            {
                "fp_to_fn_cost_ratio": ratio,
                "threshold": t,
                "val_recall": r["recall"],
                "val_fpr": r["fpr"],
                "val_expected_cost_per_1000": 1000 * (ratio * r["fp"] + r["fn"]) / len(y_va),
            }
        )
    return out


# ------------------------------------------------------------------------------------ orchestration
def evaluate_all(data_root: Path, dataset_version: str, model_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(data_root, dataset_version)
    model = RiskModel.load(model_dir)
    DIRECTIONS.clear()
    DIRECTIONS.update({k: int(v) for k, v in model.meta["directions"].items()})
    va, ca = (ds.split == "validation"), (ds.split == "calibration")
    Xva, yva = ds.X[va], ds.y[va].to_numpy()
    p_va = model.predict(Xva)
    p_ca = model.predict(ds.X[ca])
    thr = model.meta["tiers"]["HUMAN_REVIEW"]
    labels = pd.read_parquet(data_root / "synthetic" / dataset_version / "labels.parquet").set_index("session_id")
    flags = pd.read_parquet(data_root / "features" / FEATURE_VERSION / dataset_version / "flags.parquet")
    report: dict[str, Any] = {
        "model_version": model.meta["model_version"],
        "dataset_version": dataset_version,
        "split": "validation",
        "operating_threshold": thr,
    }
    report["validation"] = M.summary(yva, p_va, thr)
    report["reliability"] = M.reliability(yva, p_va)
    report["interpretation"] = global_interpretation(model, Xva)
    report["flags"] = flag_iou(flags, labels, list(Xva.index))
    report["slices"] = sliced_metrics(yva, p_va, thr, ds.meta[va])
    report["missing_channel"] = missing_channel_test(model, Xva, yva)
    report["cost"] = cost_analysis(ds.y[ca].to_numpy(), p_ca, yva, p_va)
    report["ablations"] = ablations(ds, model.meta["params"])
    report["mitigation"] = mitigation(ds, model.meta["params"])
    report["by_profile"] = [
        {"profile": prof, "n": int(m.sum()), "flagged_share": float((p_va[m] >= thr).mean())}
        for prof in sorted(ds.meta.loc[va, "behavior_profile"].unique())
        for m in [(ds.meta.loc[va, "behavior_profile"] == prof).to_numpy()]
    ]
    subtle = ds.meta.loc[va].get("subtle")
    if subtle is not None:
        s = subtle.fillna(False).to_numpy(bool)
        report["subtle_recall"] = {
            "subtle": float((p_va[s & (yva == 1)] >= thr).mean()) if (s & (yva == 1)).any() else None,
            "overt": float((p_va[~s & (yva == 1)] >= thr).mean()),
        }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


def model_features(model_dir: Path) -> set[str]:
    return set(json.loads((model_dir / "bundle.json").read_text())["features"])


def holdout_evaluation(
    data_root: Path, dataset_version: str, model_dir: Path, out_dir: Path, access: int, purpose: str
) -> dict[str, Any]:
    """The only code path that reads the sealed holdout. Access 1 = Phase 4 headline; access 2 = final."""
    ids = open_sealed_holdout(data_root, dataset_version, purpose)
    fdir = data_root / "features" / FEATURE_VERSION / dataset_version
    X = pd.read_parquet(fdir / "features.parquet").set_index("session_id").loc[ids]
    X = X[[c for c in X.columns if c in model_features(model_dir)]]
    meta = (
        pd.read_parquet(data_root / "synthetic" / dataset_version / "sessions.parquet").set_index("session_id").loc[ids]
    )
    y = meta["violation"].astype(int).to_numpy()
    model = RiskModel.load(model_dir)
    p = model.predict(X)
    thr = model.meta["tiers"]["HUMAN_REVIEW"]
    base = RuleBaseline()
    pb = base.predict(X)
    res: dict[str, Any] = {
        "access": access,
        "purpose": purpose,
        "n": len(ids),
        "positives": int(y.sum()),
        "primary": M.summary(y, p, thr),
        "baseline": M.summary(y, pb, model.meta.get("baseline_threshold", 0.5)),
    }
    res["primary"]["pr_auc_ci"] = list(M.bootstrap_ci(M.pr_auc, y, p, n=1000))
    if access == 2:
        res["reliability"] = M.reliability(y, p)
        res["slices"] = sliced_metrics(y, p, thr, meta)
        res["missing_channel"] = missing_channel_test(model, X, y)
    (out_dir / f"holdout_access_{access}.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    return res
