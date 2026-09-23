from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskfusion.contracts import CHANNELS
from riskfusion.features.engine import SIGNALS, compute_session, reorder_with_watermark
from riskfusion.modeling import metrics as M
from riskfusion.modeling.model import RiskModel, explain_feature, feature_channels, mask_channels
from riskfusion.modeling.monitoring import psi, wilson_low
from riskfusion.simulator import load_config
from riskfusion.simulator.engine import sessions_to_event_frame, simulate_session

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "trained" / "fusion-v1.0.0"


@pytest.fixture(scope="module")
def sims():  # type: ignore[no-untyped-def]
    cfg = load_config(ROOT / "configs" / "simulator" / "v1_1.yaml")
    ss = [simulate_session(cfg, i, "u") for i in range(40)]
    df = sessions_to_event_frame(ss, "t")
    return [(s, df[df["session_id"] == s.meta["session_id"]]) for s in ss]


def test_grid_marks_gaps_and_features_are_finite(sims) -> None:  # type: ignore[no-untyped-def]
    s, ev = sims[0]
    sf = compute_session(s.meta["session_id"], ev, s.meta["duration_s"])
    assert len(sf.grid) == s.meta["duration_s"]
    assert set(SIGNALS) <= set(sf.grid.columns)
    assert all(np.isfinite(v) or np.isnan(v) for v in sf.features.values())
    assert {f"unk_{c}" for c in CHANNELS} <= set(sf.features)


def test_out_of_order_within_30s_is_accepted_later_is_rejected() -> None:  # FR-10
    ev = pd.DataFrame({"ts_ms": [1000, 5000, 2000, 60_000, 10_000, 61_000]})
    ok, late = reorder_with_watermark(ev)
    assert list(ok["ts_ms"]) == [1000, 2000, 5000, 60_000, 61_000]
    assert list(late["ts_ms"]) == [10_000]


def test_violations_are_flagged_with_correct_type(sims) -> None:  # type: ignore[no-untyped-def]
    hits = 0
    for s, ev in sims:
        if not s.label["violation"] or s.meta.get("subtle"):
            continue
        flags = compute_session(s.meta["session_id"], ev, s.meta["duration_s"]).flags
        hits += any(f["type"] in s.label["violation_types"] for f in flags)
    assert hits >= 1


def test_metrics_definitions() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9])
    thr = M.threshold_at_fpr(y, p, 0.25)
    assert M.rates(y, p, thr)["fpr"] <= 0.25
    assert M.ece(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
    assert 0 < psi(np.random.default_rng(0).normal(0, 1, 5000), np.random.default_rng(1).normal(1, 1, 500)) > 0.2
    assert wilson_low(0, 50) == 0.0 and wilson_low(10, 50) < 0.2


def test_every_feature_maps_to_a_channel_or_is_context() -> None:
    model = RiskModel.load(MODEL)
    unmapped = [f for f in model.features if not feature_channels(f)]
    assert set(unmapped) <= {"duration_min"}


@pytest.mark.parametrize("channel", CHANNELS)
def test_missing_channel_never_increases_risk(channel: str) -> None:  # FR-30, property test
    model = RiskModel.load(MODEL)
    rng = np.random.default_rng(3)
    n = 400
    X = pd.DataFrame({f: rng.gamma(0.6, 1.0, n) * rng.integers(0, 2, n) for f in model.features})
    for c in X.columns:
        if c.startswith("unk_"):
            X[c] = rng.uniform(0, 0.3, n)
    full = model.predict(X)
    missing = model.predict(X, missing=[channel])
    assert (missing <= full + 1e-12).all()
    assert (mask_channels(X, [channel])[f"unk_{channel}"] == 1.0).all()


def test_assessment_contract_and_language() -> None:
    model = RiskModel.load(MODEL)
    feats = {f: 0.0 for f in model.features}
    ra = model.assess("sess_unit_1", feats, [], list(CHANNELS), [])
    assert ra["schema"] == "risk_assessment.v1" and ra["recommendation"] in (
        "NO_ACTION",
        "ROUTINE_REVIEW",
        "HUMAN_REVIEW",
        "PRIORITY_REVIEW",
    )
    assert len(ra["top_contributors"]) == 5
    text = " ".join(c["explanation"] for c in ra["top_contributors"]).lower()
    for word in ("cheat", "guilty", "violation detected", "misconduct"):  # ETH-6
        assert word not in text
    assert "never counts against" in explain_feature("unk_audio_voice", "decreases_risk")
