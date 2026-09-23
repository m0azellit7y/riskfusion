from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskfusion.contracts import CHANNELS, event_errors, flat_to_event, validate_event_frame
from riskfusion.simulator import generate, load_config
from riskfusion.simulator.config import PROFILE_VIOLATIONS
from riskfusion.simulator.engine import sessions_to_event_frame, simulate_session

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "simulator" / "v1.yaml"


@pytest.fixture(scope="module")
def cfg():  # type: ignore[no-untyped-def]
    return load_config(CFG)


@pytest.fixture(scope="module")
def many(cfg):  # type: ignore[no-untyped-def]
    return [simulate_session(cfg, i, "t") for i in range(400)]


def test_config_rates_follow_dr3_formula(cfg) -> None:  # type: ignore[no-untyped-def]
    d = cfg.detectors["phone"]
    expect = cfg.pessimism * d.ref_prevalence * d.recall * (1 - d.precision) / d.precision
    assert cfg.fp_rate("phone") == pytest.approx(expect)
    assert cfg.fp_rate("gaze_off_screen") > cfg.fp_rate("phone") > cfg.fp_rate("face_present")  # gaze noisiest
    assert cfg.fp_rate("tab_hidden") == 0.0


def test_session_is_deterministic_and_independent_of_order(cfg) -> None:  # type: ignore[no-untyped-def]
    a = simulate_session(cfg, 17, "t")
    _ = [simulate_session(cfg, i, "t") for i in range(5)]
    b = simulate_session(cfg, 17, "t")
    assert a.meta == b.meta and a.label == b.label
    for k in a.events:
        np.testing.assert_array_equal(a.events[k], b.events[k])
    c = load_config(CFG, seed=cfg.seed + 1)
    assert simulate_session(c, 17, "t").meta != a.meta


def test_labels_match_profiles_and_intervals_in_range(many) -> None:  # type: ignore[no-untyped-def]
    for s in many:
        prof = s.meta["behavior_profile"]
        assert s.label["violation"] == (prof not in ("clean", "fidgety_clean", "poor_environment"))
        if prof in PROFILE_VIOLATIONS:
            assert s.label["violation_types"] == list(PROFILE_VIOLATIONS[prof])
        if s.label["violation"]:
            assert s.label["intervals"], "a violating session must have at least one episode"
        for iv in s.label["intervals"]:
            assert 0 <= iv["start_ms"] < iv["end_ms"] <= s.meta["duration_s"] * 1000


def test_all_events_are_contract_valid(cfg, many) -> None:  # type: ignore[no-untyped-def]
    df = sessions_to_event_frame(many[:60], cfg.config_version)
    valid, rejected = validate_event_frame(df)
    assert len(rejected) == 0
    sample = valid.sample(2000, random_state=0)
    for row in sample.to_dict(orient="records"):
        assert event_errors(flat_to_event(row)) == []


def test_every_channel_emits_explicit_unknown(cfg, many) -> None:  # FR-7 demonstrated for all channels
    df = sessions_to_event_frame(many, cfg.config_version)
    seen = set(df.loc[df["event_type"].astype(str).str.endswith("_UNKNOWN"), "channel"].astype(str))
    # behavioural telemetry is generated in-browser and never goes missing in the simulator;
    # its UNKNOWN path is exercised by the API/browser contract tests instead.
    assert set(CHANNELS) - {"behavioral"} <= seen


def test_detector_noise_is_present_and_nuisance_dependent(cfg, many) -> None:  # type: ignore[no-untyped-def]
    """Clean sessions still produce false phone and gaze detections (a perfect simulator is a defect)."""
    df = sessions_to_event_frame(many, cfg.config_version)
    meta = pd.DataFrame([s.meta for s in many])
    clean_ids = set(meta.loc[~meta["violation"], "session_id"])
    obj = df[(df["event_type"] == "OBJECT_DETECTED") & (df["label"] == "cell_phone")]
    assert obj["session_id"].astype(str).isin(clean_ids).sum() > 0
    gaze = df[df["event_type"] == "HEAD_GAZE"].merge(meta[["session_id", "lighting"]], on="session_id")
    gaze["off"] = gaze["p3"] < 0.5
    rate = gaze.groupby("lighting")["off"].mean()
    assert rate["dim"] > rate["bright"]


def test_nuisance_is_independent_of_label(cfg) -> None:  # type: ignore[no-untyped-def]
    c = load_config(CFG)
    c.profiles["clean"].weight = 0.5
    for p in ("phone_user", "note_reader"):
        c.profiles[p].weight = 0.25
    for p in (
        "fidgety_clean",
        "poor_environment",
        "second_person",
        "impersonation",
        "tab_switcher",
        "remote_assisted",
        "mixed",
    ):
        c.profiles[p].weight = 0.0
    meta = pd.DataFrame([simulate_session(c, i, "n").meta for i in range(600)])
    tab = pd.crosstab(meta["violation"], meta["lighting"], normalize="index")
    assert (tab.max() - tab.min()).max() < 0.12


def test_generate_is_reproducible_and_versioned(tmp_path: Path) -> None:
    c = load_config(CFG, n_sessions=30, shard_size=12)
    r1 = generate(c, tmp_path / "a")
    r2 = generate(c, tmp_path / "b")
    assert r1.manifest["content_hash"] == r2.manifest["content_hash"]
    assert r1.manifest["dead_letter_events"] == 0
    assert r1.manifest["config_hash"] == c.config_hash()
    with pytest.raises(FileExistsError):
        generate(c, tmp_path / "a")
    manifest = json.loads((r1.out_dir / "manifest.json").read_text())
    assert manifest["n_sessions"] == 30
    labels = pd.read_parquet(r1.out_dir / "labels.parquet")
    assert len(labels) == 30 and set(labels["source"]) == {"simulated"}
    events = pd.read_parquet(r1.out_dir / "events")
    assert events["session_id"].nunique() == 30
