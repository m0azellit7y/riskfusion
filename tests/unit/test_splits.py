from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskfusion.data import splits as S


def _sessions(n: int = 2000, seed: int = 0, participants: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    profiles = ["clean", "fidgety_clean", "phone_user", "note_reader"]
    prof = rng.choice(profiles, size=n, p=[0.6, 0.3, 0.05, 0.05])
    df = pd.DataFrame(
        {
            "session_id": [f"s{i:05d}" for i in range(n)],
            "behavior_profile": prof,
            "violation": np.isin(prof, ["phone_user", "note_reader"]),
        }
    )
    if participants:
        df["participant_id"] = [f"p{i % 97}" for i in range(n)]
    return df


def test_splits_disjoint_complete_and_proportional() -> None:
    df = _sessions()
    out = S.assign_splits(df, seed=1)
    assert len(out) == len(df) and out["session_id"].is_unique
    frac = out["split"].value_counts(normalize=True)
    for k, v in S.SPLIT_FRACTIONS.items():
        assert frac[k] == pytest.approx(v, abs=0.01)


def test_stratified_by_violation_and_profile() -> None:
    df = _sessions()
    out = S.assign_splits(df, seed=1).merge(df, on="session_id")
    tab = pd.crosstab(out["behavior_profile"], out["split"])
    assert (tab > 0).all().all(), "every profile must appear in every split"
    pos = out.groupby("split")["violation"].mean()
    assert pos.max() - pos.min() < 0.015


def test_participants_never_span_splits() -> None:  # DR-5
    df = _sessions(participants=True)
    out = S.assign_splits(df, seed=3).merge(df, on="session_id")
    assert (out.groupby("participant_id")["split"].nunique() == 1).all()


def test_deterministic() -> None:
    df = _sessions()
    pd.testing.assert_frame_equal(S.assign_splits(df, seed=5), S.assign_splits(df, seed=5))


def test_sealed_holdout_guard(tmp_path: Path) -> None:
    df = _sessions(500)
    manifest = S.create_splits(tmp_path, "dv1", df, seed=2)
    with pytest.raises(S.HoldoutSealedError):
        S.create_splits(tmp_path, "dv1", df, seed=2)  # created exactly once
    with pytest.raises(S.HoldoutSealedError):
        S.load_split_ids(tmp_path, "dv1", "holdout")
    train = S.load_split_ids(tmp_path, "dv1", "train")
    assert not set(train) & set(manifest["holdout_session_ids"])
    with pytest.raises(ValueError):
        S.open_sealed_holdout(tmp_path, "dv1", "x")
    ids1 = S.open_sealed_holdout(tmp_path, "dv1", "phase 4: first sealed evaluation")
    ids2 = S.open_sealed_holdout(tmp_path, "dv1", "phase 7: final sealed evaluation")
    assert ids1 == ids2 == manifest["holdout_session_ids"]
    with pytest.raises(S.HoldoutSealedError):
        S.open_sealed_holdout(tmp_path, "dv1", "third access must be refused")
    log = S.holdout_accesses(tmp_path, "dv1")
    assert [e["access_no"] for e in log] == [1, 2]


def test_tampered_manifest_detected(tmp_path: Path) -> None:
    import json

    S.create_splits(tmp_path, "dv2", _sessions(300), seed=2)
    p = tmp_path / "splits" / "dv2" / "manifest.json"
    m = json.loads(p.read_text())
    m["holdout_session_ids"] = m["holdout_session_ids"][:-1]
    p.write_text(json.dumps(m))
    with pytest.raises(S.HoldoutSealedError):
        S.open_sealed_holdout(tmp_path, "dv2", "should detect tampering")
