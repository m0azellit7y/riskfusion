from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest

from riskfusion import contracts as C

ROOT = Path(__file__).resolve().parents[2]


def good_event(**over: object) -> dict:
    ev = {
        "schema": "event.v1",
        "session_id": "sess_8f2a1c",
        "ts_ms": 1834200,
        "channel": "environment",
        "detector": "object_detector",
        "detector_version": "rtdetr-r18-coco",
        "event_type": "OBJECT_DETECTED",
        "payload": {"label": "cell_phone", "bbox_x": 0.41, "bbox_y": 0.72, "bbox_w": 0.09, "bbox_h": 0.14},
        "confidence": 0.71,
        "quality": {"frame_blur": 0.12, "frame_luma": 118, "usable": True},
    }
    ev.update(over)
    return ev


def test_every_channel_has_unknown_state() -> None:  # FR-7
    for ch in C.CHANNELS:
        et = C.EVENT_TYPES[f"{ch.upper()}_UNKNOWN"]
        assert et.channel == ch and et.unknown


def test_recommendations_never_contain_a_verdict() -> None:  # CON-5 / ETH-6
    assert "VIOLATION" not in C.RECOMMENDATIONS
    assert all("VIOLAT" not in r for r in C.RECOMMENDATIONS)


def test_valid_event_passes() -> None:
    assert C.event_errors(good_event()) == []
    C.validate_event(good_event())


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"schema": "event.v2"}, "schema"),
        ({"session_id": "a b"}, "session_id"),
        ({"ts_ms": -1}, "ts_ms"),
        ({"ts_ms": 1.5}, "ts_ms"),
        ({"channel": "gps"}, "channel"),
        ({"confidence": 1.2}, "confidence"),
        ({"confidence": True}, "confidence"),
        ({"event_type": "HEAD_GAZE"}, "belongs to channel"),
        ({"event_type": "NOT_A_TYPE"}, "not registered"),
        ({"detector": ""}, "detector"),
        ({"payload": {"label": "banana", "bbox_x": 0, "bbox_y": 0, "bbox_w": 0, "bbox_h": 0}}, "label"),
        ({"payload": {"label": "cell_phone"}}, "missing"),
        ({"quality": {"frame_luma": 400}}, "frame_luma"),
    ],
)
def test_invalid_events_are_rejected_with_reason(override: dict, fragment: str) -> None:
    errs = C.event_errors(good_event(**override))
    assert errs and any(fragment in e for e in errs)
    with pytest.raises(C.ContractError):
        C.validate_event(good_event(**override))


def test_extra_payload_field_rejected() -> None:
    ev = good_event()
    ev["payload"] = {**ev["payload"], "secret": 1}
    assert any("unexpected" in e for e in C.event_errors(ev))


def test_flat_roundtrip_all_types() -> None:
    samples = {
        "HEAD_GAZE": {"yaw": 1.0, "pitch": -3.0, "roll": 0.5, "gaze_on_screen_prob": 0.9},
        "VOICE_ACTIVITY": {"vad_active": True, "speaker_is_candidate": False, "n_speakers": 2, "foreign_speech": True},
        "PASTE": {"length": 42},
        "PRESENCE_UNKNOWN": {"reason": "low_light"},
    }
    for et, payload in samples.items():
        ev = good_event(event_type=et, channel=C.EVENT_TYPES[et].channel, payload=payload)
        assert C.event_errors(ev) == []
        back = C.flat_to_event(C.event_to_flat(ev))
        assert back["payload"] == payload
        assert C.event_errors(back) == []


def test_frame_validation_never_drops_rows_and_matches_dict_validation() -> None:
    good = [good_event(ts_ms=i) for i in range(20)]
    rows = [C.event_to_flat(e) for e in good]
    df = pd.DataFrame(rows)
    df.loc[3, "confidence"] = 7.0
    df.loc[5, "p0"] = np.nan
    df.loc[7, "channel"] = "presence"
    df.loc[9, "session_id"] = "!"
    valid, rejected = C.validate_event_frame(df)
    assert len(valid) + len(rejected) == len(df)  # FR-8: zero silent drops
    assert sorted(rejected.index) == [3, 5, 7, 9]
    assert rejected["reason"].str.len().gt(0).all()
    # the dict validator agrees row by row
    for i, row in df.iterrows():
        ok = i not in rejected.index
        try:
            ev = C.flat_to_event(row.to_dict())
            dict_ok = C.event_errors(ev) == []
        except (ValueError, KeyError, TypeError):
            dict_ok = False
        assert ok == dict_ok, i


def test_dead_letter_file(tmp_path: Path) -> None:
    df = pd.DataFrame([C.event_to_flat(good_event())])
    df.loc[0, "confidence"] = 2.0
    _, rej = C.validate_event_frame(df)
    n = C.write_dead_letter(rej, tmp_path / "dl.jsonl")
    assert n == 1
    rec = json.loads((tmp_path / "dl.jsonl").read_text().splitlines()[0])
    assert "confidence" in rec["reason"]


def test_published_json_schemas_are_current() -> None:
    assert json.loads((ROOT / "contracts" / "event.v1.schema.json").read_text()) == C.event_json_schema()
    assert (
        json.loads((ROOT / "contracts" / "risk_assessment.v1.schema.json").read_text())
        == C.risk_assessment_json_schema()
    )


def test_json_schema_agrees_with_python_validator() -> None:
    schema = C.event_json_schema()
    jsonschema.validate(good_event(), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(good_event(event_type="HEAD_GAZE"), schema)


def test_risk_assessment_example_from_srs_is_valid() -> None:
    ra = {
        "schema": "risk_assessment.v1",
        "session_id": "sess_8f2a1c",
        "overall_risk": 0.71,
        "calibrated": True,
        "recommendation": "HUMAN_REVIEW",
        "confidence_band": [0.61, 0.79],
        "flags": [
            {
                "flag_id": "flg_001",
                "type": "SECOND_VOICE",
                "t_start_ms": 1834200,
                "t_end_ms": 1851000,
                "confidence": 0.88,
                "explanation": "A voice that does not match the enrolled candidate was detected for 17 seconds.",
                "evidence_ref": "sessions/sess_8f2a1c/audio_1834.wav",
            }
        ],
        "top_contributors": [
            {"feature": "f_foreign_speech_duration_300s", "shap": 0.24, "direction": "increases_risk"}
        ],
        "channels_available": ["presence", "identity", "audio_voice", "screen"],
        "channels_missing": ["attention", "pose"],
        "model_version": "fusion-v2.4.1",
        "feature_version": "features-v1.3.0",
    }
    assert C.risk_assessment_errors(ra) == []
    jsonschema.validate(ra, C.risk_assessment_json_schema())
    assert C.risk_assessment_errors({**ra, "recommendation": "VIOLATION"})
