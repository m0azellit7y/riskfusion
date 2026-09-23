"""Real video -> detectors -> features -> risk assessment -> review, through the API (PostgreSQL)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import pytest
from skimage import data

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def face_video(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, bytes]:
    import sys

    sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
    from make_face_video import build

    d = tmp_path_factory.mktemp("video")
    video = build(d / "face.webm")
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
    assert ok
    return video.read_bytes(), jpg.tobytes()


def test_model_info_and_score(client) -> None:  # type: ignore[no-untyped-def]
    info = client.get("/model-info").json()
    assert info["model_version"] == "fusion-v1.0.0" and "verdict" in info["output"]
    ev = {
        "schema": "event.v1",
        "session_id": "api_score_1",
        "ts_ms": 0,
        "channel": "screen",
        "detector": "t",
        "detector_version": "1",
        "event_type": "TAB_VISIBILITY",
        "payload": {"hidden": False},
        "confidence": 1.0,
    }
    r = client.post(
        "/score", json={"session_id": "api_score_1", "events": [ev, {**ev, "ts_ms": -1}], "duration_s": 120}
    )
    body = r.json()
    assert r.status_code == 200 and body["_meta"]["events_rejected"] == 1
    assert body["recommendation"] in ("NO_ACTION", "ROUTINE_REVIEW", "HUMAN_REVIEW", "PRIORITY_REVIEW")
    assert set(body["channels_missing"]) >= {"presence", "identity", "audio_voice"}


def test_recording_is_analysed_end_to_end(client, face_video) -> None:  # type: ignore[no-untyped-def]
    video, photo = face_video
    p = client.post("/participants", json={"adult_confirmed": True}).json()
    client.post(
        f"/participants/{p['id']}/consent",
        json={
            "consent_version": "consent-v1.0",
            "signed_name": "Video Test",
            "consent_recording": True,
            "consent_analysis": True,
            "consent_retention": True,
            "understands_withdrawal": True,
        },
    )
    sid = client.post(
        "/sessions",
        json={
            "participant_id": p["id"],
            "script_id": "clean",
            "lighting": "normal",
            "webcam_class": "hd",
            "room_noise": "quiet",
        },
    ).json()["id"]
    client.post(f"/sessions/{sid}/confirm-consent", json={"confirmed": True})
    assert (
        client.post(
            f"/sessions/{sid}/recordings",
            data={"kind": "enrollment_image"},
            files={"file": ("e.jpg", photo, "image/jpeg")},
        ).status_code
        == 201
    )
    client.post(f"/sessions/{sid}/start", json={})
    client.post(f"/sessions/{sid}/stop", json={"duration_s": 60})
    assert (
        client.post(
            f"/sessions/{sid}/recordings",
            data={"kind": "webcam_av", "duration_s": "60"},
            files={"file": ("r.webm", video, "video/webm")},
        ).status_code
        == 201
    )
    assert client.get(f"/sessions/{sid}/assessment").json() is None
    assert client.post(f"/sessions/{sid}/process").status_code == 200
    assert client.post(f"/sessions/{sid}/process").status_code == 409  # no double processing
    for _ in range(120):
        job = client.get(f"/sessions/{sid}/job").json()
        if job["status"] in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(0.5)
    assert job["status"] == "SUCCEEDED", job
    assert job["details"]["enrolment_face_found"] is True
    d = client.get(f"/sessions/{sid}").json()
    assert d["status"] == "COMPLETED"
    assert d["event_counts"]["presence"] >= 40 and d["event_counts"]["identity"] >= 5
    tl = client.get(f"/sessions/{sid}/signal-timeline").json()
    lanes = {lane["label"]: lane["intervals"] for lane in tl["lanes"]}
    assert {"No face", "Second face", "Gaze off screen", "Phone detected"} <= set(lanes)
    assert lanes["Second face"], "the two-face scene (30-40 s) must appear on the timeline"
    assert lanes["No face"] or lanes["Channel unknown"], "the empty scene (20-30 s) must appear"
    ra = client.get(f"/sessions/{sid}/assessment").json()
    assert ra["schema"] == "risk_assessment.v1" and 0 <= ra["overall_risk"] <= 1
    assert "liveness" in ra["channels_missing"]  # a 60 s still photo: liveness is not judged in the first minute
    html = client.get(f"/sessions/{sid}/report")
    assert html.status_code == 200 and "not a finding of misconduct" in html.text
    r = client.post(f"/sessions/{sid}/review", json={"verdict": "NO_CONCERN", "note": "test"})
    assert r.status_code == 201
    assert client.get(f"/sessions/{sid}").json()["status"] == "REVIEWED"
    summary = client.get("/reviews/summary").json()
    assert summary["reviews"] >= 1 and summary["overturn_rate"] is not None


def test_mock_corpus_validation(client) -> None:  # type: ignore[no-untyped-def]
    """FR-3/FR-6 run on whatever analysed mock sessions exist (here: the test video session)."""
    r = client.post("/corpus/validate")
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["detector_errors"]["n_sessions"] >= 1
    assert set(rep["detector_errors"]["detectors"]) >= {"phone", "tab_hidden", "identity_mismatch"}
    assert rep["measured_config"]["written"] is False  # far below the 20-session minimum
    assert client.get("/corpus/validation").json()["detector_errors"]["n_sessions"] >= 1


def test_upload_starts_analysis_automatically(client, face_video) -> None:  # type: ignore[no-untyped-def]
    from riskfusion_api.settings import get_settings

    video, photo = face_video
    get_settings().auto_analyse = True
    try:
        p = client.post("/participants", json={"adult_confirmed": True}).json()
        client.post(
            f"/participants/{p['id']}/consent",
            json={
                "consent_version": "consent-v1.0",
                "signed_name": "Auto Test",
                "consent_recording": True,
                "consent_analysis": True,
                "consent_retention": True,
                "understands_withdrawal": True,
            },
        )
        sid = client.post(
            "/sessions",
            json={
                "participant_id": p["id"],
                "script_id": "clean",
                "lighting": "normal",
                "webcam_class": "hd",
                "room_noise": "quiet",
            },
        ).json()["id"]
        client.post(f"/sessions/{sid}/confirm-consent", json={"confirmed": True})
        client.post(
            f"/sessions/{sid}/recordings",
            data={"kind": "enrollment_image"},
            files={"file": ("e.jpg", photo, "image/jpeg")},
        )
        client.post(f"/sessions/{sid}/start", json={})
        client.post(f"/sessions/{sid}/stop", json={"duration_s": 60})
        client.post(
            f"/sessions/{sid}/recordings",
            data={"kind": "webcam_av", "duration_s": "60"},
            files={"file": ("r.webm", video, "video/webm")},
        )
        for _ in range(120):  # nobody pressed "Analyse": the upload itself queued it
            job = client.get(f"/sessions/{sid}/job").json()
            if job and job["status"] in ("SUCCEEDED", "FAILED"):
                break
            time.sleep(0.5)
        assert job["status"] == "SUCCEEDED", job
        channels = set(client.get(f"/sessions/{sid}").json()["event_counts"])
        assert {"presence", "identity", "liveness", "attention", "environment", "pose", "audio_voice"} <= channels
        assert client.get(f"/sessions/{sid}/assessment").json()["schema"] == "risk_assessment.v1"
    finally:
        get_settings().auto_analyse = False
