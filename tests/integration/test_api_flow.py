"""End-to-end API workflow against a real PostgreSQL database and real files on disk."""

from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.db

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 200  # minimal JPEG header bytes


def _participant(client) -> dict:  # type: ignore[no-untyped-def]
    r = client.post("/participants", json={"adult_confirmed": True})
    assert r.status_code == 201, r.text
    p = r.json()
    version = client.get("/consent/current").json()["version"]
    r = client.post(
        f"/participants/{p['id']}/consent",
        json={
            "consent_version": version,
            "signed_name": "Test Person",
            "consent_recording": True,
            "consent_analysis": True,
            "consent_retention": True,
            "understands_withdrawal": True,
            "ethics_reference": "ETH-TEST-001",
        },
    )
    assert r.status_code == 201, r.text
    return p


def _tele(sid: str, ts: int, et: str, channel: str, payload: dict, uid: str | None = None) -> dict:
    return {
        "event_uid": uid or uuid.uuid4().hex,
        "schema": "event.v1",
        "session_id": sid,
        "ts_ms": ts,
        "channel": channel,
        "detector": "browser_telemetry",
        "detector_version": "web-1.0",
        "event_type": et,
        "payload": payload,
        "confidence": 1.0,
    }


def test_health(client) -> None:  # type: ignore[no-untyped-def]
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["checks"] == {"database": "ok", "storage": "ok"}


def test_minors_and_partial_consent_are_refused(client) -> None:  # type: ignore[no-untyped-def]
    r = client.post("/participants", json={"adult_confirmed": False})
    assert r.status_code == 422 and "18" in r.json()["detail"]
    p = client.post("/participants", json={"adult_confirmed": True}).json()
    r = client.post(
        f"/participants/{p['id']}/consent",
        json={
            "consent_version": "consent-v1.0",
            "signed_name": "X Y",
            "consent_recording": True,
            "consent_analysis": False,
            "consent_retention": True,
            "understands_withdrawal": True,
        },
    )
    assert r.status_code == 422
    r = client.post(
        "/sessions",
        json={
            "participant_id": p["id"],
            "script_id": "clean",
            "lighting": "normal",
            "webcam_class": "hd",
            "room_noise": "quiet",
        },
    )
    assert r.status_code == 409 and "consent" in r.json()["detail"]


def test_full_mock_session_workflow(client, tiny_webm, api_env) -> None:  # type: ignore[no-untyped-def]
    p = _participant(client)
    r = client.post(
        "/sessions",
        json={
            "participant_id": p["id"],
            "script_id": "rehearsal-cues",
            "lighting": "dim",
            "webcam_class": "sd",
            "room_noise": "moderate",
            "eyewear": True,
        },
        headers={"X-Operator": "tester"},
    )
    assert r.status_code == 201, r.text
    s = r.json()
    sid = s["id"]
    assert s["status"] == "CREATED" and s["is_rehearsal"] is True

    # cannot start before consent is confirmed
    assert client.post(f"/sessions/{sid}/start", json={}).status_code == 409
    assert client.post(f"/sessions/{sid}/confirm-consent", json={"confirmed": True}).json()["status"] == "CONSENTED"

    # enrolment photo
    r = client.post(
        f"/sessions/{sid}/recordings",
        data={"kind": "enrollment_image"},
        files={"file": ("enrol.jpg", JPEG, "image/jpeg")},
    )
    assert r.status_code == 201, r.text

    r = client.post(f"/sessions/{sid}/start", json={"browser": {"userAgent": "pytest"}})
    assert r.status_code == 200 and r.json()["status"] == "RECORDING"
    assert len(r.json()["episodes"]) == 1

    # telemetry: 2 valid, 1 invalid (dead-lettered), then a duplicate re-send
    dup = uuid.uuid4().hex
    batch = [
        _tele(sid, 0, "TAB_VISIBILITY", "screen", {"hidden": False}, dup),
        _tele(sid, 900, "MONITOR_COUNT", "device", {"count": 1}),
        _tele(sid, 1000, "PASTE", "screen", {"length": -5}),
    ]
    res = client.post(f"/sessions/{sid}/events", json={"events": batch}).json()
    assert res == {"accepted": 2, "duplicates": 0, "rejected": 1, "rejections": res["rejections"]}
    assert "length" in res["rejections"][0]["reason"]
    res = client.post(f"/sessions/{sid}/events", json={"events": batch[:1]}).json()
    assert res["accepted"] == 0 and res["duplicates"] == 1
    # the browser may not post detector channels
    bad = _tele(sid, 5, "PERSON_COUNT", "environment", {"n_persons": 2})
    assert client.post(f"/sessions/{sid}/events", json={"events": [bad]}).json()["rejected"] == 1

    assert client.post(f"/sessions/{sid}/episodes/1", json={"shown_at_ms": 20100}).status_code == 200
    assert client.post(f"/sessions/{sid}/stop", json={"duration_s": 40.0}).json()["status"] == "RECORDED"

    # wrong content is refused even with a video MIME type
    r = client.post(
        f"/sessions/{sid}/recordings",
        data={"kind": "webcam_av"},
        files={"file": ("x.webm", b"not a video", "video/webm")},
    )
    assert r.status_code == 415
    r = client.post(
        f"/sessions/{sid}/recordings",
        data={"kind": "webcam_av", "duration_s": "40"},
        files={"file": ("rec.webm", tiny_webm, "video/webm;codecs=vp8,opus")},
    )
    assert r.status_code == 201, r.text
    rec = r.json()
    assert rec["sha256"] == hashlib.sha256(tiny_webm).hexdigest() and rec["size_bytes"] == len(tiny_webm)

    d = client.get(f"/sessions/{sid}").json()
    assert d["status"] == "UPLOADED"
    assert [h["to_status"] for h in d["history"]] == ["CREATED", "CONSENTED", "RECORDING", "RECORDED", "UPLOADED"]
    assert d["event_counts"] == {"screen": 1, "device": 1} and d["dead_letter_count"] == 2
    lab = d["label"]
    assert lab["source"] == "mock" and lab["violation"] is False  # rehearsal script is non-violating
    assert lab["intervals"] == []  # its practice cue is not ground truth
    assert lab["labeler"].startswith("script:rehearsal-cues")

    # the stored file is real and byte-identical
    content = client.get(f"/recordings/{rec['id']}/content")
    assert content.status_code == 200 and hashlib.sha256(content.content).hexdigest() == rec["sha256"]

    # invalid transitions are refused with a readable message
    r = client.post(f"/sessions/{sid}/start", json={})
    assert r.status_code == 409 and "cannot move" in r.json()["detail"]

    # rehearsals never count toward the FR-2 corpus
    assert client.get("/corpus/mock-coverage").json()["eligible_sessions"] == 0

    # filters
    page = client.get("/sessions", params={"source": "MOCK", "q": p["code"]}).json()
    assert page["total"] == 1 and page["items"][0]["participant_code"] == p["code"]

    # deletion requires confirmation, then removes media from disk
    path = Path(api_env["storage"])
    assert any(path.rglob("*.webm"))
    assert client.post(f"/sessions/{sid}/delete", json={"confirm": False}).status_code == 422
    d = client.post(f"/sessions/{sid}/delete", json={"confirm": True}).json()
    assert d["status"] == "DELETED" and all(r["status"] == "PURGED" for r in d["recordings"])
    assert d["label"] is None and d["event_counts"] == {}
    assert not any(path.rglob(f"*{rec['id']}*"))
    assert client.get(f"/recordings/{rec['id']}/content").status_code == 404


def test_violation_script_label_and_corpus_eligibility(client, tiny_webm) -> None:  # type: ignore[no-untyped-def]
    p = _participant(client)
    sid = client.post(
        "/sessions",
        json={
            "participant_id": p["id"],
            "script_id": "phone_user",
            "lighting": "bright",
            "webcam_class": "hd",
            "room_noise": "quiet",
        },
    ).json()["id"]
    client.post(f"/sessions/{sid}/confirm-consent", json={"confirmed": True})
    client.post(f"/sessions/{sid}/start", json={})
    client.post(f"/sessions/{sid}/episodes/1", json={"shown_at_ms": 300050})
    client.post(f"/sessions/{sid}/episodes/2", json={"shown_at_ms": 840020})
    client.post(f"/sessions/{sid}/stop", json={"duration_s": 1260})
    r = client.post(
        f"/sessions/{sid}/recordings",
        data={"kind": "webcam_av", "duration_s": "1260"},
        files={"file": ("rec.webm", io.BytesIO(tiny_webm), "video/webm")},
    )
    assert r.status_code == 201
    lab = client.get(f"/sessions/{sid}").json()["label"]
    assert lab["violation"] is True and lab["violation_types"] == ["PHONE_USE"] and lab["confidence"] == "certain"
    assert lab["intervals"] == [
        {"start_ms": 300000, "end_ms": 345000, "type": "PHONE_USE"},
        {"start_ms": 840000, "end_ms": 870000, "type": "PHONE_USE"},
    ]
    cov = client.get("/corpus/mock-coverage").json()
    assert cov["eligible_sessions"] == 1 and cov["grid"]["bright"]["hd"] == 1 and cov["met"] is False


def test_helper_scripts_require_consenting_helper(client) -> None:  # type: ignore[no-untyped-def]
    p = _participant(client)
    body = {
        "participant_id": p["id"],
        "script_id": "second_person",
        "lighting": "normal",
        "webcam_class": "hd",
        "room_noise": "quiet",
    }
    assert client.post("/sessions", json=body).status_code == 422
    unconsented = client.post("/participants", json={"adult_confirmed": True}).json()
    assert client.post("/sessions", json={**body, "helper_participant_id": unconsented["id"]}).status_code == 409
    helper = _participant(client)
    assert client.post("/sessions", json={**body, "helper_participant_id": helper["id"]}).status_code == 201


def test_withdrawal_deletes_everything(client, tiny_webm, api_env) -> None:  # type: ignore[no-untyped-def]
    p = _participant(client)
    sid = client.post(
        "/sessions",
        json={
            "participant_id": p["id"],
            "script_id": "clean",
            "lighting": "normal",
            "webcam_class": "sd",
            "room_noise": "quiet",
        },
    ).json()["id"]
    client.post(f"/sessions/{sid}/confirm-consent", json={"confirmed": True})
    client.post(f"/sessions/{sid}/start", json={})
    client.post(f"/sessions/{sid}/stop", json={"duration_s": 1300})
    rec = client.post(
        f"/sessions/{sid}/recordings",
        data={"kind": "webcam_av", "duration_s": "1300"},
        files={"file": ("rec.webm", tiny_webm, "video/webm")},
    ).json()
    client.put(f"/participants/{p['id']}/demographics", json={"age_band": "25-34"})
    assert client.post(f"/participants/{p['id']}/withdraw", json={"confirm": False}).status_code == 422
    out = client.post(f"/participants/{p['id']}/withdraw", json={"confirm": True, "reason": "changed mind"}).json()
    assert out["status"] == "WITHDRAWN" and out["consent_active"] is False
    detail = client.get(f"/participants/{p['id']}").json()
    assert detail["has_demographics"] is False
    assert all(c["signed_name"] == "[redacted on withdrawal]" for c in detail["consents"])
    s = client.get(f"/sessions/{sid}").json()
    assert s["status"] == "DELETED" and s["label"] is None
    assert not any(Path(api_env["storage"]).rglob(f"*{rec['id']}*"))
    # a withdrawn participant cannot be recorded again
    r = client.post(
        "/sessions",
        json={
            "participant_id": p["id"],
            "script_id": "clean",
            "lighting": "normal",
            "webcam_class": "sd",
            "room_noise": "quiet",
        },
    )
    assert r.status_code == 409


def test_simulated_dataset_registration_and_read_only(client, api_env) -> None:  # type: ignore[no-untyped-def]
    from riskfusion.data.splits import create_splits_for_dataset
    from riskfusion.simulator import generate, load_config
    from riskfusion_api.db import _factory
    from riskfusion_api.services.datasets import register_simulated

    root = Path(api_env["data"])
    cfg = load_config(Path(__file__).resolve().parents[2] / "configs/simulator/v1.yaml", n_sessions=40, shard_size=20)
    res = generate(cfg, root)
    sm = create_splits_for_dataset(root, res.run_id, seed=1)
    with _factory()() as db:
        out = register_simulated(db, root, res.run_id)
    assert out == {"dataset_version": res.run_id, "sessions": 40, "labels": 40, "splits": 40}
    ds = client.get("/datasets").json()
    assert ds[0]["id"] == res.run_id and ds[0]["splits"]["counts"] == sm["counts"]
    page = client.get("/sessions", params={"source": "SIMULATED", "limit": 5}).json()
    assert page["total"] == 40 and page["items"][0]["split"] in ("train", "validation", "calibration", "holdout")
    sid = page["items"][0]["id"]
    assert client.post(f"/sessions/{sid}/start", json={}).status_code == 409
    held = client.get("/sessions", params={"source": "SIMULATED", "limit": 500}).json()["items"]
    sealed = [x for x in held if x["split"] == "holdout"]
    assert sealed and all(x["violation_label"] is None for x in sealed)
    assert client.get(f"/sessions/{sealed[0]['id']}").json()["label"] is None
    assert client.get(f"/sessions/{sealed[0]['id']}/signal-timeline").status_code == 403
    for flag in ("true", "false"):
        items = client.get("/sessions", params={"source": "SIMULATED", "violation": flag, "limit": 500}).json()["items"]
        assert all(x["split"] != "holdout" for x in items)
    open_id = next(x["id"] for x in held if x["split"] == "train")
    tl = client.get(f"/sessions/{open_id}/signal-timeline").json()
    assert tl["n_events"] > 1000 and {lane["label"] for lane in tl["lanes"]} >= {"Gaze off screen", "Tab hidden"}
    bd = client.get(f"/datasets/{res.run_id}/breakdown").json()
    assert sum(r["sessions"] for r in bd["lighting"]) == 40
    status = client.get("/system/status").json()
    assert status["schema_revision"] == "0002"
    assert all(s["state"] == "available" for s in status["pipeline"])
