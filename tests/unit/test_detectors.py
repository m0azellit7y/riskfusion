"""All 11 channels from real media: liveness (blinks), pose framing rule, audio events, and one combined run."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pytest
from skimage import data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from make_media import audio_events_file, face_canvas, liveness_video  # noqa: E402

from riskfusion.contracts import CHANNELS, event_errors  # noqa: E402
from riskfusion.extract import body as B  # noqa: E402
from riskfusion.extract import vision as V  # noqa: E402
from riskfusion.extract.pipeline import enrolment_embedding, extract_audio, extract_video  # noqa: E402

pytestmark = pytest.mark.skipif(not B.available(), reason="MediaPipe ONNX models not present")


@pytest.fixture(scope="module")
def enrol(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    p = tmp_path_factory.mktemp("e") / "enrol.jpg"
    cv2.imwrite(str(p), cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
    return enrolment_embedding(p)


def _liveness(events: list[dict]) -> list[tuple[str, object]]:  # type: ignore[type-arg]
    return [(e["event_type"], e["payload"].get("pad_pass")) for e in events if e["channel"] == "liveness"]


def test_printed_photo_fails_liveness_after_grace_period(tmp_path: Path, enrol) -> None:  # type: ignore[no-untyped-def]
    ev, _ = extract_video("sess_photo_01", liveness_video(tmp_path / "photo.webm", blink=False), enrol)
    lv = _liveness(ev)
    early = [x for x, e in zip(lv, [e for e in ev if e["channel"] == "liveness"], strict=True) if e["ts_ms"] < 60_000]
    late = [x for x, e in zip(lv, [e for e in ev if e["channel"] == "liveness"], strict=True) if e["ts_ms"] >= 62_000]
    assert all(t == "LIVENESS_UNKNOWN" for t, _ in early)  # no judgement in the first minute
    assert late and all(t == "LIVENESS_CHECK" and p is False for t, p in late)


def test_blinking_face_passes_liveness(tmp_path: Path, enrol) -> None:  # type: ignore[no-untyped-def]
    ev, _ = extract_video("sess_blink_01", liveness_video(tmp_path / "blink.webm", blink=True), enrol)
    checks = [p for t, p in _liveness(ev) if t == "LIVENESS_CHECK"]
    assert len(checks) >= 8 and all(checks)


def test_head_and_shoulders_framing_is_unknown_not_suspicious() -> None:
    frame = face_canvas()
    face = V.detect_faces(frame)[0]
    fm = B.face_mesh(frame, face)
    assert fm is not None and 0.25 < fm[0] < 0.5  # open eyes
    assert B.count_hands(frame) == 0


def test_audio_events_detected_at_the_right_time(tmp_path: Path) -> None:
    ev = extract_audio("sess_audio_01", audio_events_file(tmp_path / "a.webm"))
    got = {(e["ts_ms"] // 1000, e["payload"]["label"]) for e in ev if e["event_type"] == "AUDIO_EVENT"}
    expected = (
        {(s, "phone_ring") for s in (10, 11, 12)}
        | {(s, "keyboard_burst") for s in (20, 21, 22)}
        | {(s, "paper_rustle") for s in (30, 31, 32)}
        | {(40, "door")}
    )
    assert got == expected  # every sound found, nothing invented (the voice at 48-52 s is not labelled)


def test_one_pass_produces_all_eleven_channels(tmp_path: Path, enrol) -> None:  # type: ignore[no-untyped-def]
    """Video + audio detectors together cover 8 channels; the browser adds screen, device and behavioral."""
    vid = liveness_video(tmp_path / "all.webm", blink=True, seconds=70)
    ev, _ = extract_video("sess_all_01", vid, enrol)
    ev += extract_audio("sess_all_01", vid)
    assert all(not event_errors(e) for e in ev)
    covered = {e["channel"] for e in ev}
    assert covered >= set(CHANNELS) - {"screen", "device", "behavioral"}
    assert any(e["event_type"] == "LIVENESS_CHECK" for e in ev)
