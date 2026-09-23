"""Run the pretrained detectors over one recording and emit event.v1 events (FR-4, FR-7).

Video is sampled at 1 frame/s (identity every 5 s). Audio is decoded with ffmpeg at 16 kHz mono.
Channels without a suitable licensed model in this build emit explicit UNKNOWN events instead of
guesses: liveness (no permissively licensed PAD model, SRS_AUDIT A-17) and body pose.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.fft import dct
from scipy.signal import get_window

from riskfusion.contracts import event_errors

from . import vision as V

VERSION = "extract-v1.0"
DET = {
    "presence": ("scrfd_500m", "insightface-buffalo_s"),
    "identity": ("arcface_mbf", "insightface-w600k_mbf"),
    "attention": ("headpose_pnp", "pnp5-v1"),
    "environment": ("yolox_nano", "yolox-0.1.1rc0"),
    "liveness": ("none", VERSION),
    "pose": ("none", VERSION),
    "audio_voice": ("energy_vad_mfcc_speaker", "heuristic-v1"),
    "audio_event": ("none", VERSION),
}
IDENTITY_THRESHOLD = 0.35  # ArcFace cosine; typical verification threshold for w600k_mbf


def _ev(
    sid: str,
    ts: int,
    channel: str,
    etype: str,
    payload: dict[str, Any],
    conf: float,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    det, ver = DET[channel]
    ev = {
        "schema": "event.v1",
        "session_id": sid,
        "ts_ms": int(max(0, ts)),
        "channel": channel,
        "detector": det,
        "detector_version": ver,
        "event_type": etype,
        "payload": payload,
        "confidence": float(np.clip(conf, 0, 1)),
    }
    if quality is not None:
        ev["quality"] = quality
    return ev


def enrolment_embedding(image_path: Path) -> np.ndarray | None:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    faces = V.detect_faces(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
    return V.face_embedding(img, face)


# ------------------------------------------------------------------------------------ video
def extract_video(
    sid: str, video: Path, enrol: np.ndarray | None, fps: float = 1.0, progress: Any = None
) -> tuple[list[dict[str, Any]], float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("the video could not be decoded")
    events: list[dict[str, Any]] = []
    next_t = 0.0
    last_t = 0.0
    frames = 0
    period = 1000.0 / fps
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        last_t = max(last_t, t)
        if t + 1e-6 < next_t:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue
        next_t = (np.floor(t / period) + 1) * period
        ts = int(round(t))
        sec = int(ts // 1000)
        frames += 1
        luma, blur = V.frame_quality(frame)
        q = {"frame_luma": round(luma, 2), "frame_blur": round(blur, 3), "usable": bool(luma > 32 and blur < 0.85)}
        if not q["usable"]:
            reason = "low_light" if luma <= 32 else "occluded"
            for ch in ("presence", "attention", "environment"):
                events.append(_ev(sid, ts, ch, f"{ch.upper()}_UNKNOWN", {"reason": reason}, 1.0, q))
            if sec % 5 == 0:
                events.append(_ev(sid, ts + 200, "identity", "IDENTITY_UNKNOWN", {"reason": reason}, 1.0, q))
            continue
        h, w = frame.shape[:2]
        faces = V.detect_faces(frame)
        main = max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1])) if faces else None
        if main is not None:
            bw, bh = (main.box[2] - main.box[0]) / w, (main.box[3] - main.box[1]) / h
            cx, cy = (main.box[0] + main.box[2]) / 2 / w, (main.box[1] + main.box[3]) / 2 / h
            payload = {
                "face_count": len(faces),
                "bbox_area_ratio": float(np.clip(bw * bh, 0, 1)),
                "center_offset_x": float(np.clip(cx * 2 - 1, -1, 1)),
                "center_offset_y": float(np.clip(cy * 2 - 1, -1, 1)),
            }
            conf = main.score
        else:
            payload = {"face_count": 0, "bbox_area_ratio": 0.0, "center_offset_x": 0.0, "center_offset_y": 0.0}
            conf = 0.9
        events.append(_ev(sid, ts, "presence", "FACE_OBSERVATION", payload, conf, q))
        if main is not None:
            yaw, pitch, roll = V.head_pose(main, frame.shape)
            events.append(
                _ev(
                    sid,
                    ts + 5,
                    "attention",
                    "HEAD_GAZE",
                    {
                        "yaw": float(np.clip(yaw, -180, 180)),
                        "pitch": float(np.clip(pitch, -180, 180)),
                        "roll": float(np.clip(roll, -180, 180)),
                        "gaze_on_screen_prob": V.gaze_on_screen_prob(yaw, pitch),
                    },
                    main.score * 0.9,
                    q,
                )
            )
        else:
            events.append(_ev(sid, ts + 5, "attention", "ATTENTION_UNKNOWN", {"reason": "occluded"}, 1.0, q))
        if sec % 5 == 0:
            if enrol is None:
                events.append(_ev(sid, ts + 200, "identity", "IDENTITY_UNKNOWN", {"reason": "not_observable"}, 1.0))
            elif main is None:
                events.append(_ev(sid, ts + 200, "identity", "IDENTITY_UNKNOWN", {"reason": "occluded"}, 1.0, q))
            else:
                sim = float(np.clip(enrol @ V.face_embedding(frame, main), -1, 1))
                fq = float(np.clip(1 - blur, 0, 1) * min(1.0, main.score / 0.8))
                events.append(
                    _ev(
                        sid,
                        ts + 200,
                        "identity",
                        "IDENTITY_CHECK",
                        {
                            "cosine_sim": sim,
                            "verified": bool(sim >= IDENTITY_THRESHOLD),
                            "face_quality": fq,
                        },
                        main.score,
                        q,
                    )
                )
            events.append(_ev(sid, ts + 300, "liveness", "LIVENESS_UNKNOWN", {"reason": "not_observable"}, 1.0))
        if sec % 2 == 0:
            dets = V.detect_objects(frame)
            persons = sum(1 for d in dets if d.label == "person")
            events.append(
                _ev(sid, ts + 100, "environment", "PERSON_COUNT", {"n_persons": int(max(persons, len(faces)))}, 0.8, q)
            )
            for d in dets:
                if d.label == "person":
                    continue
                x, y, bw2, bh2 = (float(v) for v in d.box)
                events.append(
                    _ev(
                        sid,
                        ts + 120,
                        "environment",
                        "OBJECT_DETECTED",
                        {"label": d.label, "bbox_x": x, "bbox_y": y, "bbox_w": bw2, "bbox_h": bh2},
                        d.score,
                        q,
                    )
                )
            events.append(_ev(sid, ts + 150, "pose", "POSE_UNKNOWN", {"reason": "not_observable"}, 1.0))
        if progress and frames % 30 == 0:
            progress(last_t / 1000.0)
    cap.release()
    if frames == 0:
        events.append(_ev(sid, 0, "presence", "PRESENCE_UNKNOWN", {"reason": "no_frames"}, 1.0))
    return events, last_t / 1000.0


# ------------------------------------------------------------------------------------ audio
def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 26) -> np.ndarray:
    def hz2mel(h: np.ndarray) -> np.ndarray:
        return 2595 * np.log10(1 + h / 700)

    def mel2hz(m: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz2mel(np.array(80.0)), hz2mel(np.array(sr / 2)), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        a, b, c = bins[i - 1], bins[i], bins[i + 1]
        fb[i - 1, a:b] = (np.arange(a, b) - a) / max(1, b - a)
        fb[i - 1, b:c] = (c - np.arange(b, c)) / max(1, c - b)
    return fb


def extract_audio(sid: str, video: Path) -> list[dict[str, Any]]:
    """Energy VAD with an adaptive noise floor + an MFCC two-cluster speaker heuristic.

    Honest limits: there is no enrolled voice and no neural speaker model in this build. A second voice is
    reported only when voiced seconds split into two well-separated MFCC clusters and the candidate's cluster
    is the one dominating the start of the session. Confidence is capped at 0.6 to reflect that.
    """
    if shutil.which("ffmpeg") is None:
        return [_ev(sid, 0, "audio_voice", "AUDIO_VOICE_UNKNOWN", {"reason": "detector_error"}, 1.0)]
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True,
        check=False,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    sr = 16000
    if proc.returncode != 0 or pcm.size < sr:
        return [_ev(sid, 0, "audio_voice", "AUDIO_VOICE_UNKNOWN", {"reason": "no_frames"}, 1.0)]
    hop, win = 160, 400
    n_frames = 1 + (pcm.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = pcm[idx] * get_window("hamming", win)[None, :]
    energy_db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    n_fft = 512
    spec = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    mfcc = dct(np.log(spec @ _mel_filterbank(sr, n_fft).T + 1e-10), type=2, axis=1, norm="ortho")[:, 1:14]
    per_sec = 100
    n_sec = n_frames // per_sec
    if n_sec < 1:
        return [_ev(sid, 0, "audio_voice", "AUDIO_VOICE_UNKNOWN", {"reason": "no_frames"}, 1.0)]
    e = energy_db[: n_sec * per_sec].reshape(n_sec, per_sec)
    floor = np.array([np.percentile(e[max(0, s - 30) : s + 1].ravel(), 10) for s in range(n_sec)])
    voiced_frames = e > (floor[:, None] + 12.0)
    voiced = voiced_frames.mean(axis=1) > 0.25
    m = mfcc[: n_sec * per_sec].reshape(n_sec, per_sec, -1)
    sec_mfcc = np.array(
        [m[s][voiced_frames[s]].mean(0) if voiced_frames[s].any() else np.zeros(13) for s in range(n_sec)]
    )
    foreign = np.zeros(n_sec, dtype=bool)
    vi = np.flatnonzero(voiced)
    if vi.size >= 20:
        X = sec_mfcc[vi]
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(X)
        lab = km.labels_
        sep = np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1])
        first = lab[: max(10, vi.size // 5)]
        cand = int(np.bincount(first, minlength=2).argmax())
        other_share = float((lab != cand).mean())
        if sep > 2.5 and 0.05 <= other_share <= 0.6:
            foreign[vi[lab != cand]] = True
    out = []
    for s in range(n_sec):
        out.append(
            _ev(
                sid,
                s * 1000 + 10,
                "audio_voice",
                "VOICE_ACTIVITY",
                {
                    "vad_active": bool(voiced[s]),
                    "speaker_is_candidate": bool(not foreign[s]),
                    "n_speakers": int(voiced[s]) + int(foreign[s]),
                    "foreign_speech": bool(foreign[s]),
                },
                0.6 if foreign[s] else 0.8,
            )
        )
    out.append(_ev(sid, 30, "audio_event", "AUDIO_EVENT_UNKNOWN", {"reason": "not_observable"}, 1.0))
    return out


def extract_recording(
    sid: str, video: Path, enrolment_image: Path | None, progress: Any = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enrol = enrolment_embedding(enrolment_image) if enrolment_image else None
    vid_events, dur = extract_video(sid, video, enrol, progress=progress)
    aud_events = extract_audio(sid, video)
    events = [e for e in vid_events + aud_events if not event_errors(e)]
    for e in events:
        e["event_uid"] = uuid.uuid4().hex
    meta = {
        "version": VERSION,
        "duration_s": dur,
        "enrolment_face_found": enrol is not None,
        "n_events": len(events),
        "detectors": {k: v[0] for k, v in DET.items()},
    }
    return events, meta
