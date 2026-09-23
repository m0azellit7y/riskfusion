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

from . import body as B
from . import vision as V

VERSION = "extract-v1.0"
DET = {
    "presence": ("scrfd_500m", "insightface-buffalo_s"),
    "identity": ("arcface_mbf", "insightface-w600k_mbf"),
    "attention": ("headpose_pnp", "pnp5-v1"),
    "environment": ("yolox_nano", "yolox-0.1.1rc0"),
    "liveness": ("facemesh_blink_motion", "mediapipe-face_landmark-onnx+heuristic-v1"),
    "pose": ("pose_landmark_palm", "mediapipe-pose_landmark_full+palm_lite-onnx"),
    "audio_voice": ("energy_vad_mfcc_speaker", "heuristic-v1"),
    "audio_event": ("dsp_audio_tagger", "heuristic-v1"),
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
    body_ok = B.available()
    live = B.LivenessTracker()
    last_face: V.Face | None = None
    burst_end: float | None = None
    burst_slot = -1

    def close_burst(at_ms: float) -> None:
        res = live.close_burst(int(at_ms))
        if res is None:
            events.append(_ev(sid, int(at_ms), "liveness", "LIVENESS_UNKNOWN", {"reason": "occluded"}, 1.0))
        elif res[0] == "unknown":
            events.append(_ev(sid, int(at_ms), "liveness", "LIVENESS_UNKNOWN", res[1], 1.0))
        else:
            events.append(_ev(sid, int(at_ms), "liveness", "LIVENESS_CHECK", res[1], 0.7))

    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        last_t = max(last_t, t)
        # liveness burst: 3 s examined at ~15 frames/s, every 10 s (blinks are 100-400 ms long)
        if burst_end is not None:
            if t <= burst_end:
                slot = int(t // 66)
                if last_face is not None and slot != burst_slot:
                    burst_slot = slot
                    ok_b, bframe = cap.retrieve()
                    if ok_b and bframe is not None:
                        fm = B.face_mesh(bframe, last_face)
                        if fm is not None:
                            live.burst_sample(int(t), fm[0], fm[1], float(last_face.box[3] - last_face.box[1]))
            else:
                close_burst(burst_end)
                burst_end = None
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
        last_face = main
        if body_ok and main is not None:
            fm = B.face_mesh(frame, main)
            if fm is not None:
                live.open_sample(fm[0])
        if sec % 10 == 0 and burst_end is None:
            if not body_ok:
                events.append(_ev(sid, ts + 300, "liveness", "LIVENESS_UNKNOWN", {"reason": "detector_error"}, 1.0))
            elif main is None:
                events.append(_ev(sid, ts + 300, "liveness", "LIVENESS_UNKNOWN", {"reason": "occluded"}, 1.0, q))
            else:
                burst_end = t + 3000.0
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
            bp = B.body_pose(frame, main) if (body_ok and main is not None) else None
            if bp is None:
                # elbows not in view (normal head-and-shoulders framing): hands cannot be judged -> UNKNOWN
                events.append(_ev(sid, ts + 150, "pose", "POSE_UNKNOWN", {"reason": "not_observable"}, 1.0, q))
            else:
                events.append(
                    _ev(sid, ts + 150, "pose", "BODY_POSE", {"hands_visible_count": B.count_hands(frame), **bp}, 0.7, q)
                )
        if progress and frames % 30 == 0:
            progress(last_t / 1000.0)
    if burst_end is not None:
        close_burst(min(burst_end, last_t))
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


def _audio_unknown(sid: str, reason: str) -> list[dict[str, Any]]:
    """No usable audio: BOTH audio channels are explicitly unknown (FR-7)."""
    return [
        _ev(sid, 0, "audio_voice", "AUDIO_VOICE_UNKNOWN", {"reason": reason}, 1.0),
        _ev(sid, 0, "audio_event", "AUDIO_EVENT_UNKNOWN", {"reason": reason}, 1.0),
    ]


def extract_audio(sid: str, video: Path) -> list[dict[str, Any]]:
    """Energy VAD with an adaptive noise floor + an MFCC two-cluster speaker heuristic.

    Honest limits: there is no enrolled voice and no neural speaker model in this build. A second voice is
    reported only when voiced seconds split into two well-separated MFCC clusters and the candidate's cluster
    is the one dominating the start of the session. Confidence is capped at 0.6 to reflect that.
    """
    if shutil.which("ffmpeg") is None:
        return _audio_unknown(sid, "detector_error")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True,
        check=False,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    sr = 16000
    if proc.returncode != 0 or pcm.size < sr:
        return _audio_unknown(sid, "no_frames")
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
        return _audio_unknown(sid, "no_frames")
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
    out.extend(audio_events(sid, energy_db, spec, floor, voiced, n_sec))
    return out


def audio_events(
    sid: str, energy_db: np.ndarray, spec: np.ndarray, floor: np.ndarray, voiced: np.ndarray, n_sec: int
) -> list[dict[str, Any]]:
    """Signal-processing audio tagger (no licensed neural tagger in this build). Per second, at most one label:

    * phone_ring    — a strong, stable tone in 0.7-3.5 kHz for >= 60% of the second (ringtones are stationary;
                      speech pitch moves)
    * door          — one very loud, low-frequency transient
    * paper_rustle  — broadband, noise-like (spectrally flat) sound above the room's noise floor; speech is
                      harmonic, so it is not flat
    * keyboard_burst — >= 3 short, bright clicks
    Confidence is 0.5: these are heuristics and are treated as weak evidence.
    """
    hz = 16000 / 512
    band = slice(int(700 / hz), int(3500 / hz))
    hi = slice(int(2000 / hz), int(7000 / hz))
    freqs = np.arange(spec.shape[1]) * hz
    # loudness of the 2-7 kHz band and its own rolling noise floor: rustling raises THIS band, a voice does not
    hb_db = 10 * np.log10(spec[: n_sec * 100, hi].sum(1) + 1e-12).reshape(n_sec, 100)
    hb_floor = np.array([np.percentile(hb_db[max(0, k - 30) : k + 1].ravel(), 10) for k in range(n_sec)])
    out: list[dict[str, Any]] = []
    for s in range(n_sec):
        fr = slice(s * 100, s * 100 + 100)
        e, sp, fl = energy_db[fr], spec[fr], floor[s]
        loud = e > fl + 12
        label = None
        if loud.sum() >= 60:
            b = sp[:, band]
            peak = b.max(1) / (b.mean(1) + 1e-12)
            pk = b.argmax(1)
            tonal = loud & (peak > 30)
            if tonal.sum() >= 60 and np.std(pk[tonal]) < 3:
                label = "phone_ring"
        cent = (sp * freqs).sum(1) / (sp.sum(1) + 1e-12)
        if label is None:
            jump = e - np.median(e)
            k = int(np.argmax(jump))
            g = s * 100 + k  # global frame index of the loudest moment
            after = energy_db[g + 20 : g + 35]
            # a knock/door is short: it has faded 15 dB within ~0.3 s (a voice stays loud); peaks in the last
            # frames of a second are left to the next second so one knock is reported once
            if jump[k] > 25 and cent[k] < 800 and k < 95 and after.size and after.max() < e[k] - 15:
                label = "door"
        if label is None:
            h = sp[:, hi] + 1e-12
            flat = np.exp(np.log(h).mean(1)) / h.mean(1)
            if ((hb_db[s] > hb_floor[s] + 10) & (flat > 0.4)).sum() >= 40:
                label = "paper_rustle"
        if label is None:
            onset = np.diff(e, prepend=e[:1]) > 10
            clicks = onset & (cent > 2000) & (e > fl + 10)
            if clicks.sum() >= 3:
                label = "keyboard_burst"
        if label:
            out.append(_ev(sid, s * 1000 + 30, "audio_event", "AUDIO_EVENT", {"label": label}, 0.5))
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
