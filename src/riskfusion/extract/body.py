"""Face mesh (liveness), body pose and hand detection — MediaPipe models converted to ONNX.

Heuristics layered on the models are documented where they are used; they are deliberately conservative:
anything that cannot be judged from the camera framing is reported as UNKNOWN, never as suspicious.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .vision import Face

MP_DIR = Path(__file__).resolve().parents[3] / "models" / "mediapipe"

# MediaPipe face-mesh indices: eye corners and eyelid midpoints
_LEFT = (33, 133, 159, 145)
_RIGHT = (362, 263, 386, 374)
_NOSE = 1


@lru_cache
def _session(name: str) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(MP_DIR / name), sess_options=opts, providers=["CPUExecutionProvider"])


def available() -> bool:
    return all(
        (MP_DIR / n).exists() for n in ("face_landmark.onnx", "pose_landmark_full.onnx", "palm_detection_lite.onnx")
    )


def _sig(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------------------------ face mesh
def face_mesh(frame: np.ndarray, face: Face) -> tuple[float, np.ndarray] | None:
    """Eye aspect ratio (mean of both eyes) and the nose tip in frame pixels, or None if no face."""
    x1, y1, x2, y2 = face.box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1) * 1.6
    crop = cv2.getRectSubPix(frame, (int(size), int(size)), (float(cx), float(cy)))
    inp = cv2.cvtColor(cv2.resize(crop, (192, 192)), cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0
    lm, flag = _session("face_landmark.onnx").run(None, {"input_1": inp})
    if float(_sig(float(flag.ravel()[0]))) < 0.5:
        return None
    pts = lm.reshape(-1, 3)[:, :2]

    def ear(ix: tuple[int, int, int, int]) -> float:
        a, b, c, d = ix
        return float(np.linalg.norm(pts[c] - pts[d]) / max(np.linalg.norm(pts[a] - pts[b]), 1e-6))

    nose = pts[_NOSE] / 192.0 * size + np.array([cx - size / 2, cy - size / 2])
    return (ear(_LEFT) + ear(_RIGHT)) / 2.0, nose


class LivenessTracker:
    """Passive liveness from blinks and micro-motion.

    A live person blinks every few seconds; a printed photo or a frozen image never does. Every 10 s the video
    is examined for 3 s at 15 frames/s. The check PASSES when a blink was seen in the last 60 s. During the first
    60 s without a blink the channel is UNKNOWN (not yet judgeable), never a failure. Known limit: a replayed
    video of a real person blinks and passes.
    """

    # a blink = a brief dip below 80% of the person's open-eye level while the eyes are open for most of the
    # burst; looking down or squinting lowers the whole burst and is therefore NOT counted as a blink
    BLINK_RATIO = 0.80
    OPEN_RATIO = 0.90

    def __init__(self) -> None:
        self.open_ears: list[float] = []
        self.blinks_ms: list[int] = []
        self.first_ms: int | None = None
        self.burst_ears: list[float] = []
        self.burst_ts: list[int] = []
        self.burst_noses: list[np.ndarray] = []
        self.face_size = 1.0

    def open_sample(self, ear: float) -> None:
        self.open_ears.append(ear)

    def burst_sample(self, t_ms: int, ear: float, nose: np.ndarray, face_size: float) -> None:
        if self.first_ms is None:
            self.first_ms = t_ms
        self.burst_ears.append(ear)
        self.burst_ts.append(t_ms)
        self.burst_noses.append(nose)
        self.face_size = face_size

    def close_burst(self, t_ms: int) -> tuple[str, dict[str, Any]] | None:
        """Returns ('check', payload) or ('unknown', {'reason': ...}) or None if the burst had no face."""
        n = len(self.burst_ears)
        noses = np.array(self.burst_noses) if n else np.zeros((0, 2))
        ears = np.array(self.burst_ears)
        ts = list(self.burst_ts)
        self.burst_ears, self.burst_noses, self.burst_ts = [], [], []
        if n < 5:
            return None
        ref = float(np.median(self.open_ears[-120:])) if len(self.open_ears) >= 5 else float(np.percentile(ears, 80))
        if ears.min() < self.BLINK_RATIO * ref and np.median(ears) > self.OPEN_RATIO * ref:
            self.blinks_ms.append(ts[int(np.argmin(ears))])
        motion = float(np.std(noses, axis=0).mean() / max(self.face_size, 1.0))
        blinked = any(t_ms - b <= 60_000 for b in self.blinks_ms)
        if not blinked and self.first_ms is not None and t_ms - self.first_ms < 60_000:
            return "unknown", {"reason": "not_observable"}
        score = float(np.clip(0.2 + 0.6 * blinked + 0.2 * min(1.0, motion / 0.004), 0.0, 1.0))
        return "check", {"pad_score": score, "pad_pass": bool(blinked)}


# ------------------------------------------------------------------------------------ body pose
_SH_L, _SH_R, _EL_L, _EL_R, _WR_L, _WR_R = 11, 12, 13, 14, 15, 16


def body_pose(frame: np.ndarray, face: Face) -> dict[str, Any] | None:
    """Pose landmarks on a square crop around the upper body located from the face box.

    Returns None when the framing does not show the elbows — then hands and reaching cannot be judged
    (a head-and-shoulders webcam view is normal and must not look suspicious)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face.box
    fh = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + fh * 1.2
    size = max(fh * 4.0, 64)
    crop = cv2.getRectSubPix(frame, (int(size), int(size)), (float(cx), float(cy)))
    inp = cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0
    out = _session("pose_landmark_full.onnx").run(None, {"input_1": inp})
    if float(_sig(float(out[1].ravel()[0]))) < 0.5:
        return None
    lm = out[0].reshape(-1, 5)
    vis = _sig(lm[:, 3])
    # back to normalised frame coordinates
    px = (lm[:, 0] / 256.0 * size + cx - size / 2) / w
    py = (lm[:, 1] / 256.0 * size + cy - size / 2) / h
    shoulders_ok = vis[_SH_L] > 0.5 and vis[_SH_R] > 0.5
    elbows_in = [vis[i] > 0.5 and 0 <= px[i] <= 1 and 0 <= py[i] <= 1 for i in (_EL_L, _EL_R)]
    if not (shoulders_ok and any(elbows_in)):
        return None
    reach = any(vis[e] > 0.5 and (px[wr] < -0.02 or px[wr] > 1.02) for e, wr in ((_EL_L, _WR_L), (_EL_R, _WR_R)))
    dz = float(lm[_SH_L, 2] - lm[_SH_R, 2])
    dx = float(lm[_SH_L, 0] - lm[_SH_R, 0])
    torso = float(np.degrees(np.arctan2(dz, abs(dx) + 1e-6)))
    return {"reach_off_frame": bool(reach), "torso_rotation": float(np.clip(torso, -180, 180))}


# ------------------------------------------------------------------------------------ hands
@lru_cache
def _palm_anchors() -> np.ndarray:
    anchors = []
    for stride, per in ((8, 2), (16, 6)):
        g = 192 // stride
        for y in range(g):
            for x in range(g):
                for _ in range(per):
                    anchors.append(((x + 0.5) / g, (y + 0.5) / g))
    return np.array(anchors, dtype=np.float32)  # 24*24*2 + 12*12*6 = 2016


def count_hands(frame: np.ndarray, score_thr: float = 0.55) -> int:
    h, w = frame.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=np.uint8)
    canvas[:h, :w] = frame
    inp = cv2.cvtColor(cv2.resize(canvas, (192, 192)), cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0
    boxes, scores = _session("palm_detection_lite.onnx").run(None, {"input_1": inp})
    sc = _sig(np.clip(scores.reshape(-1), -80, 80))
    keep = np.flatnonzero(sc >= score_thr)
    if keep.size == 0:
        return 0
    a = _palm_anchors()[keep]
    b = boxes.reshape(-1, 18)[keep]
    cx = a[:, 0] + b[:, 0] / 192.0
    cy = a[:, 1] + b[:, 1] / 192.0
    bw, bh = b[:, 2] / 192.0, b[:, 3] / 192.0
    rects = [
        [float(x - ww / 2), float(y - hh / 2), float(ww), float(hh)]
        for x, y, ww, hh in zip(cx, cy, bw, bh, strict=True)
    ]
    idx = cv2.dnn.NMSBoxes(rects, sc[keep].astype(float).tolist(), score_thr, 0.3)
    return int(min(4, len(np.array(idx).reshape(-1))))
