"""Vision detectors on individual frames (BGR uint8).

* SCRFD-500M face detector (5 landmarks)          -> presence
* ArcFace MobileFaceNet embedding vs enrolment      -> identity
* Head pose from the 5 landmarks (solvePnP)         -> attention (gaze is a head-pose *proxy*, see docstring)
* YOLOX-nano (COCO)                                 -> environment: persons, phones, books, laptops, TVs
* Frame quality: luma and a blur score              -> quality / UNKNOWN states
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import session


# ------------------------------------------------------------------------------------ quality
def frame_quality(frame: np.ndarray) -> tuple[float, float]:
    """Return (luma 0-255, blur 0-1 where 1 = very blurry)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    luma = float(gray.mean())
    sharp = float(cv2.Laplacian(cv2.resize(gray, (320, 240)), cv2.CV_64F).var())
    blur = float(np.clip(1.0 - np.log1p(sharp) / np.log1p(500.0), 0.0, 1.0))
    return luma, blur


# ------------------------------------------------------------------------------------ faces
@dataclass
class Face:
    box: np.ndarray  # x1, y1, x2, y2 (pixels)
    score: float
    kps: np.ndarray  # 5 x 2


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    idx = cv2.dnn.NMSBoxes(
        [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
        scores.astype(float).tolist(),
        0.0,
        thr,
    )
    return [int(i) for i in np.array(idx).reshape(-1)]


def detect_faces(frame: np.ndarray, score_thr: float = 0.5, size: int = 640) -> list[Face]:
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
    sess = session("det_500m.onnx")
    outs = sess.run(None, {sess.get_inputs()[0].name: blob})
    boxes, scores, kpss = [], [], []
    for i, stride in enumerate((8, 16, 32)):
        sc, bb, kp = outs[i][:, 0], outs[i + 3] * stride, outs[i + 6] * stride
        fh, fw = size // stride, size // stride
        ys, xs = np.mgrid[:fh, :fw]
        centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
        centers = np.repeat(centers, 2, axis=0)  # 2 anchors per location
        keep = np.flatnonzero(sc >= score_thr)
        if keep.size == 0:
            continue
        c = centers[keep]
        b = bb[keep]
        boxes.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], 1))
        k = kp[keep].reshape(-1, 5, 2) + c[:, None, :]
        kpss.append(k)
        scores.append(sc[keep])
    if not boxes:
        return []
    B, S, K = np.concatenate(boxes) / scale, np.concatenate(scores), np.concatenate(kpss) / scale
    return [Face(B[i], float(S[i]), K[i]) for i in _nms(B, S, 0.4)]


_ARC_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def face_embedding(frame: np.ndarray, face: Face) -> np.ndarray:
    M, _ = cv2.estimateAffinePartial2D(face.kps.astype(np.float32), _ARC_TEMPLATE, method=cv2.LMEDS)
    crop = cv2.warpAffine(frame, M, (112, 112), borderValue=0)
    blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    sess = session("w600k_mbf.onnx")
    emb = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
    return emb / (np.linalg.norm(emb) + 1e-9)


# Rough 3-D positions (mm) of the SCRFD landmarks on an average head: eyes, nose tip, mouth corners.
_MODEL_3D = np.array(
    [[-32.0, 35.0, -28.0], [32.0, 35.0, -28.0], [0.0, 0.0, 0.0], [-26.0, -32.0, -24.0], [26.0, -32.0, -24.0]],
    dtype=np.float64,
)


def head_pose(face: Face, frame_shape: tuple[int, ...]) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees; yaw > 0 = turned to the image's right, pitch < 0 = looking down."""
    h, w = frame_shape[:2]
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    img = face.kps.astype(np.float64).copy()
    img[:, 1] = h - img[:, 1]  # model y axis points up
    cam_up = cam.copy()
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, img, cam_up, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = np.hypot(R[0, 0], R[1, 0])
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = (pitch + 180) % 360 - 180
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return float(-yaw), float(pitch), float(roll)


def gaze_on_screen_prob(yaw: float, pitch: float) -> float:
    """Head-pose proxy for 'looking at the screen'. There is no eye-gaze model in this build (see SRS_AUDIT)."""
    dev = np.hypot(yaw / 25.0, pitch / 20.0)
    return float(1.0 / (1.0 + np.exp((dev - 1.0) * 5.0)))


# ------------------------------------------------------------------------------------ objects
COCO = {0: "person", 63: "laptop", 62: "tv", 65: "remote", 67: "cell_phone", 73: "book"}


@dataclass
class Detection:
    label: str
    score: float
    box: np.ndarray  # normalised x, y, w, h


def detect_objects(frame: np.ndarray, score_thr: float = 0.35, size: int = 416) -> list[Detection]:
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(h * r), int(w * r)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
    sess = session("yolox_nano.onnx")
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
    grids, strides = [], []
    for s in (8, 16, 32):
        g = size // s
        ys, xs = np.mgrid[:g, :g]
        grids.append(np.stack([xs, ys], -1).reshape(-1, 2))
        strides.append(np.full((g * g, 1), s))
    grid = np.concatenate(grids).astype(np.float32)
    stride = np.concatenate(strides).astype(np.float32)
    xy = (out[:, :2] + grid) * stride
    wh = np.exp(out[:, 2:4]) * stride
    cls = out[:, 5:]
    cls_id = cls.argmax(1)
    conf = out[:, 4] * cls[np.arange(len(cls)), cls_id]
    keep = np.flatnonzero((conf >= score_thr) & np.isin(cls_id, list(COCO)))
    if keep.size == 0:
        return []
    boxes = np.concatenate([xy[keep] - wh[keep] / 2, xy[keep] + wh[keep] / 2], 1) / r
    dets = []
    for i in _nms(boxes, conf[keep], 0.45):
        x1, y1, x2, y2 = boxes[i]
        nb = np.clip(np.array([x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]), 0, 1)
        dets.append(Detection(COCO[int(cls_id[keep][i])], float(conf[keep][i]), nb))
    return dets
