"""Synthetic media with known ground truth for the liveness, pose and audio-event detectors."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, lfilter
from skimage import data


def face_canvas() -> np.ndarray:
    face = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)[0:360, 60:420].copy()
    canvas = np.full((360, 640, 3), 110, np.uint8)
    canvas[:, 140:500] = face
    return canvas


def liveness_video(path: Path, blink: bool, seconds: int = 95, fps: int = 15) -> Path:
    """A still photo (never blinks) or the same face moving slightly and 'blinking' every 4 s."""
    from riskfusion.extract import vision as V

    canvas = face_canvas()
    eyes = V.detect_faces(canvas)[0].kps[:2]
    w, h = 640, 360
    p = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libvpx",
            "-b:v",
            "800k",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert p.stdin
    for i in range(seconds * fps):
        t = i / fps
        fr = canvas
        if blink:
            dx, dy = 2 * np.sin(t * 1.3), 1.5 * np.sin(t * 0.9)
            fr = cv2.warpAffine(canvas, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderValue=(110, 110, 110))
            if (t % 4.0) < 0.2:
                for ex, ey in eyes:
                    cv2.ellipse(fr, (int(ex + dx), int(ey + dy)), (22, 12), 0, 0, 360, (120, 150, 190), -1)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    return path


def audio_events_file(path: Path) -> Path:
    """60 s: ring 10-13 s, keyboard 20-23 s, paper 30-33 s, door knock at 40 s, voice-like 48-52 s."""
    sr, T = 16000, 60
    rng = np.random.default_rng(1)
    x = 0.003 * rng.normal(size=T * sr)
    t = np.arange(sr * 3) / sr
    x[10 * sr : 13 * sr] += 0.25 * np.sin(2 * np.pi * 1500 * t) * (np.sin(2 * np.pi * 2 * t) > -0.6)
    b, a = butter(2, 3000 / (sr / 2), "high")
    for k in range(15):
        i, n = int((20 + k / 5 + 0.03) * sr), int(0.012 * sr)
        x[i : i + n] += 0.4 * lfilter(b, a, rng.normal(size=n)) * np.exp(-np.arange(n) / 60)
    b, a = butter(4, 2500 / (sr / 2), "high")
    x[30 * sr : 33 * sr] += (
        0.08 * lfilter(b, a, rng.normal(size=3 * sr)) * (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3 * t)))
    )
    n = int(0.12 * sr)
    tt = np.arange(n) / sr
    x[40 * sr : 40 * sr + n] += 0.9 * np.sin(2 * np.pi * 70 * tt) * np.exp(-tt * 25)
    tv = np.arange(sr * 4) / sr
    x[48 * sr : 52 * sr] += (
        0.2
        * np.sin(2 * np.pi * 180 * tv * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * tv)))
        * (1 + np.sin(2 * np.pi * 4 * tv))
        / 2
    )
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-i",
            "-",
            "-c:a",
            "libopus",
            str(path),
        ],
        input=pcm.tobytes(),
        check=True,
    )
    return path
