"""Build a 60 s WebM with known ground truth from the scikit-image astronaut photo (a real face).

0-20 s one face | 20-30 s nobody | 30-40 s two faces | 40-50 s very dark | 50-60 s one face
Audio: silence, with a 440 Hz "voice" 5-15 s and a different-timbre signal 32-38 s.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import data


def build(out: Path, fps: int = 5) -> Path:
    face = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)[0:360, 60:420]
    w, h = 640, 360
    proc = subprocess.Popen(
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
            "-f",
            "lavfi",
            "-i",
            "aevalsrc='if(between(t,5,15),0.3*sin(2*PI*220*t)*(1+0.5*sin(2*PI*3*t)),0)"
            "+if(between(t,32,38),0.3*sin(2*PI*1400*t)*sin(2*PI*7*t),0)+0.002*random(0)':s=16000:d=60",
            "-t",
            "60",
            "-c:v",
            "libvpx",
            "-b:v",
            "600k",
            "-c:a",
            "libopus",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin
    for i in range(60 * fps):
        t = i / fps
        frame = np.full((h, w, 3), 110, np.uint8)
        if t < 20 or t >= 50:
            frame[0:360, 140:500] = face
        elif 30 <= t < 40:
            frame[0:360, 0:320] = cv2.resize(face, (320, 360))
            frame[0:360, 320:640] = cv2.flip(cv2.resize(face, (320, 360)), 1)
        elif 40 <= t < 50:
            frame[0:360, 140:500] = face
            frame = (frame * 0.08).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/face_test.webm")))
