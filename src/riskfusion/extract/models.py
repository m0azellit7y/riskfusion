"""Pretrained model registry: download once, verify SHA-256, load with ONNX Runtime.

Licences (see models/pretrained/README.md):
* YOLOX-nano (Megvii) — Apache-2.0.
* InsightFace buffalo_s (SCRFD det_500m, ArcFace w600k_mbf) — code MIT; the pretrained weights are released
  for non-commercial research use only. Fine for this research prototype; replace before any commercial use.
The buffalo_s pack also contains a gender/age model. It is deliberately never extracted or loaded (ETH-4).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL_DIR = Path(os.environ.get("RISKFUSION_MODEL_DIR", Path(__file__).resolve().parents[3] / "models" / "pretrained"))

MODELS = {
    "yolox_nano.onnx": {
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
        "sha256_prefix": "c789161ed43c8269",
    },
    "det_500m.onnx": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        "member": "det_500m.onnx",
        "sha256_prefix": "5e4447f50245bbd7",
    },
    "w600k_mbf.onnx": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        "member": "w600k_mbf.onnx",
        "sha256_prefix": "9cc6e4a75f0e2bf0",
    },
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_models(model_dir: Path = MODEL_DIR) -> dict[str, str]:
    """Download (if needed) and verify every model. Returns name -> sha256."""
    model_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    zips: dict[str, Path] = {}
    for name, spec in MODELS.items():
        dest = model_dir / name
        if not dest.exists():
            if "member" in spec:
                url = spec["url"]
                if url not in zips:
                    zp = model_dir / Path(url).name
                    if not zp.exists():
                        urllib.request.urlretrieve(url, zp)  # noqa: S310 - fixed, pinned URL
                    zips[url] = zp
                with zipfile.ZipFile(zips[url]) as z:
                    dest.write_bytes(z.read(spec["member"]))  # only the named member is extracted
            else:
                urllib.request.urlretrieve(spec["url"], dest)  # noqa: S310
        digest = _sha(dest)
        if not digest.startswith(spec["sha256_prefix"]):
            raise RuntimeError(f"{name}: checksum mismatch ({digest[:16]}), delete it and fetch again")
        out[name] = digest
    for zp in zips.values():
        zp.unlink(missing_ok=True)
    return out


@lru_cache
def session(name: str) -> Any:
    import onnxruntime as ort

    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"model {name} missing — run `make models`")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 1))
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
