"""Build the versioned feature store for a dataset (FR-15) — streaming shard by shard (NFR-3).

Outputs in data/features/<feature_version>/<dataset_version>/:
  features.parquet   one row per session (session features, FR-11..14)
  temporal.parquet   ROCKET features over the per-10-second grid (input of the temporal model, FR-18)
  flags.parquet      time-bounded flags per session (FR-25)
  schema.json        column list + schema hash + timing
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .engine import FEATURE_VERSION, SIGNALS, WINDOWS, compute_session

BIN_S = 10
MAX_BINS = 540  # 90 minutes
N_KERNELS = 400
ROCKET_SEED = 20260923


class Rocket:
    """ROCKET (Dempster et al., 2020): random dilated convolutional kernels, PPV and max pooling."""

    def __init__(self, n_channels: int, n_kernels: int = N_KERNELS, length: int = MAX_BINS, seed: int = ROCKET_SEED):
        rng = np.random.default_rng(seed)
        self.kernels = []
        for _ in range(n_kernels):
            k = 9
            w = rng.normal(0, 1, k)
            w -= w.mean()
            max_exp = np.log2((length - 1) / (k - 1))
            d = int(2 ** rng.uniform(0, min(max_exp, 5)))
            chans = rng.choice(n_channels, size=int(rng.integers(1, 4)), replace=False)
            self.kernels.append((w, d, chans, float(rng.uniform(-1, 1))))

    def transform(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """X: [n, C, L] (zero-padded), lengths: valid bins per series -> [n, 2K]."""
        n, _, L = X.shape
        out = np.zeros((n, 2 * len(self.kernels)), dtype=np.float32)
        t = np.arange(L)
        for i, (w, d, chans, b) in enumerate(self.kernels):
            span = 8 * d
            if span >= L:
                continue
            xs = X[:, chans, :].sum(1)
            conv = np.full((n, L - span), b)
            for j in range(9):
                conv += w[j] * xs[:, j * d : j * d + L - span]
            valid = t[: L - span][None, :] < (lengths[:, None] - span)
            has = valid.any(1)
            cm = np.where(valid, conv, -np.inf)
            out[:, 2 * i] = np.where(has, cm.max(1), 0.0)
            out[:, 2 * i + 1] = np.where(has, ((conv > 0) & valid).sum(1) / np.maximum(valid.sum(1), 1), 0.0)
        return out


def bin_grid(grid: pd.DataFrame) -> np.ndarray:
    """[C, MAX_BINS] 10-second means of the signals, NaN -> 0, continuous signals squashed to ~[0, 1]."""
    X = grid[list(SIGNALS)].to_numpy(dtype=float)
    scale = np.array([1.0 if SIGNALS[s][1] == "binary" else SIGNALS[s][2] * 2 for s in SIGNALS])
    X = np.clip(np.nan_to_num(X) / scale, 0, 3)
    nb = min(MAX_BINS, int(np.ceil(len(X) / BIN_S)))
    pad = np.zeros((nb * BIN_S, X.shape[1]))
    pad[: min(len(X), nb * BIN_S)] = X[: nb * BIN_S]
    binned = pad.reshape(nb, BIN_S, -1).mean(1).T
    out = np.zeros((X.shape[1], MAX_BINS), dtype=np.float32)
    out[:, :nb] = binned
    return out


def schema_hash(columns: list[str]) -> str:
    return hashlib.sha256((FEATURE_VERSION + "|" + "|".join(columns)).encode()).hexdigest()


def build_feature_store(
    data_root: Path,
    dataset_version: str,
    *,
    limit: int | None = None,
    windows: tuple[int, ...] = WINDOWS,
    baseline: bool = True,
    out_name: str | None = None,
) -> dict[str, Any]:
    src = data_root / "synthetic" / dataset_version
    meta = pd.read_parquet(src / "sessions.parquet").set_index("session_id")
    out = data_root / "features" / FEATURE_VERSION / (out_name or dataset_version)
    out.mkdir(parents=True, exist_ok=True)
    rocket = Rocket(len(SIGNALS))
    rows: list[dict[str, Any]] = []
    trows: list[np.ndarray] = []
    tids: list[str] = []
    flags: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    n_done = 0
    for shard in sorted((src / "events").glob("shard-*.parquet")):
        df = pq.read_table(shard).to_pandas()
        codes = df["session_id"].cat.codes.to_numpy()
        cats = df["session_id"].cat.categories
        order = np.argsort(codes, kind="stable")
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        batch_grids, batch_len = [], []
        for idx in np.split(order, bounds):
            sid = str(cats[codes[idx[0]]])
            ev = df.iloc[idx]
            dur = float(meta.at[sid, "duration_s"])
            sf = compute_session(sid, ev, dur, windows=windows, baseline=baseline)
            rows.append({"session_id": sid, **sf.features})
            flags.extend({"session_id": sid, **fl} for fl in sf.flags)
            batch_grids.append(bin_grid(sf.grid))
            batch_len.append(min(MAX_BINS, int(np.ceil(dur / BIN_S))))
            tids.append(sid)
            n_done += 1
            if limit and n_done >= limit:
                break
        trows.append(rocket.transform(np.stack(batch_grids), np.array(batch_len)))
        del df
        if limit and n_done >= limit:
            break
    feats = pd.DataFrame(rows)
    cols = [c for c in feats.columns if c != "session_id"]
    feats.to_parquet(out / "features.parquet", index=False)
    temporal = pd.DataFrame(np.concatenate(trows), columns=[f"rk{i:04d}" for i in range(2 * N_KERNELS)])
    temporal.insert(0, "session_id", tids)
    temporal.to_parquet(out / "temporal.parquet", index=False)
    pd.DataFrame(flags).to_parquet(out / "flags.parquet", index=False)
    elapsed = time.perf_counter() - t0
    schema = {
        "feature_version": FEATURE_VERSION,
        "dataset_version": dataset_version,
        "n_sessions": len(feats),
        "n_features": len(cols),
        "columns": cols,
        "schema_hash": schema_hash(cols),
        "windows": list(windows),
        "baseline_normalisation": baseline,
        "temporal": {"method": "ROCKET", "kernels": N_KERNELS, "bin_s": BIN_S, "seed": ROCKET_SEED},
        "seconds": round(elapsed, 1),
        "seconds_per_session": round(elapsed / max(1, len(feats)), 4),
    }
    (out / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    return schema
