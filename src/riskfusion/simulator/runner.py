"""Generate a simulated dataset version to Parquet (FR-1, NFR-3 streaming by shard)."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from riskfusion.contracts import validate_event_frame, write_dead_letter

from .config import SimulatorConfig
from .engine import sessions_to_event_frame, simulate_session


@dataclass
class RunResult:
    run_id: str
    out_dir: Path
    manifest: dict[str, Any]


def run_id_for(cfg: SimulatorConfig) -> str:
    return f"{cfg.config_version}-s{cfg.seed}-n{cfg.n_sessions}"


def _frame_digest(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df.astype({c: "object" for c in df.select_dtypes("category").columns}), index=False)
    return hashlib.sha256(h.to_numpy().tobytes()).hexdigest()


def generate(cfg: SimulatorConfig, data_root: Path, overwrite: bool = False) -> RunResult:
    run_id = run_id_for(cfg)
    out = data_root / "synthetic" / run_id
    if out.exists() and (out / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"dataset version {run_id} already exists at {out}")
    (out / "events").mkdir(parents=True, exist_ok=True)
    for old in (out / "events").glob("*.parquet"):
        old.unlink()
    dead_path = data_root / "deadletter" / f"{run_id}.jsonl"
    if dead_path.exists():
        dead_path.unlink()
    run_tag = f"{cfg.config_version.replace('-', '')}s{cfg.seed % 10000}"

    t0 = time.perf_counter()
    metas: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    shard_digests: list[str] = []
    n_events = 0
    n_dead = 0
    for shard_no, start in enumerate(range(0, cfg.n_sessions, cfg.shard_size)):
        stop = min(cfg.n_sessions, start + cfg.shard_size)
        sessions = [simulate_session(cfg, i, run_tag) for i in range(start, stop)]
        frame = sessions_to_event_frame(sessions, detector_version=cfg.config_version)
        valid, rejected = validate_event_frame(frame)
        n_dead += write_dead_letter(rejected, dead_path)
        table = pa.Table.from_pandas(valid, preserve_index=False)
        pq.write_table(table, out / "events" / f"shard-{shard_no:04d}.parquet", compression="zstd")
        shard_digests.append(_frame_digest(valid))
        n_events += len(valid)
        metas.extend(s.meta for s in sessions)
        labels.extend(s.label for s in sessions)
        del sessions, frame, valid, table
    elapsed = time.perf_counter() - t0

    meta_df = pd.DataFrame(metas)
    meta_df["dataset_version"] = run_id
    meta_df.to_parquet(out / "sessions.parquet", index=False)
    lab_df = pd.DataFrame(labels)
    lab_df["intervals"] = lab_df["intervals"].map(json.dumps)
    lab_df.to_parquet(out / "labels.parquet", index=False)

    content_hash = hashlib.sha256(
        ("".join(shard_digests) + _frame_digest(meta_df.drop(columns=["violation_types", "channels_outage"]))).encode()
    ).hexdigest()
    manifest = {
        "dataset_version": run_id,
        "kind": "simulated",
        "config_version": cfg.config_version,
        "config_hash": cfg.config_hash(),
        "seed": cfg.seed,
        "n_sessions": cfg.n_sessions,
        "n_events": n_events,
        "dead_letter_events": n_dead,
        "positive_rate": float(meta_df["violation"].mean()),
        "expected_positive_rate": cfg.expected_positive_rate(),
        "profile_counts": {k: int(v) for k, v in meta_df["behavior_profile"].value_counts().items()},
        "noise_source": cfg.noise_source,
        "content_hash": content_hash,
        "generation_seconds": round(elapsed, 2),
        "platform": f"{platform.system()} {platform.machine()} py{platform.python_version()}",
        "numpy": np.__version__,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    build_session_index(out)
    return RunResult(run_id=run_id, out_dir=out, manifest=manifest)


def build_session_index(run_dir: Path) -> Path:
    """Write ``session_index.parquet`` (session_id -> shard file) so one session is read from one shard."""
    rows: list[tuple[str, str]] = []
    for shard in sorted((run_dir / "events").glob("shard-*.parquet")):
        sids = pq.read_table(shard, columns=["session_id"]).column("session_id").unique().to_pylist()
        rows.extend((str(sid), shard.name) for sid in sids)
    path = run_dir / "session_index.parquet"
    pd.DataFrame(rows, columns=["session_id", "shard"]).to_parquet(path, index=False)
    return path
