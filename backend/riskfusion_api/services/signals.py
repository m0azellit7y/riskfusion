"""Signal timeline for one simulated session, read from the Parquet event store.

Reduces raw event.v1 rows to 10-second bins per signal, then merges consecutive bins into intervals
for display. This is a view of *detector output* (noisy), not of ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

BIN_MS = 10_000


def _intervals(bins: np.ndarray) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    idx = np.flatnonzero(bins)
    if idx.size == 0:
        return out
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i != prev + 1:
            out.append({"start_ms": start * BIN_MS, "end_ms": (prev + 1) * BIN_MS})
            start = i
        prev = i
    out.append({"start_ms": start * BIN_MS, "end_ms": (prev + 1) * BIN_MS})
    return out


def signal_timeline(events_dir: Path, session_id: str, duration_s: float) -> dict[str, Any]:
    index = events_dir.parent / "session_index.parquet"
    if not index.exists():
        from riskfusion.simulator.runner import build_session_index

        build_session_index(events_dir.parent)
    idx = pd.read_parquet(index)
    shards = idx.loc[idx["session_id"] == session_id, "shard"].tolist()
    if not shards:
        raise KeyError(session_id)
    dataset = ds.dataset([str(events_dir / f) for f in shards], format="parquet")
    table = dataset.to_table(
        columns=["ts_ms", "event_type", "p0", "p1", "p3", "label"],
        filter=pc.field("session_id") == session_id,
    )
    return summarise_signals(table.to_pandas(), duration_s)


def summarise_signals(df: pd.DataFrame, duration_s: float) -> dict[str, Any]:
    """10-second signal lanes from flat event.v1 rows — simulated (Parquet) or real (detector events in the DB)."""
    n = int(np.ceil(duration_s * 1000 / BIN_MS)) + 1
    b = (df["ts_ms"].to_numpy() // BIN_MS).astype(int).clip(0, n - 1)
    et = df["event_type"].astype(str).to_numpy()
    lab = df["label"].astype(str).to_numpy()
    p0, p1, p3 = (df[c].to_numpy(dtype=float) for c in ("p0", "p1", "p3"))

    def any_in_bins(mask: np.ndarray, min_count: int = 1) -> np.ndarray:
        counts = np.bincount(b[mask], minlength=n)
        return counts >= min_count

    tab = np.zeros(n, dtype=bool)
    tv = np.flatnonzero(et == "TAB_VISIBILITY")
    hidden_from = None
    for i in tv[np.argsort(df["ts_ms"].to_numpy()[tv])]:
        if p0[i] == 1 and hidden_from is None:
            hidden_from = b[i]
        elif p0[i] == 0 and hidden_from is not None:
            tab[hidden_from : b[i] + 1] = True
            hidden_from = None
    if hidden_from is not None:
        tab[hidden_from:] = True
    signals = {
        "Second face": any_in_bins((et == "FACE_OBSERVATION") & (p0 >= 2), 2),
        "No face": any_in_bins((et == "FACE_OBSERVATION") & (p0 == 0), 2),
        "Gaze off screen": any_in_bins((et == "HEAD_GAZE") & (p3 < 0.5), 4),
        "Phone detected": any_in_bins((et == "OBJECT_DETECTED") & (lab == "cell_phone")),
        "Paper detected": any_in_bins((et == "OBJECT_DETECTED") & (lab == "paper")),
        "Identity mismatch": any_in_bins((et == "IDENTITY_CHECK") & (p1 == 0)),
        "Other voice": any_in_bins((et == "VOICE_ACTIVITY") & (p3 == 1), 2),
        "Tab hidden": tab,
        "Channel unknown": any_in_bins(np.char.endswith(et.astype(str), "_UNKNOWN"), 3),
    }
    return {
        "bin_ms": BIN_MS,
        "n_events": int(len(df)),
        "event_types": {k: int(v) for k, v in df["event_type"].astype(str).value_counts().items()},
        "lanes": [{"label": k, "intervals": _intervals(v)} for k, v in signals.items()],
    }
