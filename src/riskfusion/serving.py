"""Scoring (FR-34, FR-36) and the reviewer session report (FR-37).

One code path turns raw event.v1 events into a risk_assessment.v1: validate -> flat -> features -> model.
The API, the batch CLI and the dashboard all call ``score_events``.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from riskfusion.contracts import CHANNELS, FLAT_COLUMNS, event_errors, event_to_flat
from riskfusion.features.engine import compute_session
from riskfusion.modeling.model import RiskModel, channels_from_features

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "trained" / "fusion-v1.0.0"


@lru_cache
def load_model(path: str = str(DEFAULT_MODEL)) -> RiskModel:
    return RiskModel.load(Path(path))


def events_to_frame(events: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows, rejected = [], []
    for i, ev in enumerate(events):
        ev = {k: v for k, v in ev.items() if k != "event_uid"}
        errs = event_errors(ev)
        if errs:
            rejected.append({"index": i, "reason": "; ".join(errs)})
        else:
            rows.append(event_to_flat(ev))
    df = pd.DataFrame(rows, columns=list(FLAT_COLUMNS))
    return df, rejected


def score_frame(
    session_id: str, df: pd.DataFrame, duration_s: float | None, model: RiskModel | None = None, reorder: bool = True
) -> tuple[dict[str, Any], Any]:
    model = model or load_model()
    if duration_s is None:
        duration_s = float(df["ts_ms"].max() / 1000 + 1) if len(df) else 1.0
    sf = compute_session(session_id, df, duration_s, reorder=reorder)
    available, missing = channels_from_features(sf.features)
    ra = model.assess(session_id, sf.features, sf.flags, available, missing)
    return ra, sf


def score_events(
    session_id: str, events: list[dict[str, Any]], duration_s: float | None = None, model: RiskModel | None = None
) -> dict[str, Any]:
    t0 = time.perf_counter()
    df, rejected = events_to_frame(events)
    ra, sf = score_frame(session_id, df, duration_s, model)
    ra["_meta"] = {
        "events_accepted": len(df),
        "events_rejected": len(rejected),
        "late_events": sf.late_events,
        "rejections": rejected[:50],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return ra


# ------------------------------------------------------------------------------------ FR-37 report
def render_report(
    ra: dict[str, Any],
    duration_s: float,
    grid: pd.DataFrame | None = None,
    model: RiskModel | None = None,
    events: pd.DataFrame | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Self-contained HTML report for a human reviewer (see riskfusion.reporting)."""
    from riskfusion.reporting import render_report as _render

    return _render(ra, duration_s, grid, model or load_model(), events, meta)


# ------------------------------------------------------------------------------------ FR-36 batch CLI
def score_directory(in_dir: Path, out_dir: Path, model_path: Path = DEFAULT_MODEL) -> dict[str, Any]:
    """Score every session in a directory. Format: one file per session, named <session_id>.jsonl (one event.v1
    per line) or <session_id>.json (a JSON list). Optional <session_id>.meta.json with {"duration_s": ...}."""
    model = load_model(str(model_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)
    results, failures = [], []
    for f in sorted(
        list(in_dir.glob("*.jsonl")) + [p for p in in_dir.glob("*.json") if not p.name.endswith(".meta.json")]
    ):
        sid = f.stem
        try:
            if f.suffix == ".jsonl":
                events = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            else:
                events = json.loads(f.read_text())
            meta_p = f.with_name(f"{sid}.meta.json")
            dur = json.loads(meta_p.read_text()).get("duration_s") if meta_p.exists() else None
            df, rejected = events_to_frame(events)
            ra, sf = score_frame(sid, df, dur, model)
            ra["_meta"] = {"events_accepted": len(df), "events_rejected": len(rejected)}
            (out_dir / "reports" / f"{sid}.html").write_text(
                render_report(ra, len(sf.grid), sf.grid, model, df, {"source": "batch", "script_title": f.name}),
                encoding="utf-8",
            )
            results.append(ra)
        except Exception as e:  # one bad session must not stop the batch
            failures.append({"session_id": sid, "error": f"{type(e).__name__}: {e}"})
    with (out_dir / "assessments.jsonl").open("w") as fh:
        for ra in results:
            fh.write(json.dumps(ra) + "\n")
    summary = {
        "scored": len(results),
        "failed": failures,
        "recommendations": pd.Series([r["recommendation"] for r in results]).value_counts().to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def export_session_events(parquet_dir: Path, session_id: str, out: Path) -> None:
    """Write one simulated session as <session_id>.jsonl — an example input for the batch CLI."""
    import pyarrow.dataset as ds

    from riskfusion.contracts import flat_to_event

    tbl = ds.dataset(parquet_dir, format="parquet").to_table(filter=ds.field("session_id") == session_id)
    df = tbl.to_pandas()
    with out.open("w") as fh:
        for row in df.to_dict(orient="records"):
            fh.write(json.dumps(flat_to_event(row)) + "\n")


__all__ = ["CHANNELS", "load_model", "render_report", "score_directory", "score_events", "score_frame"]
