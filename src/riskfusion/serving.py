"""Scoring (FR-34, FR-36) and the reviewer session report (FR-37).

One code path turns raw event.v1 events into a risk_assessment.v1: validate -> flat -> features -> model.
The API, the batch CLI and the dashboard all call ``score_events``.
"""

from __future__ import annotations

import html
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
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
_REC_TEXT = {
    "NO_ACTION": ("No review needed", "#2e7d4f"),
    "ROUTINE_REVIEW": ("Routine review", "#5a6878"),
    "HUMAN_REVIEW": ("Human review recommended", "#9a5f0e"),
    "PRIORITY_REVIEW": ("Priority review recommended", "#b42318"),
}


def _timeline_svg(duration_ms: int, flags: list[dict[str, Any]], grid: pd.DataFrame | None) -> str:
    W, left, lane_h = 900, 150, 22
    lanes: list[tuple[str, list[tuple[int, int]], str]] = []
    types = sorted({f["type"] for f in flags})
    for t in types:
        lanes.append(
            (
                t.replace("_", " ").title(),
                [(f["t_start_ms"], f["t_end_ms"]) for f in flags if f["type"] == t],
                "#9a5f0e",
            )
        )
    if grid is not None and len(grid):
        unk = grid[["no_face"]].isna().to_numpy()[:, 0]
        d = np.diff(np.concatenate(([0], unk.astype(int), [0])))
        ivs = [
            (int(s) * 1000, int(e) * 1000) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True)
        ]
        lanes.append(("Video unavailable", ivs, "#b8c2cc"))
    H = 10 + lane_h * max(1, len(lanes)) + 24
    dur = max(duration_ms, 1000)

    def x(ms: float) -> float:
        return left + (min(max(ms, 0), dur) / dur) * (W - left - 10)

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Session timeline">']
    for i, (name, ivs, color) in enumerate(lanes):
        y = 10 + i * lane_h
        parts.append(f'<text x="0" y="{y + 15}" font-size="12" fill="#5a6878">{html.escape(name)}</text>')
        parts.append(
            f'<rect x="{left}" y="{y + 4}" width="{W - left - 10}" height="{lane_h - 8}" rx="3" fill="#eef1f4"/>'
        )
        for s, e in ivs:
            parts.append(
                f'<rect x="{x(s):.1f}" y="{y + 4}" width="{max(2, x(e) - x(s)):.1f}" height="{lane_h - 8}" '
                f'rx="2" fill="{color}"/>'
            )
    step = max(60_000, int(np.ceil(dur / 10 / 60_000)) * 60_000)
    for t in range(0, dur + 1, step):
        parts.append(
            f'<text x="{x(t):.1f}" y="{H - 4}" font-size="11" fill="#8a96a3" text-anchor="middle">{t // 60000} min</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_report(
    ra: dict[str, Any], duration_s: float, grid: pd.DataFrame | None = None, model: RiskModel | None = None
) -> str:
    """Self-contained HTML report for a human reviewer. It recommends; it never states a verdict (ETH-6)."""
    model = model or load_model()
    rec, color = _REC_TEXT[ra["recommendation"]]
    band = ra["confidence_band"]
    flags_rows = (
        "".join(
            f"<tr><td>{html.escape(f['type'].replace('_', ' ').title())}</td>"
            f"<td>{f['t_start_ms'] // 60000:02d}:{f['t_start_ms'] // 1000 % 60:02d} - "
            f"{f['t_end_ms'] // 60000:02d}:{f['t_end_ms'] // 1000 % 60:02d}</td>"
            f"<td>{f['confidence']:.2f}</td><td>{html.escape(f['explanation'])}</td>"
            f"<td>{html.escape(str(f.get('evidence_ref') or '—'))}</td></tr>"
            for f in ra["flags"]
        )
        or '<tr><td colspan="5">No time-bounded observations.</td></tr>'
    )
    contrib = "".join(f"<li>{html.escape(c['explanation'])}</li>" for c in ra["top_contributors"])
    missing = ", ".join(ra["channels_missing"]) or "none"
    tiers = model.meta["tiers"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Review report {html.escape(ra["session_id"])}</title>
<style>
body{{font-family:"Segoe UI",-apple-system,Helvetica,Arial,sans-serif;color:#1f2933;max-width:980px;margin:32px auto;padding:0 20px;line-height:1.5}}
h1{{color:#17324d;font-size:24px;margin:0 0 4px}} h2{{color:#17324d;font-size:17px;margin:28px 0 8px}}
.rec{{display:inline-block;padding:6px 12px;border-radius:6px;color:#fff;background:{color};font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{text-align:left;padding:8px;border-bottom:1px solid #dce1e7;vertical-align:top}}
th{{color:#5a6878;font-weight:600}} .muted{{color:#5a6878;font-size:13px}} .box{{border:1px solid #dce1e7;border-radius:8px;padding:14px 16px;background:#fafbfc}}
</style></head><body>
<p class="muted">RiskFusion review report · generated {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}</p>
<h1>Session {html.escape(ra["session_id"])}</h1>
<p><span class="rec">{rec}</span></p>
<div class="box"><strong>This is a recommendation for a human reviewer, not a finding of misconduct.</strong>
The score estimates how likely it is that this session contains behaviour worth checking. It can be wrong: detectors
make mistakes, especially in poor lighting, with low-quality webcams and in noisy rooms. Please review the evidence
yourself before drawing any conclusion.</div>
<h2>Risk</h2>
<table><tr><th>Calibrated risk</th><td>{ra["overall_risk"]:.2f} (90% band {band[0]:.2f}–{band[1]:.2f})</td></tr>
<tr><th>Session length</th><td>{duration_s / 60:.0f} minutes</td></tr>
<tr><th>Signals unavailable</th><td>{html.escape(missing)} — missing signals never raise the score</td></tr>
<tr><th>Review tiers</th><td class="muted">routine ≥ {tiers["ROUTINE_REVIEW"]:.2f}, human ≥ {tiers["HUMAN_REVIEW"]:.2f}, priority ≥ {tiers["PRIORITY_REVIEW"]:.2f}</td></tr></table>
<h2>Why the score is what it is</h2><ul>{contrib}</ul>
<h2>Timeline</h2>{_timeline_svg(int(duration_s * 1000), ra["flags"], grid)}
<h2>Observations to check</h2>
<table><tr><th>Type</th><th>When</th><th>Confidence</th><th>What was observed</th><th>Evidence</th></tr>{flags_rows}</table>
<p class="muted">Evidence links point to the recording at the given time. Raw recordings are deleted at the end of the
project; after that, links show as unavailable (SRS ETH-3).</p>
<h2>Model</h2><p class="muted">{html.escape(ra["model_version"])}, {html.escape(ra["feature_version"])}.
See the model card for performance, known limitations and fairness results.</p>
</body></html>"""


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
            (out_dir / "reports" / f"{sid}.html").write_text(render_report(ra, len(sf.grid), sf.grid, model))
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
