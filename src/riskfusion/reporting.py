"""Reviewer session report (FR-37): a complete, printable account of what happened in one session.

``session_digest`` turns the per-second grid and the raw events into durations, counts and a time-ordered event
log; ``render_report`` lays it out as one self-contained HTML page (no external assets; prints to A4).
The report recommends; it never states a verdict (SRS CON-5, ETH-6).
"""

from __future__ import annotations

import html
import time
from typing import Any

import numpy as np
import pandas as pd

from riskfusion.contracts import CHANNELS

# ------------------------------------------------------------------------------------ formatting
REC = {
    "NO_ACTION": ("No review needed", "#2e7d4f", "#e6f2eb"),
    "ROUTINE_REVIEW": ("Routine review", "#2c5f8a", "#e5eef6"),
    "HUMAN_REVIEW": ("Human review recommended", "#9a5f0e", "#fbf0dc"),
    "PRIORITY_REVIEW": ("Priority review recommended", "#b42318", "#fbe8e6"),
}
CHANNEL_NAMES = {
    "presence": "Face presence",
    "identity": "Identity match",
    "liveness": "Liveness",
    "attention": "Head pose and gaze",
    "environment": "People and objects",
    "pose": "Body pose and hands",
    "audio_voice": "Speech and voices",
    "audio_event": "Sound events",
    "screen": "Exam tab and screen",
    "device": "Displays",
    "behavioral": "Typing",
}


def clock(ms: float) -> str:
    s = max(0, int(round(ms / 1000)))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def dur(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def times(n: int) -> str:
    return f"{n} time" if n == 1 else f"{n} times"


def e(t: Any) -> str:
    return html.escape(str(t), quote=True)


# ------------------------------------------------------------------------------------ digest
def _runs(active: np.ndarray, merge_gap: int = 2) -> list[tuple[int, int]]:
    """Consecutive active seconds as (start_s, end_s), merging gaps of <= merge_gap seconds."""
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []
    runs: list[list[int]] = [[int(idx[0]), int(idx[0]) + 1]]
    for i in idx[1:]:
        i = int(i)
        if i - runs[-1][1] <= merge_gap:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    return [(a, b) for a, b in runs]


# signal -> (label in the report, channel, how to read it, minimum episode length to list in the event log)
ACTIVITY = [
    ("phone", "Phone visible", "environment", 1, "{d} with a phone in view"),
    ("multi_face", "More than one face", "presence", 2, "{d} with a second face in view"),
    ("extra_person", "Another person detected", "environment", 2, "{d} with another person in the room"),
    ("no_face", "No face visible", "presence", 3, "{d} with no face in view"),
    (
        "id_mismatch",
        "Face did not match enrolment",
        "identity",
        5,
        "{d} where the face did not match the enrolment photo",
    ),
    ("gaze_off", "Looking away from the screen", "attention", 4, "{d} looking away from the screen"),
    ("paper", "Paper or book visible", "environment", 2, "{d} with paper or a book in view"),
    ("reach", "Reaching out of frame", "pose", 3, "{d} reaching out of the camera view"),
    ("foreign", "Another voice", "audio_voice", 2, "{d} with a voice other than the candidate's"),
    ("vad", "Speech in the room", "audio_voice", 99999, "{d} of speech"),
    ("pad_fail", "Liveness check failed", "liveness", 1, "liveness failed for {d}"),
    ("tab_hidden", "Exam tab hidden", "screen", 1, "{d} away from the exam tab"),
    ("fs_off", "Outside full screen", "screen", 1, "{d} outside full-screen mode"),
]


def session_digest(grid: pd.DataFrame, events: pd.DataFrame | None, duration_s: float) -> dict[str, Any]:
    T = len(grid)
    rows = []
    for key, label, channel, _min_len, _txt in ACTIVITY:
        if key not in grid:
            continue
        x = grid[key].to_numpy(dtype=float)
        observed = np.isfinite(x)
        active = np.nan_to_num(x) > 0.5
        runs = _runs(active)
        rows.append(
            {
                "key": key,
                "label": label,
                "channel": channel,
                "seconds": int(active.sum()),
                "share": float(active.sum() / max(1, observed.sum())) if observed.any() else None,
                "episodes": len(runs),
                "longest": max((b - a for a, b in runs), default=0),
                "first": runs[0][0] if runs else None,
                "observed_s": int(observed.sum()),
            }
        )
    ev = events if events is not None else pd.DataFrame(columns=["ts_ms", "event_type", "label", "p0", "p1"])
    et = ev["event_type"].astype(str).to_numpy() if len(ev) else np.array([])
    lab = ev["label"].astype(object).to_numpy() if len(ev) else np.array([])
    p0 = pd.to_numeric(ev.get("p0", pd.Series(dtype=float)), errors="coerce").to_numpy()
    p1 = pd.to_numeric(ev.get("p1", pd.Series(dtype=float)), errors="coerce").to_numpy()
    ts = pd.to_numeric(ev.get("ts_ms", pd.Series(dtype=float)), errors="coerce").to_numpy()

    def count(t: str) -> int:
        return int((et == t).sum())

    tab_leaves = int(((et == "TAB_VISIBILITY") & (p0 == 1)).sum())
    fs_exits = int(((et == "FULLSCREEN_CHANGE") & (p0 == 0)).sum())
    paste_m = et == "PASTE"
    audio = (
        {str(k): int(v) for k, v in pd.Series(lab[et == "AUDIO_EVENT"]).value_counts().items()}
        if count("AUDIO_EVENT")
        else {}
    )
    objects = (
        {str(k): int(v) for k, v in pd.Series(lab[et == "OBJECT_DETECTED"]).value_counts().items()}
        if count("OBJECT_DETECTED")
        else {}
    )
    live_m = et == "LIVENESS_CHECK"
    id_m = et == "IDENTITY_CHECK"
    mon = p0[et == "MONITOR_COUNT"]
    counters = {
        "tab_leaves": tab_leaves,
        "fullscreen_exits": fs_exits,
        "pastes": int(paste_m.sum()),
        "pasted_chars": int(np.nansum(p0[paste_m])) if paste_m.any() else 0,
        "keystrokes": int(np.nansum(p0[et == "INPUT_ACTIVITY"])) if count("INPUT_ACTIVITY") else None,
        "max_monitors": int(np.nanmax(mon)) if mon.size else None,
        "liveness_checks": int(live_m.sum()),
        "liveness_failed": int(((p1 == 0) & live_m).sum()),
        "identity_checks": int(id_m.sum()),
        "identity_min_similarity": float(np.nanmin(p0[id_m])) if id_m.any() else None,
        "audio_events": audio,
        "objects": objects,
        "events_total": int(len(ev)),
    }
    return {
        "duration_s": duration_s,
        "seconds_analysed": T,
        "activity": rows,
        "counters": counters,
        "log": _event_log(grid, ev, et, lab, p0, ts),
    }


def _event_log(
    grid: pd.DataFrame, ev: pd.DataFrame, et: np.ndarray, lab: np.ndarray, p0: np.ndarray, ts: np.ndarray
) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    # browser events are exact to the millisecond
    order = np.argsort(ts, kind="stable") if ts.size else np.array([], dtype=int)
    hidden_at = None
    fs_off_at = None
    for i in order:
        t, typ = float(ts[i]), et[i]
        if typ == "TAB_VISIBILITY":
            if p0[i] == 1 and hidden_at is None:
                hidden_at = t
            elif p0[i] == 0 and hidden_at is not None:
                log.append(
                    {
                        "t": hidden_at,
                        "end": t,
                        "kind": "screen",
                        "text": f"Left the exam tab; returned after {dur((t - hidden_at) / 1000)}",
                    }
                )
                hidden_at = None
        elif typ == "FULLSCREEN_CHANGE":
            if p0[i] == 0 and fs_off_at is None:
                fs_off_at = t
            elif p0[i] == 1 and fs_off_at is not None:
                log.append(
                    {
                        "t": fs_off_at,
                        "end": t,
                        "kind": "screen",
                        "text": f"Left full-screen mode for {dur((t - fs_off_at) / 1000)}",
                    }
                )
                fs_off_at = None
        elif typ == "PASTE":
            log.append(
                {"t": t, "end": None, "kind": "screen", "text": f"Pasted {int(p0[i])} characters into an answer"}
            )
        elif typ == "MONITOR_COUNT" and p0[i] >= 2:
            log.append({"t": t, "end": None, "kind": "device", "text": f"{int(p0[i])} displays connected"})
        elif typ == "LIVENESS_CHECK" and pd.notna(ev.iloc[i].get("p1")) and float(ev.iloc[i]["p1"]) == 0:
            log.append(
                {"t": t, "end": None, "kind": "liveness", "text": "Liveness check failed (no blink in the last minute)"}
            )
    if hidden_at is not None:
        log.append(
            {
                "t": hidden_at,
                "end": None,
                "kind": "screen",
                "text": "Left the exam tab and did not return before the end",
            }
        )
    # sound events: merge consecutive seconds of the same label
    if (et == "AUDIO_EVENT").any():
        a = pd.DataFrame({"t": ts[et == "AUDIO_EVENT"], "label": lab[et == "AUDIO_EVENT"]}).sort_values("t")
        names = {
            "phone_ring": "Phone ringing",
            "keyboard_burst": "Keyboard typing",
            "paper_rustle": "Paper rustling",
            "door": "Door or knock",
            "other": "Other sound",
        }
        cur = None
        a = a[a["label"] != "keyboard_burst"]  # typing is normal exam behaviour: counted in the summary, not logged
        for t, lbl in zip(a["t"], a["label"], strict=True):
            if cur and cur["label"] == lbl and t - cur["end"] <= 2500:
                cur["end"] = t + 1000
            else:
                if cur:
                    log.append(
                        {
                            "t": cur["t"],
                            "end": cur["end"],
                            "kind": "sound",
                            "text": f"{names.get(cur['label'], cur['label'])} heard for {dur((cur['end'] - cur['t']) / 1000)}",
                        }
                    )
                cur = {"t": t, "end": t + 1000, "label": lbl}
        if cur:
            log.append(
                {
                    "t": cur["t"],
                    "end": cur["end"],
                    "kind": "sound",
                    "text": f"{names.get(cur['label'], cur['label'])} heard for {dur((cur['end'] - cur['t']) / 1000)}",
                }
            )
    # camera and audio signals: episodes from the per-second grid
    for key, _label, channel, min_len, txt in ACTIVITY:
        if key not in grid or channel == "screen" or key == "pad_fail":
            continue
        for a0, b0 in _runs(np.nan_to_num(grid[key].to_numpy(dtype=float)) > 0.5):
            if b0 - a0 >= min_len:
                log.append(
                    {"t": a0 * 1000, "end": b0 * 1000, "kind": channel, "text": txt.format(d=dur(b0 - a0)).capitalize()}
                )
    log.sort(key=lambda r: r["t"])
    return log


# ------------------------------------------------------------------------------------ timeline
def _timeline(duration_ms: int, lanes: list[tuple[str, list[tuple[float, float]], str]]) -> str:
    W, left, lane_h = 1000, 190, 22
    H = 8 + lane_h * max(1, len(lanes)) + 26
    dur_ms = max(duration_ms, 1000)

    def x(ms: float) -> float:
        return left + (min(max(ms, 0), dur_ms) / dur_ms) * (W - left - 8)

    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Session timeline" class="tl">']
    step_min = next((m for m in (1, 2, 5, 10, 15, 30, 60) if dur_ms / (m * 60000) <= 10), 60)
    step = step_min * 60000 if dur_ms > 120000 else (10000 if dur_ms > 40000 else 5000)
    for t in range(0, dur_ms + 1, step):
        p.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="6" y2="{H - 22}" stroke="#e4e7eb"/>'
            f'<text x="{x(t):.1f}" y="{H - 6}" font-size="11" fill="#6b7785" text-anchor="middle">{clock(t)}</text>'
        )
    for i, (name, ivs, color) in enumerate(lanes):
        y = 8 + i * lane_h
        p.append(
            f'<text x="0" y="{y + 15}" font-size="12" fill="#3d4a58">{e(name)}</text>'
            f'<rect x="{left}" y="{y + 4}" width="{W - left - 8}" height="{lane_h - 8}" rx="3" fill="#f0f2f4"/>'
        )
        for a0, b0 in ivs:
            p.append(
                f'<rect x="{x(a0):.1f}" y="{y + 4}" width="{max(2.0, x(b0) - x(a0)):.1f}" height="{lane_h - 8}" '
                f'rx="2" fill="{color}"><title>{e(name)} {clock(a0)}–{clock(b0)}</title></rect>'
            )
    p.append("</svg>")
    return "".join(p)


# ------------------------------------------------------------------------------------ HTML
CSS = """
:root{--ink:#17324d;--text:#1f2933;--muted:#5a6878;--line:#dce1e7;--soft:#f5f6f8}
*{box-sizing:border-box}
body{margin:0;background:#eef0f3;color:var(--text);font:14px/1.5 "Segoe UI",-apple-system,Helvetica,Arial,sans-serif}
.page{max-width:1040px;margin:24px auto;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}
header.top{background:var(--ink);color:#fff;padding:26px 36px 22px}
header.top .brand{font-size:12px;letter-spacing:.02em;color:#c9d6e4}
header.top h1{margin:6px 0 4px;font-size:24px;font-weight:600}
header.top .sub{color:#c9d6e4;font-size:13px}
.content{padding:8px 36px 32px}
h2{font-size:16px;color:var(--ink);margin:28px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h2 .n{color:var(--muted);font-weight:500;margin-right:6px}
.verdict{display:flex;gap:18px;align-items:center;flex-wrap:wrap;border-radius:8px;padding:16px 18px;margin-top:22px}
.verdict .score{font-size:34px;font-weight:700;font-variant-numeric:tabular-nums}
.verdict .tier{font-size:16px;font-weight:700}
.verdict .band{font-size:13px;color:var(--muted)}
.note{background:var(--soft);border-left:3px solid var(--ink);padding:10px 14px;font-size:13px;color:#33414f;margin-top:12px}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:14px}
.kpi{border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.kpi .v{font-size:20px;font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
.kpi .l{font-size:12px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--muted);font-weight:600;border-bottom:1px solid var(--line);padding:7px 8px;background:#fafbfc}
td{padding:7px 8px;border-bottom:1px solid #eef0f3;vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.zero td{color:#9aa4ae}
.dl{display:grid;grid-template-columns:170px 1fr 170px 1fr;gap:6px 14px;font-size:13px}
.dl dt{color:var(--muted)}.dl dd{margin:0}
.bar{height:8px;background:#eef0f3;border-radius:4px;overflow:hidden;min-width:80px}
.bar span{display:block;height:100%;background:#9a5f0e}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;background:var(--soft);color:#33414f}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--muted);margin-top:6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.muted{color:var(--muted)}.small{font-size:12px}
footer{border-top:1px solid var(--line);padding:14px 36px;font-size:12px;color:var(--muted);background:#fafbfc}
.actions{max-width:1040px;margin:16px auto 0;text-align:right}
.actions button{font:inherit;font-weight:600;padding:8px 14px;border-radius:6px;border:1px solid var(--line);background:#fff;cursor:pointer}
@media print{body{background:#fff}.page{border:0;margin:0;max-width:none;border-radius:0}.actions{display:none}
 h2{break-after:avoid}table,.kpis,.verdict,svg{break-inside:avoid}.pb{break-before:page}
 header.top,.verdict,.bar span{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
@page{size:A4;margin:14mm}
@media(max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}.dl{grid-template-columns:130px 1fr}.content,header.top,footer{padding-left:18px;padding-right:18px}}
"""

KIND_COLOR = {
    "screen": "#2c5f8a",
    "device": "#2c5f8a",
    "environment": "#9a5f0e",
    "presence": "#7a4bb3",
    "identity": "#b42318",
    "attention": "#5a6878",
    "audio_voice": "#2e7d4f",
    "sound": "#2e7d4f",
    "pose": "#9a5f0e",
    "liveness": "#b42318",
}
KIND_NAME = {
    "screen": "Browser",
    "device": "Browser",
    "environment": "Camera",
    "presence": "Camera",
    "identity": "Camera",
    "attention": "Camera",
    "audio_voice": "Microphone",
    "sound": "Microphone",
    "pose": "Camera",
    "liveness": "Camera",
}


def render_report(
    ra: dict[str, Any],
    duration_s: float,
    grid: pd.DataFrame | None,
    model: Any,
    events: pd.DataFrame | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    meta = meta or {}
    grid = grid if grid is not None else pd.DataFrame()
    dg = session_digest(grid, events, duration_s) if len(grid) else None
    rec_name, rec_fg, rec_bg = REC[ra["recommendation"]]
    lo, hi = ra["confidence_band"]
    tiers = model.meta["tiers"]
    dur_ms = int(duration_s * 1000)
    c = dg["counters"] if dg else {}
    act = {r["key"]: r for r in (dg["activity"] if dg else [])}

    def secs(key: str) -> str:
        r = act.get(key)
        return dur(r["seconds"]) if r else "n/a"

    # ---- header + summary
    simulated = str(meta.get("source", "")).startswith("Simulated")
    title = "Simulated session" if simulated else (meta.get("participant_code") or "Session")
    started = meta.get("started_at") or ""
    sub = " · ".join(
        x
        for x in [f"Session {ra['session_id']}", started, f"length {dur(duration_s)}", meta.get("script_title") or ""]
        if x
    )
    out = [
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Review report — {e(ra['session_id'])}</title><style>{CSS}</style></head><body>"
        '<div class="actions"><button onclick="window.print()">Print or save as PDF</button></div><div class="page">'
        f'<header class="top"><div class="brand">RiskFusion · Session review report</div><h1>{e(title)}</h1>'
        f'<div class="sub">{e(sub)}</div></header><div class="content">'
    ]
    out.append(
        f'<div class="verdict" style="background:{rec_bg}"><div class="score" style="color:{rec_fg}">{ra["overall_risk"]:.2f}</div>'
        f'<div><div class="tier" style="color:{rec_fg}">{e(rec_name)}</div>'
        f'<div class="band">Calibrated risk 0–1 · 90% band {lo:.2f}–{hi:.2f} · human-review threshold {tiers["HUMAN_REVIEW"]:.2f}</div></div></div>'
    )
    out.append(
        '<div class="note"><strong>This is a recommendation for a human reviewer, not a finding of misconduct.</strong> '
        "The score estimates how likely it is that the session contains behaviour worth checking. Detectors make "
        "mistakes, especially in poor light, with low-quality webcams and in noisy rooms. Look at the recording "
        "before drawing any conclusion. Missing signals never raise the score.</div>"
    )
    if dg:
        kp = [
            (dur(duration_s), "Session length"),
            (secs("gaze_off"), "Looking away"),
            (secs("phone"), "Phone in view"),
            (secs("no_face"), "No face in view"),
            (f"{c['tab_leaves']}×", f"Left exam tab ({secs('tab_hidden')})"),
            (f"{c['pastes']}×", f"Pastes ({c['pasted_chars']} chars)"),
            (secs("foreign"), "Another voice"),
            (secs("multi_face"), "Second face"),
        ]
        out.append(
            '<div class="kpis">'
            + "".join(
                f'<div class="kpi"><div class="v">{e(v)}</div><div class="l">{e(lbl)}</div></div>' for v, lbl in kp
            )
            + "</div>"
        )

    # ---- 1 session details
    cond = [
        ("Participant", "— (simulated)" if simulated else meta.get("participant_code") or "—"),
        ("Recorded at", meta.get("started_at") or "—"),
        ("Behaviour profile" if simulated else "Script", meta.get("script_title") or meta.get("script_id") or "—"),
        ("Length", dur(duration_s)),
        ("Lighting", meta.get("lighting", "—")),
        ("Webcam", meta.get("webcam_class", "—")),
        ("Room noise", meta.get("room_noise", "—")),
        ("Glasses / head covering", f"{meta.get('eyewear', '—')} / {meta.get('head_covering', '—')}"),
        ("Browser", meta.get("browser", "—")),
        ("Source", meta.get("source", "—")),
    ]
    out.append(
        '<h2><span class="n">1</span>Session details</h2><dl class="dl">'
        + "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in cond)
        + "</dl>"
    )

    # ---- 2 activity summary
    if dg:
        rows = []
        for r in dg["activity"]:
            share = r["share"]
            bar = (
                f'<div class="bar"><span style="width:{min(100, (share or 0) * 100):.1f}%"></span></div>'
                if share is not None
                else ""
            )
            zero = ' class="zero"' if r["seconds"] == 0 else ""
            rows.append(
                f"<tr{zero}><td>{e(r['label'])}</td><td class='small muted'>{e(CHANNEL_NAMES[r['channel']])}</td>"
                f"<td class='num'>{dur(r['seconds'])}</td><td class='num'>{'' if share is None else f'{share * 100:.1f}%'}</td>"
                f"<td>{bar}</td><td class='num'>{r['episodes']}</td><td class='num'>{dur(r['longest']) if r['longest'] else '—'}</td>"
                f"<td class='num'>{clock(r['first'] * 1000) if r['first'] is not None else '—'}</td></tr>"
            )
        out.append(
            '<h2><span class="n">2</span>What happened in the session</h2>'
            '<p class="small muted">Time each behaviour was observed, as detected automatically. "Share" is the '
            "share of the time that signal could be observed.</p>"
            "<table><tr><th>Observation</th><th>Source</th><th class='num'>Total time</th><th class='num'>Share</th><th></th>"
            "<th class='num'>Times</th><th class='num'>Longest</th><th class='num'>First at</th></tr>"
            + "".join(rows)
            + "</table>"
        )
        other = [
            ("Left the exam tab", times(c["tab_leaves"])),
            ("Left full-screen mode", times(c["fullscreen_exits"])),
            ("Pasted into answers", f"{times(c['pastes'])}, {c['pasted_chars']} characters"),
            ("Keystrokes", "—" if c["keystrokes"] is None else f"{c['keystrokes']:,}"),
            (
                "Displays connected",
                "not reported by this browser" if c["max_monitors"] is None else str(c["max_monitors"]),
            ),
            (
                "Identity checks",
                f"{c['identity_checks']}"
                + (
                    ""
                    if c["identity_min_similarity"] is None
                    else f", lowest similarity {c['identity_min_similarity']:.2f}"
                ),
            ),
            ("Liveness checks", f"{c['liveness_checks']}, failed {c['liveness_failed']}"),
            (
                "Sounds (time heard)",
                ", ".join(
                    f"{k.replace('_burst', ' typing').replace('_', ' ')} {dur(v)}" for k, v in c["audio_events"].items()
                )
                or "none detected",
            ),
            ("Objects seen", ", ".join(f"{k.replace('_', ' ')} ×{v}" for k, v in c["objects"].items()) or "none"),
        ]
        out.append(
            "<table style='margin-top:12px'>"
            + "".join(f"<tr><td style='width:34%'>{e(k)}</td><td>{e(v)}</td></tr>" for k, v in other)
            + "</table>"
        )

    # ---- 3 timeline
    if dg:
        lanes: list[tuple[str, list[tuple[float, float]], str]] = []
        if meta.get("script_intervals"):
            lanes.append(
                ("Scripted episodes*", [(i["start_ms"], i["end_ms"]) for i in meta["script_intervals"]], "#b42318")
            )
        lanes.append(("Moments to check", [(f["t_start_ms"], f["t_end_ms"]) for f in ra["flags"]], "#9a5f0e"))
        for key, label, _ch, _m, _t in ACTIVITY:
            if key in grid and key != "vad":
                x = np.nan_to_num(grid[key].to_numpy(dtype=float)) > 0.5
                lanes.append((label, [(a * 1000, b * 1000) for a, b in _runs(x)], "#2c4a67"))
        unk = grid["no_face"].isna().to_numpy() if "no_face" in grid else np.zeros(0, bool)
        lanes.append(("Video unavailable", [(a * 1000, b * 1000) for a, b in _runs(unk)], "#b8c2cc"))
        out.append(
            '<h2><span class="n">3</span>Timeline</h2>'
            + _timeline(dur_ms, lanes)
            + '<div class="legend"><span><i style="background:#9a5f0e"></i>Moments to check</span>'
            '<span><i style="background:#2c4a67"></i>Observed signal</span><span><i style="background:#b8c2cc"></i>Not observable</span>'
            + (
                '<span><i style="background:#b42318"></i>*Scripted mock-session episodes (research ground truth)</span>'
                if meta.get("script_intervals")
                else ""
            )
            + "</div>"
        )

    # ---- 4 why
    out.append(
        '<h2><span class="n">4</span>Why the score is what it is</h2><table><tr><th>Reason</th><th class="num">Effect</th></tr>'
        + "".join(
            f"<tr><td>{e(t['explanation'])}</td><td class='num'>{'+' if t['shap'] > 0 else '−'}{abs(t['shap']):.2f}</td></tr>"
            for t in ra["top_contributors"]
        )
        + "</table>"
        '<p class="small muted">Effect is each factor\'s contribution to the model score (log-odds); larger means stronger.</p>'
    )

    # ---- 5 moments to check
    out.append('<h2><span class="n">5</span>Moments to check</h2>')
    if ra["flags"]:
        out.append(
            "<table><tr><th class='num'>From</th><th class='num'>To</th><th class='num'>Length</th><th>What was observed</th><th class='num'>Confidence</th></tr>"
            + "".join(
                f"<tr><td class='num'>{clock(f['t_start_ms'])}</td><td class='num'>{clock(f['t_end_ms'])}</td>"
                f"<td class='num'>{dur((f['t_end_ms'] - f['t_start_ms']) / 1000)}</td><td>{e(f['explanation'])}</td>"
                f"<td class='num'>{f['confidence']:.2f}</td></tr>"
                for f in ra["flags"]
            )
            + "</table>"
        )
    else:
        out.append('<p class="muted">None. Sessions that need no review list no moments.</p>')

    # ---- 6 event log
    if dg:
        log = dg["log"]
        shown = log[:300]
        out.append(
            '<h2 class="pb"><span class="n">6</span>Event log</h2>'
            f'<p class="small muted">Everything notable, in order. {len(log)} entries'
            f"{'; the first 300 are shown' if len(log) > 300 else ''}. Browser events are exact; camera and "
            "microphone events are detected second by second.</p>"
        )
        if shown:
            out.append(
                "<table><tr><th class='num'>Time</th><th class='num'>Until</th><th>Source</th><th>Event</th></tr>"
                + "".join(
                    f"<tr><td class='num'>{clock(r['t'])}</td><td class='num'>{clock(r['end']) if r['end'] else ''}</td>"
                    f"<td><span class='pill' style='border-left:3px solid {KIND_COLOR.get(r['kind'], '#5a6878')}'>"
                    f"{KIND_NAME.get(r['kind'], r['kind'])}</span></td><td>{e(r['text'])}</td></tr>"
                    for r in shown
                )
                + "</table>"
            )
        else:
            out.append('<p class="muted">No notable events.</p>')

    # ---- 7 data quality
    missing = set(ra["channels_missing"])
    out.append(
        '<h2><span class="n">7</span>Signal availability</h2><table><tr><th>Signal</th><th>Status</th></tr>'
        + "".join(
            f"<tr><td>{e(CHANNEL_NAMES[ch])}</td><td>{'<span class=pill>Unavailable</span> — never counts against the candidate' if ch in missing else 'Available'}</td></tr>"
            for ch in CHANNELS
        )
        + "</table>"
    )

    # ---- 8 reviews + technical
    if meta.get("reviews"):
        out.append(
            '<h2><span class="n">8</span>Reviewer decisions</h2><table><tr><th>When</th><th>Reviewer</th><th>Decision</th><th>Note</th></tr>'
            + "".join(
                f"<tr><td>{e(r['at'])}</td><td>{e(r['reviewer'])}</td><td>{e(r['verdict'])}</td><td>{e(r.get('note') or '')}</td></tr>"
                for r in meta["reviews"]
            )
            + "</table>"
        )
    tech = [
        ("Model", f"{ra['model_version']} / {ra['feature_version']}"),
        ("Events analysed", f"{c.get('events_total', 0):,}"),
        (
            "Review tiers",
            f"routine ≥ {tiers['ROUTINE_REVIEW']:.2f}, human ≥ {tiers['HUMAN_REVIEW']:.2f}, priority ≥ {tiers['PRIORITY_REVIEW']:.2f}",
        ),
    ]
    for rec in meta.get("recordings", []):
        tech.append((rec["kind"], f"{rec['size']} · SHA-256 {rec['sha256'][:24]}…"))
    out.append(
        '<h2><span class="n">9</span>Technical details</h2><dl class="dl">'
        + "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in tech)
        + "</dl>"
    )
    out.append(
        "</div><footer>Generated "
        + time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        + " by RiskFusion. Research prototype trained on simulated data; see the model card for performance, "
        "fairness results and known limitations. Raw recordings are deleted at the end of the project.</footer></div></body></html>"
    )
    return "".join(out)
