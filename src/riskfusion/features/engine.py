"""Feature engineering (Phase 3: FR-9 .. FR-14) and time-bounded flags (FR-25).

Input: the flat event.v1 rows of ONE session (simulator Parquet, detector output or browser telemetry — the
same code serves all three) plus the session duration. Output: a session feature vector, the per-second grid
and a list of flags.

Design decisions (SRS_AUDIT A-9, A-10):
* Grid: one row per second. A signal is NaN where its channel produced no usable observation; slow-cadence
  detectors (identity every 5 s, environment/pose every 2 s, liveness every 10 s) are carried forward for
  exactly one cadence period, never further.
* Windowed aggregates are reduced to session level by max and fraction-over-threshold of the rolling mean.
* Baseline normalisation applies only to continuous behavioural signals, uses median/MAD of the first 60
  usable seconds with a floor, and is skipped (flagged) when fewer than 30 usable baseline seconds exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from riskfusion.contracts import CHANNELS

FEATURE_VERSION = "features-v1.0.0"
WINDOWS = (10, 60, 300)
LATE_WINDOW_MS = 30_000

# signal name -> (channel, kind, threshold for count_over_threshold / longest_run)
SIGNALS: dict[str, tuple[str, str, float]] = {
    "no_face": ("presence", "binary", 0.5),
    "multi_face": ("presence", "binary", 0.5),
    "face_area": ("presence", "continuous", 0.02),
    "face_offset": ("presence", "continuous", 0.5),
    "id_mismatch": ("identity", "binary", 0.5),
    "id_sim_low": ("identity", "continuous", 0.5),
    "pad_fail": ("liveness", "binary", 0.5),
    "yaw_abs": ("attention", "continuous", 25.0),
    "pitch_down": ("attention", "continuous", 15.0),
    "gaze_off": ("attention", "binary", 0.5),
    "extra_person": ("environment", "binary", 0.5),
    "phone": ("environment", "binary", 0.5),
    "paper": ("environment", "binary", 0.5),
    "reach": ("pose", "binary", 0.5),
    "hands_low": ("pose", "binary", 0.5),
    "vad": ("audio_voice", "binary", 0.5),
    "foreign": ("audio_voice", "binary", 0.5),
    "phone_ring": ("audio_event", "binary", 0.5),
    "paper_rustle": ("audio_event", "binary", 0.5),
    "tab_hidden": ("screen", "binary", 0.5),
    "fs_off": ("screen", "binary", 0.5),
    "paste_chars": ("screen", "continuous", 50.0),
    "multi_monitor": ("device", "binary", 0.5),
    "keys_per_s": ("behavioral", "continuous", 3.0),
}
BASELINE_SIGNALS = ("yaw_abs", "pitch_down", "face_area", "face_offset", "keys_per_s")
CADENCE = {"identity": 5, "liveness": 10, "environment": 2, "pose": 2}


@dataclass
class SessionFeatures:
    session_id: str
    features: dict[str, float]
    grid: pd.DataFrame  # per-second signals (NaN = unknown)
    flags: list[dict[str, Any]] = field(default_factory=list)
    late_events: int = 0


# ------------------------------------------------------------------------------------ FR-10
def reorder_with_watermark(ev: pd.DataFrame, window_ms: int = LATE_WINDOW_MS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accept events in arrival order; anything older than (max ts seen - window) is late and rejected."""
    ts = ev["ts_ms"].to_numpy(dtype=np.int64)
    running_max = np.maximum.accumulate(ts) if ts.size else ts
    late = ts < (running_max - window_ms)
    ok = ev.loc[~late].sort_values("ts_ms", kind="stable")
    return ok, ev.loc[late]


# ------------------------------------------------------------------------------------ FR-9 / FR-11
def build_grid(ev: pd.DataFrame, duration_s: float) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    T = max(1, int(np.ceil(duration_s)))
    sec = np.clip(ev["ts_ms"].to_numpy(dtype=np.int64) // 1000, 0, T - 1)
    et = ev["event_type"].astype(str).to_numpy()
    lab = ev["label"].astype(object).to_numpy()
    p = [ev[f"p{i}"].to_numpy(dtype=float) for i in range(4)]
    luma = ev["frame_luma"].to_numpy(dtype=float)

    def per_sec(mask: np.ndarray, values: np.ndarray, how: str = "max") -> tuple[np.ndarray, np.ndarray]:
        out = np.full(T, np.nan)
        seen = np.zeros(T, dtype=bool)
        if mask.any():
            s, v = sec[mask], values[mask]
            seen[s] = True
            if how == "max":
                tmp = np.full(T, -np.inf)
                np.maximum.at(tmp, s, v)
            elif how == "min":
                tmp = np.full(T, np.inf)
                np.minimum.at(tmp, s, v)
            else:
                tmp = np.zeros(T)
                np.add.at(tmp, s, v)
            out[seen] = tmp[seen]
        return out, seen

    def carry(x: np.ndarray, seen: np.ndarray, k: int) -> np.ndarray:
        """Forward-fill an observation for k-1 further seconds (one cadence period), never beyond."""
        if k <= 1:
            return x
        idx = np.where(seen, np.arange(T), -1)
        last = np.maximum.accumulate(idx)
        ok = (last >= 0) & (np.arange(T) - last < k)
        out = np.full(T, np.nan)
        out[ok] = x[last[ok]]
        return out

    G: dict[str, np.ndarray] = {}
    observed: dict[str, np.ndarray] = {}
    # presence
    m = et == "FACE_OBSERVATION"
    fc, seen = per_sec(m, p[0])
    G["no_face"] = np.where(seen, (fc < 1).astype(float), np.nan)
    G["multi_face"] = np.where(seen, (fc >= 2).astype(float), np.nan)
    G["face_area"], _ = per_sec(m, p[1])
    off = np.hypot(p[2], p[3])
    G["face_offset"], _ = per_sec(m, off)
    observed["presence"] = seen
    # identity
    m = et == "IDENTITY_CHECK"
    sim, seen = per_sec(m, p[0], "min")
    ver, _ = per_sec(m, p[1], "min")
    G["id_sim_low"] = carry(np.where(seen, 1.0 - sim, np.nan), seen, CADENCE["identity"])
    G["id_mismatch"] = carry(np.where(seen, (ver < 0.5).astype(float), np.nan), seen, CADENCE["identity"])
    observed["identity"] = ~np.isnan(G["id_mismatch"])
    # liveness
    m = et == "LIVENESS_CHECK"
    pp, seen = per_sec(m, p[1], "min")
    G["pad_fail"] = carry(np.where(seen, (pp < 0.5).astype(float), np.nan), seen, CADENCE["liveness"])
    observed["liveness"] = ~np.isnan(G["pad_fail"])
    # attention
    m = et == "HEAD_GAZE"
    ya, seen = per_sec(m, np.abs(p[0]))
    pi, _ = per_sec(m, p[1], "min")
    gp, _ = per_sec(m, p[3], "min")
    G["yaw_abs"] = ya
    base_pitch = np.nanmedian(pi) if np.isfinite(pi).any() else 0.0
    G["pitch_down"] = np.where(seen, np.clip(base_pitch - pi, 0, None), np.nan)
    G["gaze_off"] = np.where(seen, (gp < 0.5).astype(float), np.nan)
    observed["attention"] = seen
    # environment
    m = et == "PERSON_COUNT"
    npers, seen_pc = per_sec(m, p[0])
    env_obs = ~np.isnan(carry(np.where(seen_pc, 1.0, np.nan), seen_pc, CADENCE["environment"]))
    G["extra_person"] = carry(np.where(seen_pc, (npers >= 2).astype(float), np.nan), seen_pc, CADENCE["environment"])
    for name, labels in (("phone", {"cell_phone"}), ("paper", {"book", "paper"})):
        mm = (et == "OBJECT_DETECTED") & np.isin(lab, list(labels))
        hit, seen_o = per_sec(mm, np.ones(len(et)))
        G[name] = np.where(env_obs, np.where(seen_o, 1.0, 0.0), np.nan)
    observed["environment"] = env_obs
    # pose
    m = et == "BODY_POSE"
    hands, seen = per_sec(m, p[0], "min")
    reach, _ = per_sec(m, p[1])
    G["reach"] = carry(np.where(seen, reach, np.nan), seen, CADENCE["pose"])
    G["hands_low"] = carry(np.where(seen, (hands < 1).astype(float), np.nan), seen, CADENCE["pose"])
    observed["pose"] = ~np.isnan(G["reach"])
    # audio
    m = et == "VOICE_ACTIVITY"
    G["vad"], seen = per_sec(m, p[0])
    G["foreign"], _ = per_sec(m, p[3])
    observed["audio_voice"] = seen
    for name, labels in (("phone_ring", {"phone_ring"}), ("paper_rustle", {"paper_rustle"})):
        mm = (et == "AUDIO_EVENT") & np.isin(lab, list(labels))
        _, seen_e = per_sec(mm, np.ones(len(et)))
        G[name] = np.where(seen, np.where(seen_e, 1.0, 0.0), np.nan)
    audio_ev_unknown = (et == "AUDIO_EVENT_UNKNOWN").any() and not (et == "AUDIO_EVENT").any()
    if audio_ev_unknown:
        G["phone_ring"][:] = np.nan
        G["paper_rustle"][:] = np.nan
    observed["audio_event"] = ~np.isnan(G["phone_ring"])

    # screen/device: state machines from change events
    def state(mask: np.ndarray, vals: np.ndarray, init: float) -> np.ndarray:
        out = np.full(T, init)
        if mask.any():
            order = np.argsort(ev["ts_ms"].to_numpy()[mask], kind="stable")
            for s, v in zip(sec[mask][order], vals[mask][order], strict=True):
                out[s:] = v
        return out

    screen_unknown = (et == "SCREEN_UNKNOWN").any() and not (et == "TAB_VISIBILITY").any()
    if screen_unknown:
        for k in ("tab_hidden", "fs_off", "paste_chars"):
            G[k] = np.full(T, np.nan)
        observed["screen"] = np.zeros(T, dtype=bool)
    else:
        G["tab_hidden"] = state(et == "TAB_VISIBILITY", p[0], 0.0)
        G["fs_off"] = 1.0 - state(et == "FULLSCREEN_CHANGE", p[0], 1.0)
        pc, _ = per_sec(et == "PASTE", p[0], "sum")
        G["paste_chars"] = np.nan_to_num(pc)
        observed["screen"] = np.ones(T, dtype=bool)
    mc = et == "MONITOR_COUNT"
    if mc.any():
        G["multi_monitor"] = (state(mc, p[0], 1.0) >= 2).astype(float)
        observed["device"] = np.ones(T, dtype=bool)
    else:
        G["multi_monitor"] = np.full(T, np.nan)
        observed["device"] = np.zeros(T, dtype=bool)
    m = et == "INPUT_ACTIVITY"
    keys = np.full(T, np.nan)
    if m.any():
        keys[:] = 0.0
        for s, k, w in zip(sec[m], p[0][m], p[1][m], strict=True):
            w = max(1, int(round(w)))
            lo = max(0, s - w + 1)
            keys[lo : s + 1] = k / w
    G["keys_per_s"] = keys
    observed["behavioral"] = ~np.isnan(keys)
    lu, _ = per_sec(np.isfinite(luma) & (et == "FACE_OBSERVATION"), luma)
    grid = pd.DataFrame({k: G[k] for k in SIGNALS})
    grid["luma"] = lu
    return grid, observed


# ------------------------------------------------------------------------------------ FR-12
def _rolling_mean(X: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean over the last w seconds, NaN-aware; NaN when fewer than w/2 observed seconds."""
    S, T = X.shape
    obs = ~np.isnan(X)
    v = np.where(obs, X, 0.0)
    cs = np.concatenate([np.zeros((S, 1)), np.cumsum(v, 1)], 1)
    cn = np.concatenate([np.zeros((S, 1)), np.cumsum(obs, 1)], 1)
    hi = np.arange(1, T + 1)
    lo = np.maximum(0, hi - w)
    s = cs[:, hi] - cs[:, lo]
    n = cn[:, hi] - cn[:, lo]
    need = max(1, min(w, T) // 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= need, s / np.maximum(n, 1), np.nan)


def _longest_run(b: np.ndarray) -> int:
    if not b.any():
        return 0
    d = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def _nanstat(fn: Any, x: np.ndarray, default: float = 0.0) -> float:
    x = x[np.isfinite(x)]
    return float(fn(x)) if x.size else default


def session_features(
    grid: pd.DataFrame,
    observed: dict[str, np.ndarray],
    duration_s: float,
    windows: tuple[int, ...] = WINDOWS,
    baseline: bool = True,
) -> dict[str, float]:
    T = len(grid)
    f: dict[str, float] = {"duration_min": duration_s / 60.0}
    names = list(SIGNALS)
    X = grid[names].to_numpy(dtype=float).T
    for i, name in enumerate(names):
        x = X[i]
        thr = SIGNALS[name][2]
        f[f"{name}__mean"] = _nanstat(np.mean, x)
        f[f"{name}__max"] = _nanstat(np.max, x)
        f[f"{name}__std"] = _nanstat(np.std, x)
        f[f"{name}__longest_run_s"] = float(_longest_run(np.nan_to_num(x) > thr))
    for w in windows:
        R = _rolling_mean(X, w)
        for i, name in enumerate(names):
            thr = SIGNALS[name][2]
            r = R[i]
            f[f"{name}__w{w}_max"] = _nanstat(np.max, r)
            f[f"{name}__w{w}_over"] = float(
                np.mean(np.nan_to_num(r) > (thr if SIGNALS[name][1] == "continuous" else 0.5 * thr))
            )
    # FR-13 baseline normalisation (continuous behaviour only)
    usable = np.isfinite(grid["yaw_abs"].to_numpy())
    first = np.flatnonzero(usable)[:60]
    f["baseline_available"] = float(baseline and first.size >= 30)
    for name in BASELINE_SIGNALS:
        x = grid[name].to_numpy(dtype=float)
        if baseline and first.size >= 30 and np.isfinite(x[first]).sum() >= 30:
            ref = x[first][np.isfinite(x[first])]
            med = np.median(ref)
            mad = 1.4826 * np.median(np.abs(ref - med))
            floor = {"yaw_abs": 3.0, "pitch_down": 3.0, "face_area": 0.005, "face_offset": 0.05, "keys_per_s": 0.5}[
                name
            ]
            z = (x - med) / max(mad, floor)
            f[f"bn_{name}__w60_max"] = _nanstat(np.max, _rolling_mean(np.abs(z)[None, :], 60)[0])
            f[f"bn_{name}__frac_over3"] = float(np.mean(np.nan_to_num(np.abs(z)) > 3))
        else:
            f[f"bn_{name}__w60_max"] = np.nan
            f[f"bn_{name}__frac_over3"] = np.nan
    # FR-14 cross-channel interactions
    g = {k: np.nan_to_num(grid[k].to_numpy(dtype=float)) for k in SIGNALS}
    f["x_gaze_off_and_foreign"] = float(np.mean((g["gaze_off"] > 0.5) & (g["foreign"] > 0.5)))
    f["x_phone_and_pitch_down"] = float(np.mean((g["phone"] > 0.5) & (g["pitch_down"] > 15)))
    f["x_multi_face_and_voice"] = float(
        np.mean(((g["multi_face"] > 0.5) | (g["extra_person"] > 0.5)) & (g["vad"] > 0.5))
    )
    f["x_mismatch_and_face"] = float(np.mean((g["id_mismatch"] > 0.5) & (g["no_face"] < 0.5)))
    f["x_gaze_off_no_typing"] = float(np.mean((g["gaze_off"] > 0.5) & (g["keys_per_s"] < 0.2)))
    f["x_paper_and_pitch_down"] = float(np.mean((g["paper"] > 0.5) & (g["pitch_down"] > 12)))
    tab_end = np.flatnonzero(np.diff(np.concatenate(([0], (g["tab_hidden"] > 0.5).astype(int)))) == -1)
    paste_at = np.flatnonzero(g["paste_chars"] > 0)
    f["x_paste_after_tab_return"] = float(sum(((paste_at >= e) & (paste_at <= e + 30)).any() for e in tab_end))
    f["x_foreign_and_keys"] = float(np.mean((g["foreign"] > 0.5) & (g["keys_per_s"] > 1)))
    # availability (monotone non-increasing in the model: missing data must not raise risk, FR-30)
    for ch in CHANNELS:
        obs = observed.get(ch)
        f[f"unk_{ch}"] = 1.0 if obs is None else float(1.0 - obs.mean()) if T else 1.0
    lu = grid["luma"].to_numpy(dtype=float)
    f["q_luma_mean"] = _nanstat(np.mean, lu, np.nan)
    f["q_luma_p10"] = _nanstat(lambda v: np.percentile(v, 10), lu, np.nan)
    return f


# ------------------------------------------------------------------------------------ FR-25 flags
FLAG_RULES: dict[str, dict[str, Any]] = {
    # flag type -> signal expression over the grid, rolling window (s), density threshold
    "PHONE_USE": {"expr": lambda g: g["phone"], "w": 10, "thr": 0.2, "text": "A phone was detected in view"},
    "SECOND_PERSON": {
        "expr": lambda g: np.maximum(g["multi_face"], g["extra_person"]),
        "w": 10,
        "thr": 0.3,
        "text": "More than one person was visible",
    },
    "IMPERSONATION": {
        "expr": lambda g: g["id_mismatch"],
        "w": 30,
        "thr": 0.5,
        "gap": 90,
        "text": "The face did not match the enrolment photo",
    },
    "TAB_SWITCH": {"expr": lambda g: g["tab_hidden"], "w": 1, "thr": 0.5, "text": "The exam tab was not visible"},
    "REMOTE_ASSISTANCE": {
        "expr": lambda g: g["foreign"] * (1 - np.nan_to_num(g["multi_face"])),
        "w": 20,
        "thr": 0.25,
        "text": "A voice other than the candidate's was heard",
    },
    "NOTE_READING": {
        "expr": lambda g: np.minimum(g["paper"] + 0.0, 1) * (np.nan_to_num(g["pitch_down"]) > 12),
        "w": 20,
        "thr": 0.25,
        "text": "Paper was visible while the candidate looked down",
    },
}


def generate_flags(
    session_id: str, grid: pd.DataFrame, min_len_s: int = 5, merge_gap_s: int = 10
) -> list[dict[str, Any]]:
    gd = {k: grid[k].to_numpy(dtype=float) for k in grid.columns}
    flags: list[dict[str, Any]] = []
    n = 0
    for ftype, rule in FLAG_RULES.items():
        sig = np.nan_to_num(np.asarray(rule["expr"](gd), dtype=float))
        dens = _rolling_mean(sig[None, :], rule["w"])[0] if rule["w"] > 1 else sig
        # centre the window so intervals line up with the behaviour, not trail it
        shift = rule["w"] // 2
        dens = np.concatenate([np.nan_to_num(dens[shift:]), np.zeros(shift)])
        on = dens >= rule["thr"]
        d = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
        starts = [int(v) for v in np.flatnonzero(d == 1)]
        ends = [int(v) for v in np.flatnonzero(d == -1)]
        merged: list[list[int]] = []
        gap = rule.get("gap", merge_gap_s)
        for s, e in zip(starts, ends, strict=True):
            if merged and s - merged[-1][1] <= gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        for s, e in merged:
            if e - s < min_len_s:
                continue
            n += 1
            conf = float(np.clip(np.mean(sig[s:e]), 0, 1))
            flags.append(
                {
                    "flag_id": f"flg_{n:03d}",
                    "type": ftype,
                    "t_start_ms": int(s * 1000),
                    "t_end_ms": int(e * 1000),
                    "confidence": round(conf, 3),
                    "explanation": f"{rule['text']} for {e - s} seconds.",
                    "evidence_ref": f"sessions/{session_id}#t={s},{e}",
                }
            )
    return sorted(flags, key=lambda x: x["t_start_ms"])


def compute_session(
    session_id: str,
    events: pd.DataFrame,
    duration_s: float,
    *,
    windows: tuple[int, ...] = WINDOWS,
    baseline: bool = True,
    reorder: bool = False,
) -> SessionFeatures:
    late = 0
    if reorder:
        events, late_df = reorder_with_watermark(events)
        late = len(late_df)
    grid, observed = build_grid(events, duration_s)
    feats = session_features(grid, observed, duration_s, windows=windows, baseline=baseline)
    return SessionFeatures(session_id, feats, grid, generate_flags(session_id, grid), late)
