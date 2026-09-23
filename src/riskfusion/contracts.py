"""Internal data contract (SRS Section 6).

One registry defines every channel and event type. The JSON Schemas in ``contracts/`` are
generated from it (``python -m riskfusion.contracts --write``) so the documented contract and
the validators can never drift apart.

Two validation paths share the same rules:

* :func:`validate_event` — one event as a JSON-like dict (API ingestion, real detectors).
* :func:`validate_event_frame` — a flattened columnar batch (simulator output, Parquet).

Invalid events are never dropped silently: callers receive a reason for each rejection and must
route it to the dead-letter store (FR-8).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

EVENT_SCHEMA = "event.v1"
RISK_SCHEMA = "risk_assessment.v1"

CHANNELS: tuple[str, ...] = (
    "presence",
    "identity",
    "liveness",
    "attention",
    "environment",
    "pose",
    "audio_voice",
    "audio_event",
    "screen",
    "device",
    "behavioral",
)

# Recommendations deliberately contain no VIOLATION value (SRS 6.2, CON-5, ETH-6).
RECOMMENDATIONS: tuple[str, ...] = ("NO_ACTION", "ROUTINE_REVIEW", "HUMAN_REVIEW", "PRIORITY_REVIEW")

FieldKind = Literal["float", "int", "bool", "label"]


@dataclass(frozen=True)
class PayloadField:
    name: str
    kind: FieldKind
    lo: float | None = None
    hi: float | None = None
    allowed: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EventType:
    name: str
    channel: str
    fields: tuple[PayloadField, ...]
    unknown: bool = False


_UNKNOWN_REASONS = (
    "no_frames",
    "low_light",
    "occluded",
    "device_unavailable",
    "connection_lost",
    "detector_error",
    "permission_denied",
    "not_observable",
)


def _unknown(channel: str) -> EventType:
    return EventType(
        name=f"{channel.upper()}_UNKNOWN",
        channel=channel,
        fields=(PayloadField("reason", "label", allowed=_UNKNOWN_REASONS),),
        unknown=True,
    )


OBJECT_LABELS = ("cell_phone", "book", "laptop", "tv", "remote", "person", "headphones", "paper")
AUDIO_EVENT_LABELS = ("keyboard_burst", "phone_ring", "paper_rustle", "door", "other")

_REGISTRY: tuple[EventType, ...] = (
    EventType(
        "FACE_OBSERVATION",
        "presence",
        (
            PayloadField("face_count", "int", 0, 20),
            PayloadField("bbox_area_ratio", "float", 0, 1),
            PayloadField("center_offset_x", "float", -1, 1),
            PayloadField("center_offset_y", "float", -1, 1),
        ),
    ),
    EventType(
        "IDENTITY_CHECK",
        "identity",
        (
            PayloadField("cosine_sim", "float", -1, 1),
            PayloadField("verified", "bool"),
            PayloadField("face_quality", "float", 0, 1),
        ),
    ),
    EventType(
        "LIVENESS_CHECK",
        "liveness",
        (PayloadField("pad_score", "float", 0, 1), PayloadField("pad_pass", "bool")),
    ),
    EventType(
        "HEAD_GAZE",
        "attention",
        (
            PayloadField("yaw", "float", -180, 180),
            PayloadField("pitch", "float", -180, 180),
            PayloadField("roll", "float", -180, 180),
            PayloadField("gaze_on_screen_prob", "float", 0, 1),
        ),
    ),
    EventType("PERSON_COUNT", "environment", (PayloadField("n_persons", "int", 0, 20),)),
    EventType(
        "OBJECT_DETECTED",
        "environment",
        (
            PayloadField("label", "label", allowed=OBJECT_LABELS),
            PayloadField("bbox_x", "float", 0, 1),
            PayloadField("bbox_y", "float", 0, 1),
            PayloadField("bbox_w", "float", 0, 1),
            PayloadField("bbox_h", "float", 0, 1),
        ),
    ),
    EventType(
        "BODY_POSE",
        "pose",
        (
            PayloadField("hands_visible_count", "int", 0, 4),
            PayloadField("reach_off_frame", "bool"),
            PayloadField("torso_rotation", "float", -180, 180),
        ),
    ),
    EventType(
        "VOICE_ACTIVITY",
        "audio_voice",
        (
            PayloadField("vad_active", "bool"),
            PayloadField("speaker_is_candidate", "bool"),
            PayloadField("n_speakers", "int", 0, 10),
            PayloadField("foreign_speech", "bool"),
        ),
    ),
    EventType("AUDIO_EVENT", "audio_event", (PayloadField("label", "label", allowed=AUDIO_EVENT_LABELS),)),
    EventType("TAB_VISIBILITY", "screen", (PayloadField("hidden", "bool"),)),
    EventType("FULLSCREEN_CHANGE", "screen", (PayloadField("active", "bool"),)),
    EventType("PASTE", "screen", (PayloadField("length", "int", 0, 10_000_000),)),
    EventType("MONITOR_COUNT", "device", (PayloadField("count", "int", 0, 16),)),
    EventType(
        "INPUT_ACTIVITY",
        "behavioral",
        (PayloadField("keystrokes", "int", 0, 100_000), PayloadField("window_s", "float", 0, 3600)),
    ),
) + tuple(_unknown(c) for c in CHANNELS)

EVENT_TYPES: dict[str, EventType] = {e.name: e for e in _REGISTRY}

# Flattened (columnar) storage layout. Numeric payload fields map in order onto p0..p3; a single
# categorical field maps onto ``label``. Every event type fits this layout (checked at import).
NUMERIC_SLOTS = ("p0", "p1", "p2", "p3")
FLAT_COLUMNS = (
    "session_id",
    "ts_ms",
    "channel",
    "detector",
    "detector_version",
    "event_type",
    "confidence",
    *NUMERIC_SLOTS,
    "label",
    "usable",
    "frame_blur",
    "frame_luma",
)


def _slot_map(et: EventType) -> dict[str, str]:
    mapping: dict[str, str] = {}
    numeric = [f for f in et.fields if f.kind != "label"]
    labels = [f for f in et.fields if f.kind == "label"]
    if len(numeric) > len(NUMERIC_SLOTS) or len(labels) > 1:
        raise ValueError(f"{et.name} does not fit the flat layout")
    for f, slot in zip(numeric, NUMERIC_SLOTS, strict=False):
        mapping[f.name] = slot
    for f in labels:
        mapping[f.name] = "label"
    return mapping


SLOT_MAP: dict[str, dict[str, str]] = {name: _slot_map(et) for name, et in EVENT_TYPES.items()}

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{3,64}$")


class ContractError(ValueError):
    """Raised when an event or assessment violates the contract."""


# --------------------------------------------------------------------------------------------
# Single-event validation
# --------------------------------------------------------------------------------------------
def _check_field(f: PayloadField, value: Any) -> str | None:
    if value is None:
        return f"payload.{f.name} missing"
    if f.kind == "bool":
        if not isinstance(value, bool):
            return f"payload.{f.name} must be boolean"
        return None
    if f.kind == "label":
        if not isinstance(value, str) or (f.allowed is not None and value not in f.allowed):
            return f"payload.{f.name} must be one of {list(f.allowed or [])}"
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"payload.{f.name} must be numeric"
    if f.kind == "int" and float(value) != int(value):
        return f"payload.{f.name} must be an integer"
    if not np.isfinite(float(value)):
        return f"payload.{f.name} must be finite"
    if f.lo is not None and value < f.lo:
        return f"payload.{f.name} below {f.lo}"
    if f.hi is not None and value > f.hi:
        return f"payload.{f.name} above {f.hi}"
    return None


def event_errors(event: dict[str, Any]) -> list[str]:
    """Return every contract violation in ``event`` (empty list = valid)."""
    errs: list[str] = []
    if event.get("schema") != EVENT_SCHEMA:
        errs.append(f"schema must be '{EVENT_SCHEMA}'")
    sid = event.get("session_id")
    if not isinstance(sid, str) or not _SESSION_ID_RE.match(sid):
        errs.append("session_id must match ^[A-Za-z0-9_-]{3,64}$")
    ts = event.get("ts_ms")
    if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
        errs.append("ts_ms must be a non-negative integer")
    channel = event.get("channel")
    if channel not in CHANNELS:
        errs.append(f"channel must be one of {list(CHANNELS)}")
    for key in ("detector", "detector_version"):
        if not isinstance(event.get(key), str) or not event.get(key):
            errs.append(f"{key} must be a non-empty string")
    conf = event.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        errs.append("confidence must be a number in [0, 1]")
    et = EVENT_TYPES.get(str(event.get("event_type")))
    if et is None:
        errs.append("event_type is not registered in event.v1")
    else:
        if channel in CHANNELS and et.channel != channel:
            errs.append(f"event_type {et.name} belongs to channel '{et.channel}', not '{channel}'")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            errs.append("payload must be an object")
        else:
            for f in et.fields:
                msg = _check_field(f, payload.get(f.name))
                if msg:
                    errs.append(msg)
            extra = set(payload) - {f.name for f in et.fields}
            if extra:
                errs.append(f"payload has unexpected fields {sorted(extra)}")
    quality = event.get("quality")
    if quality is not None:
        if not isinstance(quality, dict):
            errs.append("quality must be an object")
        else:
            if "usable" in quality and not isinstance(quality["usable"], bool):
                errs.append("quality.usable must be boolean")
            for k in ("frame_blur",):
                v = quality.get(k)
                if v is not None and (not isinstance(v, (int, float)) or not 0 <= v <= 1):
                    errs.append(f"quality.{k} must be in [0, 1]")
            v = quality.get("frame_luma")
            if v is not None and (not isinstance(v, (int, float)) or not 0 <= v <= 255):
                errs.append("quality.frame_luma must be in [0, 255]")
    return errs


def validate_event(event: dict[str, Any]) -> dict[str, Any]:
    errs = event_errors(event)
    if errs:
        raise ContractError("; ".join(errs))
    return event


# --------------------------------------------------------------------------------------------
# Flat <-> canonical conversion
# --------------------------------------------------------------------------------------------
def event_to_flat(event: dict[str, Any]) -> dict[str, Any]:
    """Convert a canonical (valid) event.v1 dict to one flat storage row."""
    row: dict[str, Any] = {c: None for c in FLAT_COLUMNS}
    for c in ("session_id", "ts_ms", "channel", "detector", "detector_version", "event_type", "confidence"):
        row[c] = event[c]
    for slot in NUMERIC_SLOTS:
        row[slot] = np.nan
    for field_name, slot in SLOT_MAP[event["event_type"]].items():
        value = event["payload"][field_name]
        row[slot] = value if slot == "label" else float(value)
    q = event.get("quality") or {}
    row["usable"] = bool(q.get("usable", True))
    row["frame_blur"] = float(q["frame_blur"]) if q.get("frame_blur") is not None else np.nan
    row["frame_luma"] = float(q["frame_luma"]) if q.get("frame_luma") is not None else np.nan
    return row


def flat_to_event(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the canonical event.v1 dict from a flat storage row."""
    et = EVENT_TYPES[row["event_type"]]
    payload: dict[str, Any] = {}
    for f in et.fields:
        slot = SLOT_MAP[et.name][f.name]
        raw = row[slot]
        if f.kind == "label":
            payload[f.name] = raw
        elif f.kind == "bool":
            payload[f.name] = bool(raw)
        elif f.kind == "int":
            payload[f.name] = int(raw)
        else:
            payload[f.name] = float(raw)
    quality: dict[str, Any] = {"usable": bool(row["usable"])}
    for k in ("frame_blur", "frame_luma"):
        v = row.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            quality[k] = float(v)
    return {
        "schema": EVENT_SCHEMA,
        "session_id": row["session_id"],
        "ts_ms": int(row["ts_ms"]),
        "channel": row["channel"],
        "detector": row["detector"],
        "detector_version": row["detector_version"],
        "event_type": et.name,
        "payload": payload,
        "confidence": float(row["confidence"]),
        "quality": quality,
    }


# --------------------------------------------------------------------------------------------
# Vectorised batch validation (same rules as event_errors, applied to columns)
# --------------------------------------------------------------------------------------------
def validate_event_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a flat event batch into (valid, rejected). ``rejected`` carries a ``reason`` column.

    The number of rows in ``valid`` + ``rejected`` always equals ``len(df)`` — nothing is dropped.
    """
    missing_cols = [c for c in FLAT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ContractError(f"batch is missing columns {missing_cols}")
    n = len(df)
    reasons = np.full(n, "", dtype=object)

    def flag(mask: np.ndarray, msg: str) -> None:
        idx = np.flatnonzero(mask)
        if idx.size:
            reasons[idx] = np.where(reasons[idx] == "", msg, reasons[idx] + "; " + msg)

    def cat(col: str) -> tuple[np.ndarray, pd.Index]:
        c = df[col] if isinstance(df[col].dtype, pd.CategoricalDtype) else df[col].astype("category")
        return c.cat.codes.to_numpy(), c.cat.categories

    def cat_bad(col: str, ok: Any) -> np.ndarray:
        """Evaluate a predicate once per distinct value, then broadcast to rows (null = bad)."""
        codes, cats = cat(col)
        good = np.array([bool(ok(v)) for v in cats] + [False], dtype=bool)  # index -1 -> False
        return ~good[codes]

    flag(cat_bad("session_id", lambda v: isinstance(v, str) and _SESSION_ID_RE.match(v)), "invalid session_id")
    ts = pd.to_numeric(df["ts_ms"], errors="coerce").to_numpy(float)
    flag(~np.isfinite(ts) | (ts < 0) | (ts != np.floor(ts)), "invalid ts_ms")
    flag(cat_bad("channel", lambda v: v in CHANNELS), "unknown channel")
    for key in ("detector", "detector_version"):
        flag(cat_bad(key, lambda v: isinstance(v, str) and len(v) > 0), f"empty {key}")
    conf = pd.to_numeric(df["confidence"], errors="coerce").to_numpy(float)
    flag(~np.isfinite(conf) | (conf < 0) | (conf > 1), "confidence out of [0,1]")
    flag(cat_bad("event_type", lambda v: v in EVENT_TYPES), "unregistered event_type")

    et_codes, et_cats = cat("event_type")
    ch_codes, ch_cats = cat("channel")
    lab_codes, lab_cats = cat("label")
    for k, name in enumerate(et_cats):
        et = EVENT_TYPES.get(name)
        if et is None:
            continue
        m = et_codes == k
        if not m.any():
            continue
        exp = ch_cats.get_loc(et.channel) if et.channel in ch_cats else -2
        flag(m & (ch_codes != exp), f"{name} on wrong channel")
        for f in et.fields:
            slot = SLOT_MAP[name][f.name]
            if f.kind == "label":
                good = np.array([v in (f.allowed or ()) for v in lab_cats] + [False], dtype=bool)
                flag(m & ~good[lab_codes], f"payload.{f.name} invalid")
                continue
            v = pd.to_numeric(df[slot], errors="coerce").to_numpy(float)[m]
            bad = ~np.isfinite(v)
            if f.kind == "bool":
                bad |= ~np.isin(v, (0.0, 1.0))
            elif f.kind == "int":
                bad |= np.isfinite(v) & (v != np.floor(v))
            if f.lo is not None:
                bad |= v < f.lo
            if f.hi is not None:
                bad |= v > f.hi
            full = np.zeros(n, dtype=bool)
            full[np.flatnonzero(m)[bad]] = True
            flag(full, f"payload.{f.name} invalid")
    blur = pd.to_numeric(df["frame_blur"], errors="coerce")
    flag(((blur < 0) | (blur > 1)).fillna(False).to_numpy(bool), "quality.frame_blur out of range")
    luma = pd.to_numeric(df["frame_luma"], errors="coerce")
    flag(((luma < 0) | (luma > 255)).fillna(False).to_numpy(bool), "quality.frame_luma out of range")

    bad_rows = reasons != ""
    rejected = df.loc[bad_rows].copy()
    rejected["reason"] = reasons[bad_rows]
    return df.loc[~bad_rows], rejected


def write_dead_letter(rejected: pd.DataFrame, path: Path) -> int:
    """Append rejected events (with reasons) to a JSONL dead-letter file. Returns rows written."""
    if rejected.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in rejected.to_dict(orient="records"):
            fh.write(json.dumps(rec, default=_json_default) + "\n")
    return len(rejected)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if np.isnan(f) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# --------------------------------------------------------------------------------------------
# risk_assessment.v1
# --------------------------------------------------------------------------------------------
def risk_assessment_errors(ra: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if ra.get("schema") != RISK_SCHEMA:
        errs.append(f"schema must be '{RISK_SCHEMA}'")
    risk = ra.get("overall_risk")
    if not isinstance(risk, (int, float)) or isinstance(risk, bool) or not 0 <= risk <= 1:
        errs.append("overall_risk must be in [0, 1]")
    if ra.get("recommendation") not in RECOMMENDATIONS:
        errs.append(f"recommendation must be one of {list(RECOMMENDATIONS)}")
    band = ra.get("confidence_band")
    if not (isinstance(band, list) and len(band) == 2 and all(isinstance(b, (int, float)) for b in band)):
        errs.append("confidence_band must be [low, high]")
    elif not (0 <= band[0] <= band[1] <= 1):
        errs.append("confidence_band must satisfy 0 <= low <= high <= 1")
    for key in ("channels_available", "channels_missing"):
        vals = ra.get(key)
        if not isinstance(vals, list) or any(v not in CHANNELS for v in vals):
            errs.append(f"{key} must list known channels")
    for i, fl in enumerate(ra.get("flags") or []):
        for k in ("flag_id", "type", "t_start_ms", "t_end_ms", "confidence", "explanation"):
            if k not in fl:
                errs.append(f"flags[{i}].{k} missing")
        if "t_start_ms" in fl and "t_end_ms" in fl and fl["t_end_ms"] < fl["t_start_ms"]:
            errs.append(f"flags[{i}] ends before it starts")
    for key in ("model_version", "feature_version", "session_id"):
        if not isinstance(ra.get(key), str) or not ra.get(key):
            errs.append(f"{key} must be a non-empty string")
    return errs


# --------------------------------------------------------------------------------------------
# JSON Schema generation (documentation artefacts in contracts/)
# --------------------------------------------------------------------------------------------
def _field_schema(f: PayloadField) -> dict[str, Any]:
    if f.kind == "bool":
        return {"type": "boolean"}
    if f.kind == "label":
        return {"type": "string", "enum": list(f.allowed or ())}
    s: dict[str, Any] = {"type": "integer" if f.kind == "int" else "number"}
    if f.lo is not None:
        s["minimum"] = f.lo
    if f.hi is not None:
        s["maximum"] = f.hi
    return s


def event_json_schema() -> dict[str, Any]:
    branches = []
    for et in EVENT_TYPES.values():
        branches.append(
            {
                "if": {"properties": {"event_type": {"const": et.name}}},
                "then": {
                    "properties": {
                        "channel": {"const": et.channel},
                        "payload": {
                            "type": "object",
                            "properties": {f.name: _field_schema(f) for f in et.fields},
                            "required": [f.name for f in et.fields],
                            "additionalProperties": False,
                        },
                    }
                },
            }
        )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://riskfusion.local/contracts/event.v1.schema.json",
        "title": "event.v1",
        "type": "object",
        "required": [
            "schema",
            "session_id",
            "ts_ms",
            "channel",
            "detector",
            "detector_version",
            "event_type",
            "payload",
            "confidence",
        ],
        "properties": {
            "schema": {"const": EVENT_SCHEMA},
            "session_id": {"type": "string", "pattern": "^[A-Za-z0-9_\\-]{3,64}$"},
            "ts_ms": {"type": "integer", "minimum": 0},
            "channel": {"enum": list(CHANNELS)},
            "detector": {"type": "string", "minLength": 1},
            "detector_version": {"type": "string", "minLength": 1},
            "event_type": {"enum": list(EVENT_TYPES)},
            "payload": {"type": "object"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "quality": {
                "type": "object",
                "properties": {
                    "frame_blur": {"type": "number", "minimum": 0, "maximum": 1},
                    "frame_luma": {"type": "number", "minimum": 0, "maximum": 255},
                    "usable": {"type": "boolean"},
                },
            },
        },
        "allOf": branches,
    }


def risk_assessment_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://riskfusion.local/contracts/risk_assessment.v1.schema.json",
        "title": "risk_assessment.v1",
        "type": "object",
        "required": [
            "schema",
            "session_id",
            "overall_risk",
            "calibrated",
            "recommendation",
            "confidence_band",
            "flags",
            "top_contributors",
            "channels_available",
            "channels_missing",
            "model_version",
            "feature_version",
        ],
        "properties": {
            "schema": {"const": RISK_SCHEMA},
            "session_id": {"type": "string"},
            "overall_risk": {"type": "number", "minimum": 0, "maximum": 1},
            "calibrated": {"type": "boolean"},
            "recommendation": {"enum": list(RECOMMENDATIONS)},
            "confidence_band": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": 2,
                "maxItems": 2,
            },
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["flag_id", "type", "t_start_ms", "t_end_ms", "confidence", "explanation"],
                    "properties": {
                        "flag_id": {"type": "string"},
                        "type": {"type": "string"},
                        "t_start_ms": {"type": "integer", "minimum": 0},
                        "t_end_ms": {"type": "integer", "minimum": 0},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "explanation": {"type": "string", "minLength": 1},
                        "evidence_ref": {"type": ["string", "null"]},
                    },
                },
            },
            "top_contributors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["feature", "shap", "direction"],
                    "properties": {
                        "feature": {"type": "string"},
                        "shap": {"type": "number"},
                        "direction": {"enum": ["increases_risk", "decreases_risk"]},
                    },
                },
            },
            "channels_available": {"type": "array", "items": {"enum": list(CHANNELS)}},
            "channels_missing": {"type": "array", "items": {"enum": list(CHANNELS)}},
            "model_version": {"type": "string"},
            "feature_version": {"type": "string"},
        },
    }


def write_schemas(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "event.v1.schema.json").write_text(json.dumps(event_json_schema(), indent=2) + "\n")
    (directory / "risk_assessment.v1.schema.json").write_text(
        json.dumps(risk_assessment_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--write" in sys.argv:
        write_schemas(Path(__file__).resolve().parents[2] / "contracts")
        print("schemas written to contracts/")
