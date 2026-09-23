from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select, text
from sqlalchemy.orm import Session

import riskfusion
from riskfusion.contracts import CHANNELS, EVENT_SCHEMA, RISK_SCHEMA

from ..db import get_db
from ..deps import get_storage
from ..models import (
    AuditLog,
    DatasetVersion,
    DeadLetterEvent,
    Event,
    ExamSession,
    Label,
    Participant,
    Recording,
)
from ..services.scripts import load_scripts
from ..settings import get_settings
from ..storage import Storage

router = APIRouter(tags=["system"])

FR2_TARGET_SESSIONS = 60
FR2_LIGHTING = ("bright", "normal", "dim")
FR2_WEBCAMS = ("hd", "sd", "low")


@router.get("/health")
def health(db: Session = Depends(get_db), storage: Storage = Depends(get_storage)) -> dict[str, Any]:
    checks: dict[str, str] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:  # pragma: no cover - exercised only when the DB is down
        checks["database"] = f"error: {type(e).__name__}"
    try:
        probe = storage.path_for(".health")
        probe.write_text("ok")
        probe.unlink()
        checks["storage"] = "ok"
    except Exception as e:  # pragma: no cover
        checks["storage"] = f"error: {type(e).__name__}"
    ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ok" if ok else "degraded",
        "version": riskfusion.__version__,
        "contracts": [EVENT_SCHEMA, RISK_SCHEMA],
        "checks": checks,
        "model": None,  # no trained model yet: models arrive in Phase 4
        "time": datetime.now(timezone.utc).isoformat(),
    }


def _mock_eligible(db: Session) -> Any:
    """Corpus-eligible mock sessions: uploaded or later, not rehearsal, >= 20 min, not deleted."""
    min_dur = float(load_scripts()["min_duration_s"])
    return select(ExamSession).where(
        ExamSession.source == "MOCK",
        ExamSession.is_rehearsal.is_(False),
        ExamSession.status.in_(("UPLOADED", "PROCESSING", "ANALYZING", "COMPLETED", "REVIEWED")),
        ExamSession.duration_s >= min_dur,
    )


@router.get("/corpus/mock-coverage")
def mock_coverage(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Progress toward FR-2: >= 60 consented sessions, >= 3 lighting and >= 2 webcam conditions."""
    sub = _mock_eligible(db).subquery()
    grid = db.execute(
        select(sub.c.lighting, sub.c.webcam_class, func.count()).group_by(sub.c.lighting, sub.c.webcam_class)
    ).all()
    cells = {f"{lt}|{wc}": int(n) for lt, wc, n in grid}
    total = sum(cells.values())
    lighting_seen = sorted({k.split("|")[0] for k in cells})
    webcam_seen = sorted({k.split("|")[1] for k in cells})
    by_script: dict[str, int] = {
        str(k): int(v) for k, v in (db.execute(select(sub.c.script_id, func.count()).group_by(sub.c.script_id)).all())
    }
    participants = db.scalar(select(func.count()).select_from(Participant).where(Participant.status == "ACTIVE"))
    distinct_participants = db.scalar(select(func.count(func.distinct(sub.c.participant_id))))
    return {
        "eligible_sessions": total,
        "target_sessions": FR2_TARGET_SESSIONS,
        "lighting_conditions": lighting_seen,
        "webcam_conditions": webcam_seen,
        "target_lighting_conditions": 3,
        "target_webcam_conditions": 2,
        "grid": {lt: {wc: cells.get(f"{lt}|{wc}", 0) for wc in FR2_WEBCAMS} for lt in FR2_LIGHTING},
        "by_script": {str(k): int(v) for k, v in by_script.items()},
        "active_participants": int(participants or 0),
        "participants_recorded": int(distinct_participants or 0),
        "met": total >= FR2_TARGET_SESSIONS and len(lighting_seen) >= 3 and len(webcam_seen) >= 2,
    }


@router.get("/overview")
def overview(db: Session = Depends(get_db)) -> dict[str, Any]:
    by_source_status = db.execute(
        select(ExamSession.source, ExamSession.status, func.count()).group_by(ExamSession.source, ExamSession.status)
    ).all()
    counts: dict[str, dict[str, int]] = {}
    for src, st, n in by_source_status:
        counts.setdefault(src, {})[st] = int(n)
    recent = db.execute(
        select(
            ExamSession.id,
            ExamSession.status,
            ExamSession.script_id,
            ExamSession.created_at,
            Participant.code,
            ExamSession.is_rehearsal,
            ExamSession.duration_s,
        )
        .outerjoin(Participant, Participant.id == ExamSession.participant_id)
        .where(ExamSession.source == "MOCK")
        .order_by(ExamSession.created_at.desc())
        .limit(6)
    ).all()
    activity = db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(8)).all()
    datasets = db.scalars(select(DatasetVersion).order_by(DatasetVersion.registered_at.desc())).all()
    needs_upload = counts.get("MOCK", {}).get("RECORDED", 0)
    return {
        "counts": counts,
        "mock_needs_upload": needs_upload,
        "mock_coverage": mock_coverage(db),
        "recent_mock_sessions": [
            {
                "id": r.id,
                "status": r.status,
                "script_id": r.script_id,
                "created_at": r.created_at,
                "participant_code": r.code,
                "is_rehearsal": r.is_rehearsal,
                "duration_s": r.duration_s,
            }
            for r in recent
        ],
        "activity": [
            {
                "at": a.at,
                "actor": a.actor,
                "action": a.action,
                "entity_type": a.entity_type,
                "entity_id": a.entity_id,
                "details": a.details,
            }
            for a in activity
        ],
        "datasets": [
            {"id": d.id, "n_sessions": d.n_sessions, "positive_rate": d.positive_rate, "noise_source": d.noise_source}
            for d in datasets
        ],
        "model": None,
    }


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    out = []
    root = get_settings().data_root.resolve()
    for d in db.scalars(select(DatasetVersion).order_by(DatasetVersion.registered_at.desc())).all():
        split_dir = root / "splits" / d.id
        split = None
        if (split_dir / "manifest.json").exists():
            sm = json.loads((split_dir / "manifest.json").read_text())
            log = [json.loads(x) for x in (split_dir / "HOLDOUT_ACCESS_LOG.jsonl").read_text().splitlines() if x]
            split = {
                "split_version": sm["split_version"],
                "counts": sm["counts"],
                "positive_rate": sm["positive_rate"],
                "holdout_ids_sha256": sm["holdout_ids_sha256"],
                "holdout_accesses": log,
                "max_holdout_accesses": sm["max_holdout_accesses"],
                "created_at": sm["created_at"],
            }
        out.append(
            {
                "id": d.id,
                "kind": d.kind,
                "config_version": d.config_version,
                "config_hash": d.config_hash,
                "content_hash": d.content_hash,
                "seed": d.seed,
                "n_sessions": d.n_sessions,
                "n_events": d.n_events,
                "dead_letter_events": d.dead_letter_events,
                "positive_rate": d.positive_rate,
                "noise_source": d.noise_source,
                "generation_seconds": d.generation_seconds,
                "profile_counts": d.manifest.get("profile_counts"),
                "registered_at": d.registered_at,
                "splits": split,
            }
        )
    return out


@router.get("/datasets/{dataset_id}/breakdown")
def dataset_breakdown(dataset_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Session counts and positive rate per nuisance factor — the slices later used for FR-27."""
    if db.get(DatasetVersion, dataset_id) is None:
        raise HTTPException(404, "Dataset not found.")
    out: dict[str, Any] = {}
    for col in (
        "lighting",
        "webcam_class",
        "room_noise",
        "connection_stability",
        "eyewear",
        "head_covering",
        "behavior_profile",
    ):
        c = getattr(ExamSession, col)
        rows = db.execute(
            select(c, func.count(), func.avg(case((Label.violation.is_(True), 1.0), else_=0.0)))
            .join(Label, Label.session_id == ExamSession.id)
            .where(ExamSession.dataset_version == dataset_id)
            .group_by(c)
            .order_by(c)
        ).all()
        out[col] = [{"value": str(v), "sessions": int(n), "positive_rate": float(p or 0)} for v, n, p in rows]
    dur = db.execute(
        select(
            func.min(ExamSession.duration_s), func.avg(ExamSession.duration_s), func.max(ExamSession.duration_s)
        ).where(ExamSession.dataset_version == dataset_id)
    ).one()
    out["duration_s"] = {"min": dur[0], "mean": dur[1], "max": dur[2]}
    return out


@router.get("/system/status")
def system_status(db: Session = Depends(get_db), storage: Storage = Depends(get_storage)) -> dict[str, Any]:
    rev = db.execute(text("SELECT version_num FROM alembic_version")).scalar()
    tables = {
        "participants": Participant,
        "sessions": ExamSession,
        "recordings": Recording,
        "events": Event,
        "labels": Label,
        "dead_letter_events": DeadLetterEvent,
        "audit_logs": AuditLog,
    }
    return {
        "health": health(db, storage),
        "schema_revision": rev,
        "row_counts": {k: int(db.scalar(select(func.count()).select_from(m)) or 0) for k, m in tables.items()},
        "storage": {
            "backend": storage.backend,
            "used_bytes": storage.usage_bytes(),
            "stored_recordings": int(
                db.scalar(select(func.count()).select_from(Recording).where(Recording.status == "STORED")) or 0
            ),
        },
        "channels": list(CHANNELS),
        "consent_version": get_settings().consent_version,
        "pipeline": [
            {"stage": "Session simulator", "state": "available"},
            {"stage": "Consent and mock recording", "state": "available"},
            {"stage": "Browser telemetry capture", "state": "available"},
            {"stage": "Signal extraction (detectors)", "state": "not_built"},
            {"stage": "Feature engineering", "state": "not_built"},
            {"stage": "Risk models and calibration", "state": "not_built"},
            {"stage": "Explanations and flags", "state": "not_built"},
            {"stage": "Reports and drift monitoring", "state": "not_built"},
        ],
    }


@router.get("/audit")
def audit_log(
    db: Session = Depends(get_db),
    entity_type: str | None = None,
    entity_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    stmt = select(AuditLog)
    if entity_type:
        stmt = stmt.where(AuditLog.entity_type == entity_type)
    if entity_id:
        stmt = stmt.where(AuditLog.entity_id == entity_id)
    rows = db.scalars(stmt.order_by(AuditLog.id.desc()).limit(limit)).all()
    return [
        {
            "at": a.at,
            "actor": a.actor,
            "action": a.action,
            "entity_type": a.entity_type,
            "entity_id": a.entity_id,
            "details": a.details,
        }
        for a in rows
    ]
