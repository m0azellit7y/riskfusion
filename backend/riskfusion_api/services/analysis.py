"""Analysis of sessions: detectors on mock recordings (Phase 2) -> features (Phase 3) -> risk assessment (4-5)."""

from __future__ import annotations

import json
import logging
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from riskfusion.contracts import FLAT_COLUMNS, event_to_flat
from riskfusion.features.engine import FEATURE_VERSION
from riskfusion.serving import load_model, score_frame

from ..models import Event, ExamSession, Prediction, ProcessingJob, Recording
from ..storage import Storage
from .audit import audit
from .lifecycle import transition

log = logging.getLogger("riskfusion.analysis")


def session_events_frame(db: Session, session_id: str) -> pd.DataFrame:
    rows = db.scalars(select(Event).where(Event.session_id == session_id).order_by(Event.ts_ms)).all()
    flat = [
        event_to_flat(
            {
                "schema": r.schema_version,
                "session_id": r.session_id,
                "ts_ms": r.ts_ms,
                "channel": r.channel,
                "detector": r.detector,
                "detector_version": r.detector_version,
                "event_type": r.event_type,
                "payload": r.payload,
                "confidence": r.confidence,
                "quality": r.quality,
            }
        )
        for r in rows
    ]
    return pd.DataFrame(flat, columns=list(FLAT_COLUMNS))


def store_prediction(db: Session, session_id: str, ra: dict[str, Any]) -> None:
    ra = {k: v for k, v in ra.items() if not k.startswith("_")}
    stmt = (
        insert(Prediction)
        .values(
            session_id=session_id,
            model_version=ra["model_version"],
            risk=ra["overall_risk"],
            recommendation=ra["recommendation"],
            band_lo=ra["confidence_band"][0],
            band_hi=ra["confidence_band"][1],
            n_flags=len(ra["flags"]),
            assessment=ra,
        )
        .on_conflict_do_update(
            index_elements=["session_id", "model_version"],
            set_={
                "risk": ra["overall_risk"],
                "recommendation": ra["recommendation"],
                "band_lo": ra["confidence_band"][0],
                "band_hi": ra["confidence_band"][1],
                "n_flags": len(ra["flags"]),
                "assessment": ra,
                "created_at": datetime.now(timezone.utc),
            },
        )
    )
    db.execute(stmt)


def _job(db: Session, job_id: str, **kw: Any) -> None:
    j = db.get(ProcessingJob, job_id)
    if j:
        for k, v in kw.items():
            setattr(j, k, v)
        db.commit()


def process_mock_session(session_factory: Any, storage: Storage, session_id: str, job_id: str, actor: str) -> None:
    """Background job. Runs detectors on the stored recording, replaces previous detector events, scores."""
    from riskfusion.extract.pipeline import extract_recording

    db: Session = session_factory()
    try:
        s = db.get(ExamSession, session_id)
        assert s is not None
        _job(db, job_id, status="RUNNING", stage="Running detectors", progress=0.02)
        video = next((r for r in s.recordings if r.kind == "webcam_av" and r.status == "STORED"), None)
        photo = next((r for r in reversed(s.recordings) if r.kind == "enrollment_image" and r.status == "STORED"), None)
        if video is None:
            raise RuntimeError("no stored recording")
        dur = float(s.duration_s or 1.0)

        def progress(t_s: float) -> None:
            _job(db, job_id, progress=min(0.85, 0.05 + 0.8 * t_s / max(dur, 1.0)))

        events, meta = extract_recording(
            session_id,
            storage.path_for(video.storage_key),
            storage.path_for(photo.storage_key) if photo else None,
            progress,
        )
        _job(db, job_id, stage="Storing detector events", progress=0.88)
        db.execute(delete(Event).where(Event.session_id == session_id, Event.origin == "detector"))
        for start in range(0, len(events), 2000):
            chunk = events[start : start + 2000]
            db.execute(
                insert(Event)
                .values(
                    [
                        {
                            "session_id": session_id,
                            "event_uid": e["event_uid"],
                            "schema_version": e["schema"],
                            "ts_ms": e["ts_ms"],
                            "channel": e["channel"],
                            "detector": e["detector"],
                            "detector_version": e["detector_version"],
                            "event_type": e["event_type"],
                            "confidence": e["confidence"],
                            "payload": e["payload"],
                            "quality": e.get("quality"),
                            "origin": "detector",
                        }
                        for e in chunk
                    ]
                )
                .on_conflict_do_nothing()
            )
        s = db.get(ExamSession, session_id)
        assert s is not None
        transition(db, s, "ANALYZING", actor, f"{len(events)} detector events ({meta['version']})")
        db.commit()
        _job(db, job_id, stage="Computing features and risk", progress=0.93)
        df = session_events_frame(db, session_id)
        ra, _ = score_frame(session_id, df, max(dur, meta["duration_s"]))
        store_prediction(db, session_id, ra)
        s = db.get(ExamSession, session_id)
        assert s is not None
        transition(db, s, "COMPLETED", actor, f"risk {ra['overall_risk']:.2f}, {ra['recommendation'].lower()}")
        audit(
            db,
            "session.analyzed",
            "session",
            session_id,
            actor,
            {"risk": ra["overall_risk"], "recommendation": ra["recommendation"], "extraction": meta},
        )
        db.commit()
        _job(
            db,
            job_id,
            status="SUCCEEDED",
            stage="Done",
            progress=1.0,
            details=meta,
            finished_at=datetime.now(timezone.utc),
        )
    except Exception as e:  # the job must always end in a recorded state
        db.rollback()
        log.error("processing %s failed: %s", session_id, traceback.format_exc())
        s = db.get(ExamSession, session_id)
        if s is not None and s.status in ("PROCESSING", "ANALYZING"):
            transition(db, s, "FAILED", actor, f"analysis failed: {type(e).__name__}: {e}")
            db.commit()
        _job(db, job_id, status="FAILED", message=f"{type(e).__name__}: {e}", finished_at=datetime.now(timezone.utc))
    finally:
        db.close()


def start_processing(session_factory: Any, storage: Storage, session_id: str, job_id: str, actor: str) -> None:
    threading.Thread(
        target=process_mock_session, args=(session_factory, storage, session_id, job_id, actor), daemon=True
    ).start()


def score_simulated(db: Session, data_root: Path, dataset_version: str) -> dict[str, Any]:
    """Score every non-holdout simulated session from the feature store and store the assessments."""
    model = load_model()
    from riskfusion.modeling.model import channels_from_features

    fdir = data_root / "features" / FEATURE_VERSION / dataset_version
    feats = pd.read_parquet(fdir / "features.parquet").set_index("session_id")
    flags = pd.read_parquet(fdir / "flags.parquet")
    fl = {k: g.drop(columns="session_id").to_dict(orient="records") for k, g in flags.groupby("session_id")}
    holdout = set(db.execute(text("SELECT session_id FROM data_splits WHERE split = 'holdout'")).scalars())
    n = 0
    for sid, row in feats.iterrows():
        if sid in holdout:
            continue  # sealed: no predictions are stored or displayed for holdout sessions
        f = row.to_dict()
        avail, missing = channels_from_features(f)
        ra = model.assess(str(sid), f, fl.get(sid, []), avail, missing)
        store_prediction(db, str(sid), ra)
        n += 1
        if n % 500 == 0:
            db.commit()
    db.commit()
    return {
        "dataset_version": dataset_version,
        "scored": n,
        "skipped_holdout": len(holdout & set(feats.index)),
        "model_version": model.meta["model_version"],
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


__all__ = ["Recording", "load_json", "process_mock_session", "score_simulated", "start_processing", "store_prediction"]


def mock_corpus_for_validation(db: Session) -> list[dict[str, Any]]:
    """Analysed, labelled, non-rehearsal mock sessions with their events (FR-3 / FR-6 input)."""
    from ..models import Label

    rows = db.execute(
        select(ExamSession, Label)
        .join(Label, Label.session_id == ExamSession.id)
        .where(
            ExamSession.source == "MOCK",
            ExamSession.is_rehearsal.is_(False),
            ExamSession.status.in_(("COMPLETED", "REVIEWED")),
        )
    ).all()
    return [
        {
            "session_id": s.id,
            "duration_s": float(s.duration_s or 1),
            "events": session_events_frame(db, s.id),
            "intervals": lab.intervals or [],
        }
        for s, lab in rows
    ]
