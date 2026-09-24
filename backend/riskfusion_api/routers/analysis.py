"""Scoring API (FR-34/35), session analysis, reviewer reports (FR-37) and the reviewer workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from riskfusion.serving import DEFAULT_MODEL, load_model, render_report, score_events, score_frame

from ..db import _factory, get_db
from ..deps import get_actor, get_storage
from ..models import DataSplit, ExamSession, Label, Participant, Prediction, ProcessingJob, ReviewVerdict
from ..services.analysis import load_json, queue_analysis, session_events_frame
from ..services.audit import audit
from ..services.lifecycle import transition
from ..settings import get_settings
from ..storage import Storage

router = APIRouter(tags=["analysis"])
ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports" / "evaluation"


def _model_available() -> bool:
    return (DEFAULT_MODEL / "bundle.json").exists()


def _model() -> Any:
    if not _model_available():
        raise HTTPException(503, "No trained model is installed. Run `make pipeline` first.")
    return load_model()


class ScoreRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{3,64}$")
    events: list[dict[str, Any]] = Field(max_length=200_000)
    duration_s: float | None = Field(default=None, gt=0, le=6 * 3600)


@router.post("/score")
def score(body: ScoreRequest) -> dict[str, Any]:
    """FR-34: score a session from its event.v1 events. Invalid events are reported, never silently dropped."""
    _model()
    if not body.events:
        raise HTTPException(422, "At least one event is required.")
    return score_events(body.session_id, body.events, body.duration_s)


@router.get("/model-info")
def model_info() -> dict[str, Any]:
    """FR-35: version, training date and headline metrics of the deployed model."""
    m = _model()
    meta = m.meta
    return {
        "model_version": meta["model_version"],
        "feature_version": meta["feature_version"],
        "trained_at": meta["trained_at"],
        "dataset_version": meta["dataset_version"],
        "split_version": meta["split_version"],
        "calibration": meta["calibration"]["method"],
        "operating_point": meta["operating_point"],
        "tiers": meta["tiers"],
        "headline_metrics_validation": meta["headline_metrics_validation"],
        "holdout": [
            load_json(REPORTS / f"holdout_access_{i}.json")
            for i in (1, 2)
            if (REPORTS / f"holdout_access_{i}.json").exists()
        ],
        "n_features": len(m.features),
        "output": "review recommendation only — never a verdict",
    }


@router.get("/model/evaluation")
def model_evaluation() -> dict[str, Any]:
    _model()
    tr = load_json(DEFAULT_MODEL / "training_results.json") or {}
    runs = []
    idx = ROOT / "runs" / "index.jsonl"
    if idx.exists():
        runs = [json.loads(line) for line in idx.read_text().splitlines() if line.strip()]
    return {
        "training": {k: tr.get(k) for k in ("baseline", "primary", "temporal", "dataset_version", "split_version")},
        "evaluation": load_json(REPORTS / "evaluation.json"),
        "noise_robustness": load_json(REPORTS / "noise_robustness.json"),
        "drift": load_json(REPORTS / "drift.json"),
        "holdout": [load_json(REPORTS / f"holdout_access_{i}.json") for i in (1, 2)],
        "runs": runs,
    }


@router.post("/sessions/{sid}/process")
def process_session(
    sid: str, db: Session = Depends(get_db), storage: Storage = Depends(get_storage), actor: str = Depends(get_actor)
) -> dict[str, Any]:
    """Run the detectors on an uploaded mock recording and score it (background job)."""
    _model()
    s = db.get(ExamSession, sid)
    if s is None:
        raise HTTPException(404, "Session not found.")
    if s.source != "MOCK":
        raise HTTPException(409, "Simulated sessions are scored in bulk from the feature store.")
    running = db.scalar(
        select(ProcessingJob).where(ProcessingJob.session_id == sid, ProcessingJob.status.in_(("QUEUED", "RUNNING")))
    )
    if running:
        raise HTTPException(409, "This session is already being analysed.")
    job_id = queue_analysis(_factory(), db, storage, s, actor)
    return {"job_id": job_id, "status": "QUEUED"}


@router.get("/sessions/{sid}/job")
def session_job(sid: str, db: Session = Depends(get_db)) -> dict[str, Any] | None:
    j = db.scalar(
        select(ProcessingJob).where(ProcessingJob.session_id == sid).order_by(ProcessingJob.created_at.desc())
    )
    if j is None:
        return None
    return {
        "id": j.id,
        "status": j.status,
        "stage": j.stage,
        "progress": j.progress,
        "message": j.message,
        "details": j.details,
        "created_at": j.created_at,
        "finished_at": j.finished_at,
    }


def _sealed(db: Session, sid: str) -> bool:
    s = db.get(ExamSession, sid)
    split = db.scalar(select(DataSplit.split).where(DataSplit.session_id == sid))
    return bool(s and s.source == "SIMULATED" and split == "holdout")


@router.get("/sessions/{sid}/assessment")
def session_assessment(sid: str, db: Session = Depends(get_db)) -> dict[str, Any] | None:
    if _sealed(db, sid):
        return None
    p = db.scalar(select(Prediction).where(Prediction.session_id == sid).order_by(Prediction.created_at.desc()))
    if p is None:
        return None
    verdicts = db.scalars(select(ReviewVerdict).where(ReviewVerdict.session_id == sid).order_by(ReviewVerdict.id)).all()
    return {
        **p.assessment,
        "created_at": p.created_at,
        "reviews": [
            {"verdict": v.verdict, "note": v.note, "reviewer": v.reviewer, "created_at": v.created_at} for v in verdicts
        ],
    }


@router.get("/sessions/{sid}/report", response_class=HTMLResponse)
def session_report(sid: str, db: Session = Depends(get_db)) -> HTMLResponse:
    """FR-37: the reviewer's HTML report, recomputed from the stored events."""
    model = _model()
    s = db.get(ExamSession, sid)
    if s is None:
        raise HTTPException(404, "Session not found.")
    if _sealed(db, sid):
        raise HTTPException(403, "This session is in the sealed holdout.")
    if s.source == "MOCK":
        df = session_events_frame(db, sid)
        if df.empty:
            raise HTTPException(409, "This session has not been analysed yet.")
    else:
        import pyarrow.dataset as pads

        from ..models import DatasetVersion

        dv = db.get(DatasetVersion, s.dataset_version)
        path = Path(dv.uri) / "events" if dv and dv.uri else None
        if path is None or not path.exists():
            raise HTTPException(404, "The event files for this dataset are not available on this machine.")
        idx = __import__("pandas").read_parquet(path.parent / "session_index.parquet")
        shards = [str(path / f) for f in idx.loc[idx["session_id"] == sid, "shard"]]
        df = pads.dataset(shards, format="parquet").to_table(filter=pads.field("session_id") == sid).to_pandas()
    ra, sf = score_frame(sid, df, float(s.duration_s or 1.0), model)
    return HTMLResponse(render_report(ra, float(s.duration_s or len(sf.grid)), sf.grid, model, df, _report_meta(db, s)))


def _report_meta(db: Session, s: ExamSession) -> dict[str, Any]:
    """Session context for the reviewer report: who, when, which script, under which conditions."""
    from ..services.scripts import get_script

    def yn(v: bool | None) -> str:
        return "—" if v is None else ("yes" if v else "no")

    code = db.scalar(select(Participant.code).where(Participant.id == s.participant_id)) if s.participant_id else None
    try:
        title = get_script(s.script_id)["title"] if s.script_id else (s.behavior_profile or "").replace("_", " ")
    except KeyError:
        title = s.script_id or ""
    label = db.get(Label, s.id)
    ua = (s.browser or {}).get("userAgent", "") if s.browser else ""
    browser = next((b for b in ("Edg/", "Chrome/", "Firefox/", "Safari/") if b in ua), "")
    reviews = db.scalars(select(ReviewVerdict).where(ReviewVerdict.session_id == s.id).order_by(ReviewVerdict.id)).all()
    return {
        "participant_code": code or (s.id if s.source == "SIMULATED" else None),
        "source": "Recorded mock session" if s.source == "MOCK" else f"Simulated ({s.dataset_version})",
        "started_at": s.started_at.strftime("%Y-%m-%d %H:%M UTC") if s.started_at else "",
        "script_id": s.script_id,
        "script_title": title + (" (rehearsal)" if s.is_rehearsal else ""),
        "lighting": s.lighting or "—",
        "webcam_class": (s.webcam_class or "—").upper(),
        "room_noise": s.room_noise or "—",
        "eyewear": yn(s.eyewear),
        "head_covering": yn(s.head_covering),
        "browser": browser.rstrip("/").replace("Edg", "Edge") or "—",
        "script_intervals": (label.intervals if label and label.intervals else []),
        "recordings": [
            {
                "kind": "Video and audio" if r.kind == "webcam_av" else "Enrolment photo",
                "size": f"{r.size_bytes / 1e6:.1f} MB",
                "sha256": r.sha256,
            }
            for r in s.recordings
            if r.status == "STORED"
        ],
        "reviews": [
            {
                "at": v.created_at.strftime("%Y-%m-%d %H:%M"),
                "reviewer": v.reviewer,
                "verdict": v.verdict.replace("_", " ").lower(),
                "note": v.note,
            }
            for v in reviews
        ],
    }


class ReviewIn(BaseModel):
    verdict: Literal["NO_CONCERN", "CONCERN_CONFIRMED", "INCONCLUSIVE"]
    note: str | None = Field(default=None, max_length=4000)


@router.post("/sessions/{sid}/review", status_code=201)
def review_session(
    sid: str, body: ReviewIn, db: Session = Depends(get_db), actor: str = Depends(get_actor)
) -> dict[str, Any]:
    s = db.get(ExamSession, sid)
    if s is None:
        raise HTTPException(404, "Session not found.")
    if _sealed(db, sid):
        raise HTTPException(403, "This session is in the sealed holdout.")
    p = db.scalar(select(Prediction).where(Prediction.session_id == sid).order_by(Prediction.created_at.desc()))
    if p is None:
        raise HTTPException(409, "Only sessions with a risk assessment can be reviewed.")
    v = ReviewVerdict(
        session_id=sid,
        model_version=p.model_version,
        recommendation=p.recommendation,
        risk=p.risk,
        verdict=body.verdict,
        note=body.note,
        reviewer=actor,
    )
    db.add(v)
    if s.source == "MOCK" and s.status == "COMPLETED":
        transition(db, s, "REVIEWED", actor, body.verdict.lower().replace("_", " "))
    audit(db, "session.review", "session", sid, actor, {"verdict": body.verdict, "recommendation": p.recommendation})
    db.commit()
    return {"id": v.id, "verdict": v.verdict}


@router.get("/review-queue")
def review_queue(
    db: Session = Depends(get_db),
    tier: str = Query(default="HUMAN_REVIEW"),
    include_reviewed: bool = False,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Sessions ordered by calibrated risk — the ranked queue a reviewer works through."""
    order = ["ROUTINE_REVIEW", "HUMAN_REVIEW", "PRIORITY_REVIEW"]
    if tier not in order:
        raise HTTPException(422, "Unknown tier.")
    tiers = order[order.index(tier) :]
    reviewed = select(ReviewVerdict.session_id)
    stmt = (
        select(Prediction, ExamSession, Participant.code)
        .join(ExamSession, ExamSession.id == Prediction.session_id)
        .outerjoin(Participant, Participant.id == ExamSession.participant_id)
        .where(Prediction.recommendation.in_(tiers), ExamSession.status != "DELETED")
    )
    if not include_reviewed:
        stmt = stmt.where(Prediction.session_id.not_in(reviewed))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.execute(stmt.order_by(Prediction.risk.desc()).limit(limit).offset(offset)).all()
    counts: dict[str, int] = {
        str(k): int(v)
        for k, v in db.execute(
            select(Prediction.recommendation, func.count()).group_by(Prediction.recommendation)
        ).all()
    }
    return {
        "total": int(total),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "items": [
            {
                "session_id": p.session_id,
                "risk": p.risk,
                "recommendation": p.recommendation,
                "band": [p.band_lo, p.band_hi],
                "n_flags": p.n_flags,
                "source": s.source,
                "status": s.status,
                "participant_code": code,
                "lighting": s.lighting,
                "webcam_class": s.webcam_class,
                "duration_s": s.duration_s,
                "top_reason": (p.assessment.get("top_contributors") or [{}])[0].get("explanation"),
            }
            for p, s, code in rows
        ],
    }


@router.get("/reviews/summary")
def reviews_summary(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Overturn rate: share of reviewed sessions the reviewer found to be of no concern (SRS E3)."""
    rows = db.execute(select(ReviewVerdict.verdict, func.count()).group_by(ReviewVerdict.verdict)).all()
    c = {str(k): int(v) for k, v in rows}
    n = sum(c.values())
    agree = db.execute(
        select(func.count())
        .select_from(ReviewVerdict)
        .join(Label, Label.session_id == ReviewVerdict.session_id)
        .where(Label.violation.is_(True), ReviewVerdict.verdict == "CONCERN_CONFIRMED")
    ).scalar()
    return {
        "reviews": n,
        "by_verdict": c,
        "overturn_rate": (c.get("NO_CONCERN", 0) / n) if n else None,
        "confirmed_matching_ground_truth": int(agree or 0),
    }


@router.get("/pipeline/status")
def pipeline_status() -> dict[str, Any]:
    from riskfusion.extract.models import MODEL_DIR, MODELS

    return {
        "model_installed": _model_available(),
        "detectors_installed": all((MODEL_DIR / n).exists() for n in MODELS),
        "evaluation_available": (REPORTS / "evaluation.json").exists(),
        "data_root": str(get_settings().data_root),
    }


@router.post("/corpus/validate")
def validate_corpus(db: Session = Depends(get_db), actor: str = Depends(get_actor)) -> dict[str, Any]:
    """FR-3 + FR-6 on the current mock corpus. Writes configs/simulator/v2_measured.yaml once >= 20 sessions exist."""
    from riskfusion.validation.mock import validate_mock_corpus

    from ..services.analysis import mock_corpus_for_validation

    sessions = mock_corpus_for_validation(db)
    if not sessions:
        raise HTTPException(409, "No analysed, labelled mock sessions yet. Record and analyse sessions first.")
    report = validate_mock_corpus(
        sessions,
        get_settings().data_root.resolve(),
        "sim-v1.1-s20260923-n5000",
        ROOT / "configs" / "simulator" / "v1_1.yaml",
        REPORTS,
    )
    audit(db, "corpus.validate", "dataset", "mock", actor, {"sessions": len(sessions)})
    db.commit()
    return report


@router.get("/corpus/validation")
def corpus_validation() -> dict[str, Any] | None:
    return load_json(REPORTS / "mock_validation.json")
