from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from riskfusion.contracts import event_errors

from ..db import get_db
from ..deps import get_actor, get_storage, new_id
from ..models import (
    DataSplit,
    DeadLetterEvent,
    Event,
    ExamSession,
    Label,
    Participant,
    Recording,
    ScriptEpisode,
    SessionStatusChange,
)
from ..schemas import (
    ConfirmConsent,
    DeleteRequest,
    EpisodeMark,
    EpisodeOut,
    EventBatch,
    EventBatchResult,
    FailRequest,
    LabelOut,
    Page,
    RecordingOut,
    SessionCreate,
    SessionDetail,
    SessionOut,
    StartRecording,
    StatusChangeOut,
    StopRecording,
)
from ..services.audit import audit
from ..services.labels import write_mock_label
from ..services.lifecycle import transition
from ..services.purge import delete_session_data
from ..services.scripts import get_script, load_scripts
from ..services.signals import signal_timeline
from ..settings import get_settings
from ..storage import Storage, UploadTooLarge
from .participants import active_consent

router = APIRouter(tags=["sessions"])

MAGIC = {
    "webcam_av": ((b"\x1a\x45\xdf\xa3",), ("video/webm", "audio/webm", "video/x-matroska")),
    "enrollment_image": ((b"\xff\xd8\xff", b"\x89PNG"), ("image/jpeg", "image/png")),
}
TELEMETRY_CHANNELS = {"screen", "device", "behavioral"}


def _get(db: Session, sid: str) -> ExamSession:
    s = db.get(ExamSession, sid)
    if s is None:
        raise HTTPException(404, "Session not found.")
    return s


def _mock(s: ExamSession) -> None:
    if s.source != "MOCK":
        raise HTTPException(409, "Simulated sessions are generated data and cannot be recorded or edited.")


def _sealed(s: ExamSession, split: str | None) -> bool:
    """Simulated sealed-holdout sessions: labels and detector output are hidden everywhere (FR-21)."""
    return s.source == "SIMULATED" and split == "holdout"


def _session_out(s: ExamSession, code: str | None, violation: bool | None, split: str | None) -> SessionOut:
    if _sealed(s, split):
        violation = None
    return SessionOut.model_validate(s).model_copy(
        update={"participant_code": code, "violation_label": violation, "split": split}
    )


@router.get("/mock-scripts")
def mock_scripts() -> dict[str, Any]:
    return load_scripts()


@router.get("/sessions", response_model=Page)
def list_sessions(
    db: Session = Depends(get_db),
    source: str | None = Query(default=None, pattern="^(SIMULATED|MOCK)$"),
    status: str | None = None,
    profile: str | None = None,
    lighting: str | None = None,
    webcam_class: str | None = None,
    violation: bool | None = None,
    q: str | None = Query(default=None, max_length=64),
    include_deleted: bool = False,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Page:
    stmt = (
        select(ExamSession, Participant.code, Label.violation, DataSplit.split)
        .outerjoin(Participant, Participant.id == ExamSession.participant_id)
        .outerjoin(Label, Label.session_id == ExamSession.id)
        .outerjoin(DataSplit, DataSplit.session_id == ExamSession.id)
    )
    conds = []
    if source:
        conds.append(ExamSession.source == source)
    if status:
        conds.append(ExamSession.status == status)
    elif not include_deleted:
        conds.append(ExamSession.status != "DELETED")
    if profile:
        conds.append(ExamSession.behavior_profile == profile)
    if lighting:
        conds.append(ExamSession.lighting == lighting)
    if webcam_class:
        conds.append(ExamSession.webcam_class == webcam_class)
    if violation is not None:
        conds.append(Label.violation.is_(violation))
        conds.append(or_(DataSplit.split.is_(None), DataSplit.split != "holdout"))
    if q:
        like = f"%{q.strip()}%"
        conds.append(or_(ExamSession.id.ilike(like), Participant.code.ilike(like)))
    for c in conds:
        stmt = stmt.where(c)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.execute(stmt.order_by(ExamSession.created_at.desc(), ExamSession.id).limit(limit).offset(offset)).all()
    return Page(total=int(total), items=[_session_out(s, code, v, sp) for s, code, v, sp in rows])


@router.post("/sessions", response_model=SessionDetail, status_code=201)
def create_session(
    body: SessionCreate, db: Session = Depends(get_db), actor: str = Depends(get_actor)
) -> SessionDetail:
    try:
        script = get_script(body.script_id)
    except KeyError:
        raise HTTPException(422, "Unknown script.") from None
    p = db.get(Participant, body.participant_id)
    if p is None or p.status != "ACTIVE" or active_consent(p) is None:
        raise HTTPException(409, "The participant needs signed, active consent before a session can be created.")
    if script.get("requires_helper"):
        if not body.helper_participant_id or body.helper_participant_id == body.participant_id:
            raise HTTPException(422, "This script needs a second, consenting adult as helper.")
        h = db.get(Participant, body.helper_participant_id)
        if h is None or h.status != "ACTIVE" or active_consent(h) is None:
            raise HTTPException(409, "The helper also needs signed, active consent.")
    elif body.helper_participant_id:
        raise HTTPException(422, "This script does not use a helper.")
    s = ExamSession(
        id=new_id("sess"),
        source="MOCK",
        status="CREATED",
        participant_id=p.id,
        helper_participant_id=body.helper_participant_id,
        script_id=script["id"],
        script_version=str(load_scripts()["script_version"]),
        is_rehearsal=bool(script.get("rehearsal", False)),
        behavior_profile=script["profile"],
        lighting=body.lighting,
        webcam_class=body.webcam_class,
        room_noise=body.room_noise,
        eyewear=body.eyewear,
        head_covering=body.head_covering,
        notes=body.notes,
    )
    db.add(s)
    db.flush()
    db.add(SessionStatusChange(session_id=s.id, from_status=None, to_status="CREATED", actor=actor))
    audit(db, "session.create", "session", s.id, actor, {"script": script["id"], "participant": p.code})
    db.commit()
    return get_session(s.id, db)


@router.get("/sessions/{sid}", response_model=SessionDetail)
def get_session(sid: str, db: Session = Depends(get_db)) -> SessionDetail:
    s = _get(db, sid)
    code = db.scalar(select(Participant.code).where(Participant.id == s.participant_id)) if s.participant_id else None
    label = db.get(Label, s.id)
    split = db.scalar(select(DataSplit.split).where(DataSplit.session_id == s.id))
    history = db.scalars(
        select(SessionStatusChange).where(SessionStatusChange.session_id == s.id).order_by(SessionStatusChange.id)
    ).all()
    episodes = db.scalars(
        select(ScriptEpisode).where(ScriptEpisode.session_id == s.id).order_by(ScriptEpisode.episode_no)
    ).all()
    counts: dict[str, int] = {
        str(k): int(v)
        for k, v in (
            db.execute(
                select(Event.channel, func.count()).where(Event.session_id == s.id).group_by(Event.channel)
            ).all()
        )
    }
    dead = db.scalar(select(func.count()).select_from(DeadLetterEvent).where(DeadLetterEvent.session_id == s.id))
    if _sealed(s, split):
        label = None
    base = SessionDetail.model_validate(s)
    return base.model_copy(
        update={
            "participant_code": code,
            "violation_label": label.violation if label else None,
            "split": split,
            "history": [StatusChangeOut.model_validate(h) for h in history],
            "recordings": [RecordingOut.model_validate(r) for r in s.recordings],
            "episodes": [EpisodeOut.model_validate(e) for e in episodes],
            "label": LabelOut.model_validate(label) if label else None,
            "event_counts": {str(k): int(v) for k, v in counts.items()},
            "dead_letter_count": int(dead or 0),
        }
    )


@router.post("/sessions/{sid}/confirm-consent", response_model=SessionDetail)
def confirm_consent(
    sid: str, body: ConfirmConsent, db: Session = Depends(get_db), actor: str = Depends(get_actor)
) -> SessionDetail:
    s = _get(db, sid)
    _mock(s)
    p = db.get(Participant, s.participant_id)
    c = active_consent(p) if p else None
    if c is None:
        raise HTTPException(409, "The participant's consent is no longer active.")
    transition(db, s, "CONSENTED", actor, f"consent {c.id} ({c.consent_version}) confirmed")
    db.commit()
    return get_session(sid, db)


@router.post("/sessions/{sid}/start", response_model=SessionDetail)
def start_recording(
    sid: str, body: StartRecording, db: Session = Depends(get_db), actor: str = Depends(get_actor)
) -> SessionDetail:
    s = _get(db, sid)
    _mock(s)
    transition(db, s, "RECORDING", actor)
    s.started_at = datetime.now(timezone.utc)
    s.browser = body.browser
    script = get_script(s.script_id or "")
    db.query(ScriptEpisode).filter_by(session_id=s.id).delete()
    for i, ep in enumerate(script.get("episodes") or [], start=1):
        start_ms = int(ep["start_s"] * 1000)
        end_ms = -1 if ep["duration_s"] < 0 else start_ms + int(ep["duration_s"] * 1000)
        db.add(
            ScriptEpisode(
                session_id=s.id,
                episode_no=i,
                violation_type=ep["type"],
                instruction=ep["instruction"],
                scheduled_start_ms=start_ms,
                scheduled_end_ms=end_ms,
            )
        )
    db.commit()
    return get_session(sid, db)


@router.post("/sessions/{sid}/episodes/{no}", response_model=EpisodeOut)
def mark_episode(sid: str, no: int, body: EpisodeMark, db: Session = Depends(get_db)) -> ScriptEpisode:
    s = _get(db, sid)
    if s.status != "RECORDING":
        raise HTTPException(409, "Cues can only be marked while recording.")
    ep = db.query(ScriptEpisode).filter_by(session_id=sid, episode_no=no).one_or_none()
    if ep is None:
        raise HTTPException(404, "Cue not found.")
    if body.shown_at_ms is not None and ep.shown_at_ms is None:
        ep.shown_at_ms = body.shown_at_ms
    if body.completed_at_ms is not None:
        ep.completed_at_ms = body.completed_at_ms
    db.commit()
    return ep


@router.post("/sessions/{sid}/stop", response_model=SessionDetail)
def stop_recording(
    sid: str, body: StopRecording, db: Session = Depends(get_db), actor: str = Depends(get_actor)
) -> SessionDetail:
    s = _get(db, sid)
    _mock(s)
    transition(db, s, "RECORDED", actor)
    s.ended_at = datetime.now(timezone.utc)
    s.duration_s = body.duration_s
    db.commit()
    return get_session(sid, db)


@router.post("/sessions/{sid}/fail", response_model=SessionDetail)
def fail_session(
    sid: str, body: FailRequest, db: Session = Depends(get_db), actor: str = Depends(get_actor)
) -> SessionDetail:
    s = _get(db, sid)
    _mock(s)
    transition(db, s, "FAILED", actor, body.reason)
    db.commit()
    return get_session(sid, db)


@router.post("/sessions/{sid}/retry", response_model=SessionDetail)
def retry_session(sid: str, db: Session = Depends(get_db), actor: str = Depends(get_actor)) -> SessionDetail:
    """A failed recording can be retried from the consent-confirmed state."""
    s = _get(db, sid)
    _mock(s)
    if any(r.kind == "webcam_av" and r.status == "STORED" for r in s.recordings):
        raise HTTPException(409, "This session already has a stored recording.")
    transition(db, s, "CONSENTED", actor, "retry recording")
    s.started_at = None
    s.ended_at = None
    s.duration_s = None
    db.commit()
    return get_session(sid, db)


@router.post("/sessions/{sid}/recordings", response_model=RecordingOut, status_code=201)
def upload_recording(
    sid: str,
    file: UploadFile = File(...),
    kind: str = Form("webcam_av"),
    duration_s: float | None = Form(default=None, ge=0),
    width: int | None = Form(default=None, ge=1, le=10000),
    height: int | None = Form(default=None, ge=1, le=10000),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    actor: str = Depends(get_actor),
) -> Recording:
    s = _get(db, sid)
    _mock(s)
    if kind not in MAGIC:
        raise HTTPException(422, "Unknown recording kind.")
    if kind == "webcam_av" and s.status != "RECORDED":
        raise HTTPException(409, "Stop the recording before uploading it.")
    if kind == "enrollment_image" and s.status not in ("CONSENTED", "RECORDING"):
        raise HTTPException(409, "The enrolment photo is taken before or at the start of recording.")
    magics, mimes = MAGIC[kind]
    mime = (file.content_type or "").split(";")[0].strip().lower()
    if mime not in mimes:
        raise HTTPException(415, f"Unsupported file type {mime or 'unknown'}.")
    head = file.file.read(8)
    if not any(head.startswith(m) for m in magics):
        raise HTTPException(415, "The file content does not match its declared type.")
    file.file.seek(0)
    settings = get_settings()
    limit = (settings.max_upload_mb if kind == "webcam_av" else settings.max_image_mb) * 1024 * 1024
    ext = {"image/jpeg": "jpg", "image/png": "png"}.get(mime, "webm")
    rid = new_id("rec")
    key = f"sessions/{s.id}/{kind}/{rid}.{ext}"
    try:
        size, digest = storage.put_stream(key, file.file, limit)
    except UploadTooLarge as e:
        raise HTTPException(413, f"The recording is too large ({e}).") from None
    if size == 0:
        storage.delete(key)
        raise HTTPException(422, "The uploaded file is empty.")
    rec = Recording(
        id=rid,
        session_id=s.id,
        kind=kind,
        storage_backend=storage.backend,
        storage_key=key,
        filename=f"{s.id}-{kind}.{ext}",
        mime_type=mime,
        size_bytes=size,
        sha256=digest,
        duration_s=duration_s if kind == "webcam_av" else None,
        width=width,
        height=height,
        retention_until=datetime.now(timezone.utc) + timedelta(days=settings.retention_days),
    )
    db.add(rec)
    audit(db, "recording.upload", "session", s.id, actor, {"recording": rid, "kind": kind, "sha256": digest})
    if kind == "webcam_av":
        transition(db, s, "UPLOADED", actor, f"recording {rid} stored ({size} bytes)")
        write_mock_label(db, s, duration_s if duration_s is not None else (s.duration_s or 0.0))
    db.commit()
    return rec


@router.get("/recordings/{rid}/content")
def recording_content(rid: str, db: Session = Depends(get_db), storage: Storage = Depends(get_storage)) -> FileResponse:
    rec = db.get(Recording, rid)
    if rec is None or rec.status != "STORED" or not storage.exists(rec.storage_key):
        raise HTTPException(404, "This recording is not available (it may have been deleted).")
    return FileResponse(storage.path_for(rec.storage_key), media_type=rec.mime_type, filename=rec.filename)


@router.post("/sessions/{sid}/delete", response_model=SessionDetail)
def delete_session(
    sid: str,
    body: DeleteRequest,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    actor: str = Depends(get_actor),
) -> SessionDetail:
    s = _get(db, sid)
    _mock(s)
    if not body.confirm:
        raise HTTPException(422, "Confirm the deletion to continue.")
    if s.status == "DELETED":
        raise HTTPException(409, "This session is already deleted.")
    delete_session_data(db, storage, s, actor, body.reason)
    db.commit()
    return get_session(sid, db)


@router.post("/sessions/{sid}/events", response_model=EventBatchResult)
def ingest_events(sid: str, body: EventBatch, db: Session = Depends(get_db)) -> EventBatchResult:
    """Ingest browser telemetry (FR-5). Invalid events go to the dead-letter table with a reason (FR-8)."""
    s = _get(db, sid)
    _mock(s)
    if s.status not in ("RECORDING", "RECORDED", "UPLOADED"):
        raise HTTPException(409, "Telemetry is only accepted for a session that is being or has been recorded.")
    valid_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for i, raw in enumerate(body.events):
        ev = dict(raw)
        uid = ev.pop("event_uid", None)
        errs = event_errors(ev)
        if ev.get("session_id") != sid:
            errs.append("session_id does not match the URL")
        if ev.get("channel") not in TELEMETRY_CHANNELS:
            errs.append("only screen/device/behavioral telemetry may be posted by the browser")
        if not isinstance(uid, str) or not 8 <= len(uid) <= 64:
            errs.append("event_uid (8-64 chars) is required for idempotent delivery")
        if errs:
            reason = "; ".join(errs)
            db.add(DeadLetterEvent(session_id=sid, raw=raw, reason=reason))
            rejections.append({"index": i, "reason": reason})
            continue
        valid_rows.append(
            {
                "session_id": sid,
                "event_uid": uid,
                "schema_version": ev["schema"],
                "ts_ms": ev["ts_ms"],
                "channel": ev["channel"],
                "detector": ev["detector"],
                "detector_version": ev["detector_version"],
                "event_type": ev["event_type"],
                "confidence": float(ev["confidence"]),
                "payload": ev["payload"],
                "quality": ev.get("quality"),
                "origin": "telemetry",
            }
        )
    accepted = 0
    if valid_rows:
        stmt = (
            insert(Event)
            .values(valid_rows)
            .on_conflict_do_nothing(constraint="uq_events_session_uid")
            .returning(Event.id)
        )
        accepted = len(db.execute(stmt).all())
    db.commit()
    return EventBatchResult(
        accepted=accepted, duplicates=len(valid_rows) - accepted, rejected=len(rejections), rejections=rejections
    )


@router.get("/sessions/{sid}/events")
def list_events(
    sid: str,
    db: Session = Depends(get_db),
    channel: str | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
) -> list[dict[str, Any]]:
    _get(db, sid)
    stmt = select(Event).where(Event.session_id == sid)
    if channel:
        stmt = stmt.where(Event.channel == channel)
    rows = db.scalars(stmt.order_by(Event.ts_ms, Event.id).limit(limit)).all()
    return [
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
        for r in rows
    ]


@router.get("/sessions/{sid}/signal-timeline")
def session_signal_timeline(sid: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Detector-output timeline for a simulated session (from the Parquet analytical store)."""
    from ..models import DatasetVersion

    s = _get(db, sid)
    if s.source != "SIMULATED" or not s.dataset_version:
        raise HTTPException(409, "Signal timelines from the event store exist only for simulated sessions so far.")
    split = db.scalar(select(DataSplit.split).where(DataSplit.session_id == s.id))
    if _sealed(s, split):
        raise HTTPException(403, "This session is in the sealed holdout. Its labels and detector output stay hidden.")
    dv = db.get(DatasetVersion, s.dataset_version)
    events_dir = Path(dv.uri) / "events" if dv and dv.uri else None
    if events_dir is None or not events_dir.exists():
        raise HTTPException(404, "The event files for this dataset are not available on this machine.")
    return signal_timeline(events_dir, s.id, s.duration_s or 0.0)
