"""Deletion: consent withdrawal (ETH-1), session deletion, and project-end media purge (ETH-3)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import delete
from sqlalchemy.orm import Session

from ..models import (
    Consent,
    DataSplit,
    Demographics,
    Event,
    ExamSession,
    Label,
    Participant,
    Recording,
    ScriptEpisode,
)
from ..storage import Storage
from .audit import audit
from .lifecycle import TRANSITIONS, transition


def purge_recording(db: Session, storage: Storage, rec: Recording) -> None:
    if rec.status == "STORED":
        storage.delete(rec.storage_key)
    rec.status = "PURGED"
    rec.purged_at = datetime.now(timezone.utc)


def delete_session_data(db: Session, storage: Storage, sess: ExamSession, actor: str, reason: str) -> None:
    """Delete media and all derived data of a session; the session row remains as a DELETED stub."""
    for rec in sess.recordings:
        purge_recording(db, storage, rec)
    db.execute(delete(Event).where(Event.session_id == sess.id))
    db.execute(delete(ScriptEpisode).where(ScriptEpisode.session_id == sess.id))
    db.execute(delete(Label).where(Label.session_id == sess.id))
    db.execute(delete(DataSplit).where(DataSplit.session_id == sess.id))
    sess.notes = None
    sess.browser = None
    if sess.status != "DELETED":
        if "DELETED" in TRANSITIONS[sess.status]:
            transition(db, sess, "DELETED", actor, reason)
        else:  # e.g. RECORDING/PROCESSING: force-close via FAILED first, keeping the history honest
            transition(db, sess, "FAILED", actor, "closed for deletion")
            transition(db, sess, "DELETED", actor, reason)


def withdraw_participant(db: Session, storage: Storage, p: Participant, reason: str | None, actor: str) -> int:
    """ETH-1: withdraw consent and delete the participant's data. Returns sessions affected."""
    now = datetime.now(timezone.utc)
    sessions = (
        db.query(ExamSession)
        .filter((ExamSession.participant_id == p.id) | (ExamSession.helper_participant_id == p.id))
        .all()
    )
    for s in sessions:
        delete_session_data(db, storage, s, actor, "consent withdrawn")
    for c in db.query(Consent).filter_by(participant_id=p.id).all():
        if c.withdrawn_at is None:
            c.withdrawn_at = now
            c.withdrawal_reason = reason
        c.signed_name = "[redacted on withdrawal]"
    db.execute(delete(Demographics).where(Demographics.participant_id == p.id))
    p.status = "WITHDRAWN"
    p.withdrawn_at = now
    p.notes = None
    audit(db, "participant.withdraw", "participant", p.id, actor, {"sessions_deleted": len(sessions)})
    return len(sessions)
