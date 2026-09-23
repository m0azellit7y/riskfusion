"""Session lifecycle. Every transition is validated and persisted to session_status_history."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..models import ExamSession, SessionStatusChange
from .audit import audit

TRANSITIONS: dict[str, frozenset[str]] = {
    "CREATED": frozenset({"CONSENTED", "FAILED", "DELETED"}),
    "CONSENTED": frozenset({"RECORDING", "FAILED", "DELETED"}),
    "RECORDING": frozenset({"RECORDED", "FAILED"}),
    "RECORDED": frozenset({"UPLOADED", "FAILED", "DELETED"}),
    "UPLOADED": frozenset({"PROCESSING", "DELETED"}),
    "PROCESSING": frozenset({"ANALYZING", "FAILED"}),
    "ANALYZING": frozenset({"COMPLETED", "FAILED"}),
    "COMPLETED": frozenset({"REVIEWED", "PROCESSING", "DELETED"}),
    "REVIEWED": frozenset({"PROCESSING", "DELETED"}),
    "FAILED": frozenset({"CONSENTED", "PROCESSING", "DELETED"}),
    "GENERATED": frozenset(),
    "DELETED": frozenset(),
}


class InvalidTransition(ValueError):
    pass


def transition(db: Session, sess: ExamSession, to: str, actor: str = "operator", note: str | None = None) -> None:
    frm = sess.status
    if to not in TRANSITIONS.get(frm, frozenset()):
        raise InvalidTransition(f"A session that is {frm.lower()} cannot move to {to.lower()}.")
    sess.status = to
    sess.updated_at = datetime.now(timezone.utc)
    db.add(SessionStatusChange(session_id=sess.id, from_status=frm, to_status=to, actor=actor, note=note))
    audit(db, "session.status", "session", sess.id, actor, {"from": frm, "to": to, "note": note})
