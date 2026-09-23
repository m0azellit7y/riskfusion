from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_actor, get_storage, new_id
from ..models import Consent, Demographics, ExamSession, Participant
from ..schemas import (
    ConsentCreate,
    ConsentOut,
    DemographicsIn,
    ParticipantCreate,
    ParticipantDetail,
    ParticipantOut,
    WithdrawRequest,
)
from ..services.audit import audit
from ..services.purge import withdraw_participant
from ..settings import get_settings
from ..storage import Storage

router = APIRouter(tags=["participants & consent"])
CONSENT_FORM = Path(__file__).resolve().parents[3] / "docs" / "consent" / "CONSENT_FORM.md"


def active_consent(p: Participant) -> Consent | None:
    live = [c for c in p.consents if c.withdrawn_at is None]
    return live[-1] if live else None


def _out(db: Session, p: Participant, cls: type[ParticipantOut] = ParticipantOut) -> ParticipantOut:
    c = active_consent(p)
    n = db.scalar(
        select(func.count())
        .select_from(ExamSession)
        .where(
            or_(ExamSession.participant_id == p.id, ExamSession.helper_participant_id == p.id),
            ExamSession.status != "DELETED",
        )
    )
    return cls.model_validate(p).model_copy(
        update={
            "consent_active": c is not None and p.status == "ACTIVE",
            "consent_version": c.consent_version if c else None,
            "session_count": int(n or 0),
        }
    )


@router.get("/consent/current")
def current_consent() -> dict[str, str]:
    return {"version": get_settings().consent_version, "text": CONSENT_FORM.read_text(encoding="utf-8")}


@router.get("/participants", response_model=list[ParticipantOut])
def list_participants(db: Session = Depends(get_db)) -> list[ParticipantOut]:
    ps = db.scalars(select(Participant).order_by(Participant.created_at.desc())).all()
    return [_out(db, p) for p in ps]


@router.post("/participants", response_model=ParticipantOut, status_code=201)
def create_participant(
    body: ParticipantCreate, db: Session = Depends(get_db), actor: str = Depends(get_actor)
) -> ParticipantOut:
    n = db.scalar(select(func.count()).select_from(Participant)) or 0
    code = f"P-{n + 1:03d}"
    while db.scalar(select(Participant).where(Participant.code == code)):
        n += 1
        code = f"P-{n + 1:03d}"
    p = Participant(id=new_id("par"), code=code, adult_confirmed=body.adult_confirmed, notes=body.notes)
    db.add(p)
    audit(db, "participant.create", "participant", p.id, actor, {"code": code})
    db.commit()
    return _out(db, p)


def _get(db: Session, pid: str) -> Participant:
    p = db.get(Participant, pid)
    if p is None:
        raise HTTPException(404, "Participant not found.")
    return p


@router.get("/participants/{pid}", response_model=ParticipantDetail)
def get_participant(pid: str, db: Session = Depends(get_db)) -> ParticipantOut:
    p = _get(db, pid)
    out = _out(db, p, ParticipantDetail)
    return out.model_copy(
        update={
            "consents": [ConsentOut.model_validate(c) for c in p.consents],
            "has_demographics": db.get(Demographics, p.id) is not None,
        }
    )


@router.post("/participants/{pid}/consent", response_model=ConsentOut, status_code=201)
def sign_consent(
    pid: str, body: ConsentCreate, db: Session = Depends(get_db), actor: str = Depends(get_actor)
) -> Consent:
    p = _get(db, pid)
    if p.status != "ACTIVE":
        raise HTTPException(409, "This participant has withdrawn. Register them again as a new participant.")
    if body.consent_version != get_settings().consent_version:
        raise HTTPException(409, "The consent form has changed. Reload the page and review the current version.")
    c = Consent(id=new_id("con"), participant_id=p.id, **body.model_dump())
    db.add(c)
    audit(db, "consent.sign", "participant", p.id, actor, {"consent_id": c.id, "version": c.consent_version})
    db.commit()
    return c


@router.post("/participants/{pid}/withdraw", response_model=ParticipantOut)
def withdraw(
    pid: str,
    body: WithdrawRequest,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    actor: str = Depends(get_actor),
) -> ParticipantOut:
    p = _get(db, pid)
    if p.status == "WITHDRAWN":
        raise HTTPException(409, "This participant has already withdrawn.")
    withdraw_participant(db, storage, p, body.reason, actor)
    db.commit()
    return _out(db, p)


@router.put("/participants/{pid}/demographics", status_code=204)
def set_demographics(
    pid: str, body: DemographicsIn, db: Session = Depends(get_db), actor: str = Depends(get_actor)
) -> None:
    p = _get(db, pid)
    if p.status != "ACTIVE":
        raise HTTPException(409, "This participant has withdrawn.")
    row = db.get(Demographics, p.id) or Demographics(participant_id=p.id)
    row.data = body.model_dump(exclude_none=True)
    db.add(row)
    audit(db, "demographics.set", "participant", p.id, actor)
    db.commit()
