"""Operational data model (PostgreSQL).

Design notes
* Consent records live in their own table and never reference media (FR-2: consent stored
  separately from media). Demographics live in a separate table that no feature or model code
  reads (ETH-4).
* ``labels`` follows SRS DR-4 exactly. Reviewer verdicts are stored separately in later phases so
  they can never overwrite scripted ground truth (see docs/SRS_AUDIT.md, A-3).
* Bulk simulated events live in Parquet (the analytical store); sessions, labels, splits and all
  mock-session data live here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

SESSION_SOURCES = ("SIMULATED", "MOCK")
SESSION_STATUSES = (
    "GENERATED",  # simulated sessions: produced by the simulator, never recorded
    "CREATED",
    "CONSENTED",
    "RECORDING",
    "RECORDED",
    "UPLOADED",
    "PROCESSING",
    "ANALYZING",
    "COMPLETED",
    "FAILED",
    "REVIEWED",
    "DELETED",
)


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONB, list[Any]: JSONB}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Participant(TimestampMixin, Base):
    __tablename__ = "participants"
    __table_args__ = (
        CheckConstraint("adult_confirmed = true", name="ck_participants_adult"),  # ETH-2
        CheckConstraint(_in("status", ("ACTIVE", "WITHDRAWN")), name="ck_participants_status"),
    )
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True, index=True)  # pseudonym, e.g. P-014
    adult_confirmed: Mapped[bool] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String(16), default="ACTIVE")
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)
    consents: Mapped[list[Consent]] = relationship(back_populates="participant", order_by="Consent.signed_at")


class Consent(Base):
    __tablename__ = "consents"
    __table_args__ = (
        CheckConstraint(
            "consent_recording AND consent_analysis AND consent_retention AND understands_withdrawal",
            name="ck_consents_all_items",
        ),
    )
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    participant_id: Mapped[str] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
    consent_version: Mapped[str] = mapped_column(String(40))
    signed_name: Mapped[str] = mapped_column(String(200))
    consent_recording: Mapped[bool] = mapped_column(Boolean)
    consent_analysis: Mapped[bool] = mapped_column(Boolean)
    consent_retention: Mapped[bool] = mapped_column(Boolean)
    understands_withdrawal: Mapped[bool] = mapped_column(Boolean)
    ethics_reference: Mapped[str | None] = mapped_column(String(120))  # ETH-7 approval reference
    witnessed_by: Mapped[str | None] = mapped_column(String(120))
    signed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    withdrawal_reason: Mapped[str | None] = mapped_column(Text)
    participant: Mapped[Participant] = relationship(back_populates="consents")


class Demographics(Base):
    """Voluntary, separately stored; used only for fairness evaluation, never as model input."""

    __tablename__ = "demographics"
    participant_id: Mapped[str] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))  # simulated | mock
    config_version: Mapped[str | None] = mapped_column(String(60))
    config_hash: Mapped[str | None] = mapped_column(String(64))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    seed: Mapped[int | None] = mapped_column(BigInteger)
    n_sessions: Mapped[int] = mapped_column(Integer)
    n_events: Mapped[int | None] = mapped_column(BigInteger)
    dead_letter_events: Mapped[int | None] = mapped_column(BigInteger)
    positive_rate: Mapped[float | None] = mapped_column(Float)
    noise_source: Mapped[str | None] = mapped_column(Text)
    generation_seconds: Mapped[float | None] = mapped_column(Float)
    uri: Mapped[str | None] = mapped_column(Text)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ExamSession(TimestampMixin, Base):
    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint(_in("source", SESSION_SOURCES), name="ck_sessions_source"),
        CheckConstraint(_in("status", SESSION_STATUSES), name="ck_sessions_status"),
        CheckConstraint("source <> 'MOCK' OR participant_id IS NOT NULL", name="ck_sessions_mock_participant"),
        Index("ix_sessions_source_status", "source", "status"),
        Index("ix_sessions_created", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(12))
    status: Mapped[str] = mapped_column(String(16))
    participant_id: Mapped[str | None] = mapped_column(ForeignKey("participants.id", ondelete="SET NULL"), index=True)
    helper_participant_id: Mapped[str | None] = mapped_column(ForeignKey("participants.id", ondelete="SET NULL"))
    dataset_version: Mapped[str | None] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    script_id: Mapped[str | None] = mapped_column(String(60))
    script_version: Mapped[str | None] = mapped_column(String(20))
    is_rehearsal: Mapped[bool] = mapped_column(Boolean, default=False)
    behavior_profile: Mapped[str | None] = mapped_column(String(40))
    lighting: Mapped[str | None] = mapped_column(String(16))
    webcam_class: Mapped[str | None] = mapped_column(String(16))
    room_noise: Mapped[str | None] = mapped_column(String(16))
    connection_stability: Mapped[str | None] = mapped_column(String(16))
    eyewear: Mapped[bool | None] = mapped_column(Boolean)
    head_covering: Mapped[bool | None] = mapped_column(Boolean)
    baseline_movement: Mapped[float | None] = mapped_column(Float)
    duration_s: Mapped[float | None] = mapped_column(Float)
    channels_outage: Mapped[list[Any] | None] = mapped_column(JSONB)
    browser: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)
    participant: Mapped[Participant | None] = relationship(foreign_keys=[participant_id])
    recordings: Mapped[list[Recording]] = relationship(back_populates="session", order_by="Recording.created_at")


class SessionStatusChange(Base):
    __tablename__ = "session_status_history"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    from_status: Mapped[str | None] = mapped_column(String(16))
    to_status: Mapped[str] = mapped_column(String(16))
    actor: Mapped[str] = mapped_column(String(80), default="system")
    note: Mapped[str | None] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Recording(Base):
    __tablename__ = "recordings"
    __table_args__ = (
        CheckConstraint(_in("kind", ("webcam_av", "enrollment_image")), name="ck_recordings_kind"),
        CheckConstraint(_in("status", ("STORED", "PURGED")), name="ck_recordings_status"),
        CheckConstraint("size_bytes >= 0", name="ck_recordings_size"),
    )
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    storage_backend: Mapped[str] = mapped_column(String(20), default="local")
    storage_key: Mapped[str] = mapped_column(Text)
    filename: Mapped[str] = mapped_column(String(200))
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    duration_s: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(10), default="STORED")
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    purged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    session: Mapped[ExamSession] = relationship(back_populates="recordings")


class ScriptEpisode(Base):
    """A scripted violation cue in a mock session: when it was scheduled and when it was shown."""

    __tablename__ = "script_episodes"
    __table_args__ = (UniqueConstraint("session_id", "episode_no"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    episode_no: Mapped[int] = mapped_column(Integer)
    violation_type: Mapped[str] = mapped_column(String(40))
    instruction: Mapped[str] = mapped_column(Text)
    scheduled_start_ms: Mapped[int] = mapped_column(BigInteger)
    scheduled_end_ms: Mapped[int] = mapped_column(BigInteger)
    shown_at_ms: Mapped[int | None] = mapped_column(BigInteger)
    completed_at_ms: Mapped[int | None] = mapped_column(BigInteger)


class Event(Base):
    """Operational event store for mock sessions (telemetry now; detector output from Phase 2)."""

    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("session_id", "event_uid", name="uq_events_session_uid"),  # idempotent ingest
        Index("ix_events_session_ts", "session_id", "ts_ms"),
        Index("ix_events_session_channel", "session_id", "channel"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    event_uid: Mapped[str] = mapped_column(String(64))
    schema_version: Mapped[str] = mapped_column(String(20), default="event.v1")
    ts_ms: Mapped[int] = mapped_column(BigInteger)
    channel: Mapped[str] = mapped_column(String(20))
    detector: Mapped[str] = mapped_column(String(80))
    detector_version: Mapped[str] = mapped_column(String(80))
    event_type: Mapped[str] = mapped_column(String(40))
    confidence: Mapped[float] = mapped_column(Float)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    quality: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    origin: Mapped[str] = mapped_column(String(20))  # telemetry | detector
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DeadLetterEvent(Base):
    __tablename__ = "dead_letter_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB)
    reason: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Label(Base):
    """Exactly SRS DR-4."""

    __tablename__ = "labels"
    __table_args__ = (
        CheckConstraint("source IN ('simulated','mock','reviewed')", name="ck_labels_source"),
        CheckConstraint("confidence IN ('certain','probable','uncertain')", name="ck_labels_confidence"),
    )
    session_id: Mapped[str] = mapped_column(Text, primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    violation: Mapped[bool] = mapped_column(Boolean, nullable=False)
    violation_types: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    intervals: Mapped[list[Any] | None] = mapped_column(JSONB)
    labeler: Mapped[str | None] = mapped_column(Text)
    labeled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    confidence: Mapped[str | None] = mapped_column(Text)


class DataSplit(Base):
    __tablename__ = "data_splits"
    __table_args__ = (CheckConstraint(_in("split", ("train", "validation", "calibration", "holdout"))),)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True)
    split_version: Mapped[str] = mapped_column(String(160), primary_key=True)
    split: Mapped[str] = mapped_column(String(12), index=True)


class HoldoutAccess(Base):
    __tablename__ = "holdout_accesses"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    split_version: Mapped[str] = mapped_column(String(160), index=True)
    access_no: Mapped[int] = mapped_column(Integer)
    purpose: Mapped[str] = mapped_column(Text)
    operator: Mapped[str] = mapped_column(String(80))
    git_head: Mapped[str | None] = mapped_column(String(64))
    accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_entity", "entity_type", "entity_id"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    actor: Mapped[str] = mapped_column(String(80))
    action: Mapped[str] = mapped_column(String(60))
    entity_type: Mapped[str] = mapped_column(String(40))
    entity_id: Mapped[str | None] = mapped_column(String(80))
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class Prediction(Base):
    """A risk_assessment.v1 produced by a model version for a session."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "recommendation IN ('NO_ACTION','ROUTINE_REVIEW','HUMAN_REVIEW','PRIORITY_REVIEW')",
            name="ck_predictions_recommendation",
        ),
        Index("ix_predictions_risk", "model_version", "risk"),
    )
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(60), primary_key=True)
    risk: Mapped[float] = mapped_column(Float)
    recommendation: Mapped[str] = mapped_column(String(20), index=True)
    band_lo: Mapped[float] = mapped_column(Float)
    band_hi: Mapped[float] = mapped_column(Float)
    n_flags: Mapped[int] = mapped_column(Integer, default=0)
    assessment: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReviewVerdict(Base):
    """A human reviewer's conclusion. Kept apart from `labels` and never used for training (SRS_AUDIT A-1)."""

    __tablename__ = "review_verdicts"
    __table_args__ = (
        CheckConstraint("verdict IN ('NO_CONCERN','CONCERN_CONFIRMED','INCONCLUSIVE')", name="ck_review_verdict"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    model_version: Mapped[str | None] = mapped_column(String(60))
    recommendation: Mapped[str | None] = mapped_column(String(20))
    risk: Mapped[float | None] = mapped_column(Float)
    verdict: Mapped[str] = mapped_column(String(20))
    note: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (CheckConstraint("status IN ('QUEUED','RUNNING','SUCCEEDED','FAILED')", name="ck_jobs_status"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(10))
    stage: Mapped[str | None] = mapped_column(String(40))
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
