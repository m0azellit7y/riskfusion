from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Lighting = Literal["bright", "normal", "dim"]
Webcam = Literal["hd", "sd", "low"]
Noise = Literal["quiet", "moderate", "noisy"]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---- participants & consent -------------------------------------------------------------------
class ParticipantCreate(BaseModel):
    adult_confirmed: bool = Field(description="Operator confirms the participant is 18 or older (ETH-2)")
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("adult_confirmed")
    @classmethod
    def _adult(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Participants must be 18 or older. Confirm the participant's age to continue.")
        return v


class ConsentCreate(BaseModel):
    consent_version: str
    signed_name: str = Field(min_length=2, max_length=200)
    consent_recording: bool
    consent_analysis: bool
    consent_retention: bool
    understands_withdrawal: bool
    ethics_reference: str | None = Field(default=None, max_length=120)
    witnessed_by: str | None = Field(default=None, max_length=120)

    @field_validator("consent_recording", "consent_analysis", "consent_retention", "understands_withdrawal")
    @classmethod
    def _all(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Every consent item must be agreed to before recording.")
        return v


class ConsentOut(ORM):
    id: str
    consent_version: str
    signed_name: str
    ethics_reference: str | None
    witnessed_by: str | None
    signed_at: datetime
    withdrawn_at: datetime | None
    withdrawal_reason: str | None


class WithdrawRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)
    confirm: bool

    @field_validator("confirm")
    @classmethod
    def _confirm(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Confirm the withdrawal to delete this participant's data.")
        return v


class DemographicsIn(BaseModel):
    """Voluntary. Stored separately; used only for fairness evaluation, never as a model input."""

    age_band: Literal["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "prefer_not_to_say"] | None = None
    gender: str | None = Field(default=None, max_length=60)
    other: dict[str, str] | None = None


class ParticipantOut(ORM):
    id: str
    code: str
    status: str
    adult_confirmed: bool
    created_at: datetime
    withdrawn_at: datetime | None
    notes: str | None
    consent_active: bool = False
    consent_version: str | None = None
    session_count: int = 0


class ParticipantDetail(ParticipantOut):
    consents: list[ConsentOut] = []
    has_demographics: bool = False


# ---- sessions ------------------------------------------------------------------------------------
class SessionCreate(BaseModel):
    participant_id: str
    script_id: str
    helper_participant_id: str | None = None
    lighting: Lighting
    webcam_class: Webcam
    room_noise: Noise
    eyewear: bool = False
    head_covering: bool = False
    notes: str | None = Field(default=None, max_length=2000)


class ConfirmConsent(BaseModel):
    confirmed: bool

    @field_validator("confirmed")
    @classmethod
    def _c(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Confirm that the participant has reviewed and agreed to the consent form today.")
        return v


class StartRecording(BaseModel):
    browser: dict[str, Any] | None = None


class StopRecording(BaseModel):
    duration_s: float = Field(ge=0, le=6 * 3600)


class EpisodeMark(BaseModel):
    shown_at_ms: int | None = Field(default=None, ge=0)
    completed_at_ms: int | None = Field(default=None, ge=0)


class FailRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=2000)


class DeleteRequest(BaseModel):
    confirm: bool
    reason: str = Field(default="deleted by operator", max_length=500)


class StatusChangeOut(ORM):
    from_status: str | None
    to_status: str
    actor: str
    note: str | None
    at: datetime


class RecordingOut(ORM):
    id: str
    kind: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    duration_s: float | None
    width: int | None
    height: int | None
    status: str
    created_at: datetime
    retention_until: datetime | None
    purged_at: datetime | None


class EpisodeOut(ORM):
    episode_no: int
    violation_type: str
    instruction: str
    scheduled_start_ms: int
    scheduled_end_ms: int
    shown_at_ms: int | None
    completed_at_ms: int | None


class LabelOut(ORM):
    source: str
    violation: bool
    violation_types: list[str] | None
    intervals: list[Any] | None
    labeler: str | None
    labeled_at: datetime | None
    confidence: str | None


class SessionOut(ORM):
    id: str
    source: str
    status: str
    participant_id: str | None
    participant_code: str | None = None
    helper_participant_id: str | None = None
    dataset_version: str | None
    script_id: str | None
    script_version: str | None = None
    is_rehearsal: bool
    behavior_profile: str | None
    lighting: str | None
    webcam_class: str | None
    room_noise: str | None
    connection_stability: str | None = None
    eyewear: bool | None
    head_covering: bool | None
    duration_s: float | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    violation_label: bool | None = None
    split: str | None = None


class SessionDetail(SessionOut):
    notes: str | None
    browser: dict[str, Any] | None
    channels_outage: list[Any] | None
    baseline_movement: float | None
    history: list[StatusChangeOut] = []
    recordings: list[RecordingOut] = []
    episodes: list[EpisodeOut] = []
    label: LabelOut | None = None
    event_counts: dict[str, int] = {}
    dead_letter_count: int = 0


class Page(BaseModel):
    total: int
    items: list[SessionOut]


class EventBatch(BaseModel):
    events: list[dict[str, Any]] = Field(max_length=5000)


class EventBatchResult(BaseModel):
    accepted: int
    duplicates: int
    rejected: int
    rejections: list[dict[str, Any]]
