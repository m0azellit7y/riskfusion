"""Typed, validated simulator configuration (FR-1: profiles and noise rates set by YAML)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROFILES = (
    "clean",
    "fidgety_clean",
    "poor_environment",
    "phone_user",
    "second_person",
    "impersonation",
    "tab_switcher",
    "remote_assisted",
    "note_reader",
    "mixed",
)
VIOLATION_TYPES = (
    "PHONE_USE",
    "SECOND_PERSON",
    "IMPERSONATION",
    "TAB_SWITCH",
    "REMOTE_ASSISTANCE",
    "NOTE_READING",
)
PROFILE_VIOLATIONS: dict[str, tuple[str, ...]] = {
    "phone_user": ("PHONE_USE",),
    "second_person": ("SECOND_PERSON",),
    "impersonation": ("IMPERSONATION",),
    "tab_switcher": ("TAB_SWITCH",),
    "remote_assisted": ("REMOTE_ASSISTANCE",),
    "note_reader": ("NOTE_READING",),
}
DETECTORS = (
    "face_present",
    "face_count_gt1",
    "identity_mismatch",
    "phone",
    "gaze_off_screen",
    "second_voice",
    "tab_hidden",
    "multi_monitor",
    "book_paper",
    "person_count_gt1",
    "liveness_spoof",
    "reach_off_frame",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProfileCfg(_Strict):
    weight: float = Field(ge=0)
    violation: bool


class EpisodeCfg(_Strict):
    count: tuple[int, int]
    duration_s: tuple[int, int]


class DurationCfg(_Strict):
    min: float = Field(gt=0)
    max: float
    mode: float

    @model_validator(mode="after")
    def _order(self) -> DurationCfg:
        if not self.min <= self.mode <= self.max:
            raise ValueError("duration_min requires min <= mode <= max")
        return self


class MovementCfg(_Strict):
    lognormal_mean: float
    lognormal_sigma: float = Field(ge=0)


class NuisanceCfg(_Strict):
    lighting: dict[str, float]
    webcam_class: dict[str, float]
    room_noise: dict[str, float]
    connection_stability: dict[str, float]
    eyewear_prob: float = Field(ge=0, le=1)
    head_covering_prob: float = Field(ge=0, le=1)
    duration_min: DurationCfg
    baseline_movement: MovementCfg

    @field_validator("lighting", "webcam_class", "room_noise", "connection_stability")
    @classmethod
    def _dist(cls, v: dict[str, float]) -> dict[str, float]:
        if abs(sum(v.values()) - 1.0) > 1e-6 or any(p < 0 for p in v.values()):
            raise ValueError("categorical distribution must be non-negative and sum to 1")
        return v


class DetectorCfg(_Strict):
    precision: float = Field(gt=0, le=1)
    recall: float = Field(ge=0, le=1)
    ref_prevalence: float = Field(gt=0, lt=1)
    burst_s: float = Field(ge=1)
    assumed: bool = False


class Effect(_Strict):
    fp: float = 1.0
    fn: float = 1.0


class ConnectionCfg(_Strict):
    unstable_gaps_per_hour: tuple[float, float]
    gap_duration_s: tuple[float, float]
    stable_gaps_per_hour: tuple[float, float]


class ConfoundersCfg(_Strict):
    """Persistent, realistic false-alarm sources in clean sessions (added in sim-v1.1; all off by default)."""

    phone_on_desk_prob: float = 0.0
    phone_on_desk_share: tuple[float, float] = (0.1, 0.5)
    background_talk_prob: float = 0.0
    background_talk_share: tuple[float, float] = (0.02, 0.15)
    writing_prob: float = 0.0
    writing_episodes: tuple[int, int] = (2, 8)
    writing_duration_s: tuple[int, int] = (10, 60)
    tab_notification_prob: float = 0.0
    tab_notification_count: tuple[int, int] = (1, 3)
    identity_drift_prob: float = 0.0
    identity_drift_share: tuple[float, float] = (0.05, 0.2)


class SubtleCfg(_Strict):
    """Violations that are hard to observe (sim-v1.1): short episodes and evasive behaviour."""

    prob: float = 0.0
    duration_scale: float = 1.0
    recall_scale: dict[str, float] = {}
    tab_switch_via_second_monitor: bool = False


class SimulatorConfig(_Strict):
    config_version: str
    seed: int
    n_sessions: int = Field(gt=0)
    shard_size: int = Field(gt=0)
    noise_source: str
    pessimism: float = Field(ge=1.0)
    recall_pessimism: float = Field(gt=0, le=1)
    profiles: dict[str, ProfileCfg]
    episodes: dict[str, EpisodeCfg]
    nuisance: NuisanceCfg
    detectors: dict[str, DetectorCfg]
    nuisance_effects: dict[str, dict[str, dict[str, Effect]]]
    connection: ConnectionCfg
    channel_outage: dict[str, float]
    confounders: ConfoundersCfg = ConfoundersCfg()
    subtle: SubtleCfg = SubtleCfg()

    @model_validator(mode="after")
    def _complete(self) -> SimulatorConfig:
        if set(self.profiles) != set(PROFILES):
            raise ValueError(f"profiles must be exactly {PROFILES}")
        if set(self.episodes) != set(VIOLATION_TYPES):
            raise ValueError(f"episodes must be exactly {VIOLATION_TYPES}")
        if set(self.detectors) != set(DETECTORS):
            raise ValueError(f"detectors must be exactly {DETECTORS}")
        for p, cfg in self.profiles.items():
            expect = p not in ("clean", "fidgety_clean", "poor_environment")
            if cfg.violation != expect:
                raise ValueError(f"profile {p} has violation={cfg.violation}, expected {expect}")
        return self

    # derived quantities -------------------------------------------------------------------
    def fp_rate(self, detector: str) -> float:
        d = self.detectors[detector]
        return self.pessimism * d.ref_prevalence * d.recall * (1 - d.precision) / d.precision

    def miss_prob(self, detector: str, recall_scale: float = 1.0) -> float:
        return 1.0 - self.detectors[detector].recall * self.recall_pessimism * recall_scale

    def expected_positive_rate(self) -> float:
        total = sum(p.weight for p in self.profiles.values())
        return sum(p.weight for p in self.profiles.values() if p.violation) / total

    def config_hash(self) -> str:
        dump = self.model_dump(mode="json")
        # keep v1 hashes stable: omit the v1.1 sections when they are at their (disabled) defaults
        if self.confounders == ConfoundersCfg():
            dump.pop("confounders")
        if self.subtle == SubtleCfg():
            dump.pop("subtle")
        blob = json.dumps(dump, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def load_config(path: str | Path, **overrides: object) -> SimulatorConfig:
    raw = yaml.safe_load(Path(path).read_text())
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return SimulatorConfig.model_validate(raw)
