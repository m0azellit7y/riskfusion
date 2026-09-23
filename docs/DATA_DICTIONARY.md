# Data dictionary (Phase 1)

Features are added in Phase 3. The authoritative definitions are the contract registry
(`src/riskfusion/contracts.py`) and the ORM models (`backend/riskfusion_api/models.py`). This file
explains them.

## event.v1 (contract)

| Field | Type | Meaning |
|---|---|---|
| schema | "event.v1" | Contract version |
| session_id | string `[A-Za-z0-9_-]{3,64}` | Session the event belongs to |
| ts_ms | int ≥ 0 | Milliseconds from the start of the recording |
| channel | enum (11) | presence, identity, liveness, attention, environment, pose, audio_voice, audio_event, screen, device, behavioral |
| detector, detector_version | string | Producing detector and version |
| event_type | enum | Registered per channel; each channel also has `<CHANNEL>_UNKNOWN` with a `reason` |
| payload | object | Fields fixed per event type (see the table below) |
| confidence | float 0–1 | Detector confidence |
| quality | object, optional | frame_blur 0–1, frame_luma 0–255, usable bool |

### Payloads

| Event type | Payload |
|---|---|
| FACE_OBSERVATION | face_count, bbox_area_ratio, center_offset_x/y |
| IDENTITY_CHECK | cosine_sim, verified, face_quality |
| LIVENESS_CHECK | pad_score, pad_pass |
| HEAD_GAZE | yaw, pitch, roll (degrees), gaze_on_screen_prob |
| PERSON_COUNT | n_persons |
| OBJECT_DETECTED | label (cell_phone, book, paper…), bbox_x/y/w/h |
| BODY_POSE | hands_visible_count, reach_off_frame, torso_rotation |
| VOICE_ACTIVITY | vad_active, speaker_is_candidate, n_speakers, foreign_speech |
| AUDIO_EVENT | label (keyboard_burst, phone_ring, paper_rustle, door, other) |
| TAB_VISIBILITY | hidden |
| FULLSCREEN_CHANGE | active |
| PASTE | length (characters; content never captured) |
| MONITOR_COUNT | count |
| INPUT_ACTIVITY | keystrokes, window_s (counts only; keys never captured) |
| *_UNKNOWN | reason: no_frames, low_light, occluded, device_unavailable, connection_lost, detector_error, permission_denied, not_observable |

### Parquet (flat) layout

Columns: `session_id, ts_ms, channel, detector, detector_version, event_type, confidence, p0–p3, label, usable, frame_blur, frame_luma`.
- Numeric payload fields map in order onto `p0`–`p3`.
- The categorical field maps onto `label`.
- `flat_to_event` reconstructs the canonical event.
- Strings are dictionary-encoded, and files use zstd compression.

## Simulated dataset files (`data/synthetic/<version>/`)

- `events/shard-NNNN.parquet`: flat events, 250 sessions per shard.
- `session_index.parquet`: maps each session to its shard.
- `sessions.parquet`: one row per session.
  - Identity and label: session_id, behavior_profile, violation, violation_types, duration_s.
  - Nuisance factors: lighting, webcam_class, room_noise, connection_stability, eyewear, head_covering, baseline_movement.
  - channels_outage, n_events, dataset_version.
- `labels.parquet`: in DR-4 shape.
- `manifest.json`: counts, positive rate, config hash, content hash, seed, timing, platform.

## Operational tables (PostgreSQL)

| Table | Purpose | Notes |
|---|---|---|
| participants | Pseudonymous volunteers (P-001…) | `adult_confirmed` CHECK true; no name, no age |
| consents | Signed consent per form version | Name lives only here; all four items CHECKed; redacted on withdrawal |
| demographics | Voluntary details (JSON) | Separate table; fairness evaluation only (ETH-4) |
| dataset_versions | Registered simulated runs | Manifest and hashes |
| sessions | Simulated (GENERATED) and mock sessions | Status lifecycle; nuisance columns; helper participant; script id/version; rehearsal flag |
| session_status_history | Every status change | Actor, note, timestamp |
| recordings | Video/audio and enrolment photos | SHA-256, size, retention_until, STORED/PURGED |
| script_episodes | Scheduled cues and when they were shown | Source of mock interval labels |
| events | Mock-session events (telemetry now, detectors later) | Unique (session_id, event_uid) for idempotent delivery |
| dead_letter_events | Rejected events with reason | FR-8 |
| labels | DR-4 exactly | source simulated/mock; confidence certain/uncertain |
| data_splits | Split per session and split version | |
| holdout_accesses | Reserved for DB mirroring of the committed log | |
| audit_logs | Consent, status, upload, withdrawal, deletion, registration | Actor from the operator name |

## Session statuses

`CREATED → CONSENTED → RECORDING → RECORDED → UPLOADED → PROCESSING → ANALYZING → COMPLETED → REVIEWED`, plus `FAILED`
(retry returns to CONSENTED), `DELETED` (terminal, data purged) and `GENERATED` (simulated). Allowed transitions:
`backend/riskfusion_api/services/lifecycle.py`.
