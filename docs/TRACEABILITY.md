# Traceability matrix

Status reflects what exists in this repository and was verified, as of the end of Phase 1.

**Status key**
- **Done**: implemented and tested.
- **Partial**: some parts implemented; the entry says which.
- **Needs you**: the software is ready, but completion needs human action such as recruiting, recording or ethics approval.
- **Planned**: a later phase.

Evidence refers to code (`path`), tests (`tests/...`) or a measured result.

| ID | Requirement | Phase | Status | Evidence / notes |
|---|---|---|---|---|
| FR-1 | Configurable session simulator (DR-1) | 1 | Done | `src/riskfusion/simulator/`, `configs/simulator/v1.yaml`; 5,000 sessions / 68.9M events in 125 s on 1 vCPU; seeded, content-hashed; `tests/unit/test_simulator.py` |
| FR-2 | Mock recording pipeline with consent management | 1 | Done (software) / Needs you (60 sessions) | Consent, recording, upload, labels, coverage tracker; browser E2E `tests/e2e/record_flow.py`. The 60 recordings require volunteers and ethics approval |
| FR-3 | Simulator validation against mock data | 2 | Planned | Needs FR-2 corpus + FR-4 |
| FR-4 | Pretrained detectors on mock recordings (≥4 channels) | 2 | Planned | ffmpeg and OpenCV are available; enrolment photo already captured (A-12) |
| FR-5 | Browser telemetry from mock sessions | 1–2 | Done (capture) | `frontend/src/lib/telemetry.ts`: tab visibility, full screen, paste length, monitor count, keystroke counts; stored as event.v1 |
| FR-6 | Measure detector error on mock data | 2 | Planned | Protocol in SRS_AUDIT A-19; output replaces `v1.yaml` noise rates |
| FR-7 | Explicit UNKNOWN state on every detector | 1 | Done | `*_UNKNOWN` for all 11 channels in the contract; simulator emits them for gaps, low light, outages; unsupported browsers emit `DEVICE_UNKNOWN` |
| FR-8 | Validate all events; dead-letter with reason | 1 | Done | `contracts.py` (dict and vectorised paths agree, tested); API → `dead_letter_events`; simulator → `data/deadletter/*.jsonl` |
| FR-9 | 1-second grid with explicit gaps | 3 | Planned | |
| FR-10 | Out-of-order events within 30 s | 3 | Planned | Late >30 s → dead letter (A-26) |
| FR-11 | Per-second base features | 3 | Planned | |
| FR-12 | Windowed aggregates | 3 | Planned | Session reduction per A-10 |
| FR-13 | Per-session baseline normalisation | 3 | Planned | Scope limited per A-9 |
| FR-14 | Cross-channel interaction features | 3 | Planned | |
| FR-15 | Versioned Parquet feature store with schema hash | 3 | Partial | Event store is already versioned Parquet with content/config hashes |
| FR-16 | Rule baseline | 4 | Planned | |
| FR-17 | Gradient-boosted primary model | 4 | Planned | |
| FR-18 | Temporal sequence model | 5 | Planned | |
| FR-19 | Class-imbalance handling | 4 | Planned | |
| FR-20 | Calibration on dedicated split | 4 | Partial | Calibration split exists (601 sessions) |
| FR-21 | Validation-only selection; holdout accessed exactly twice | 1–7 | Done (mechanism) | `data/splits.py` guard, committed IDs + hash + access log; API/UI hide holdout labels; `tests/unit/test_splits.py` |
| FR-22 | Ablation study | 5 | Planned | |
| FR-23 | SHAP per prediction | 5 | Planned | |
| FR-24 | Plain-language explanations | 5 | Planned | |
| FR-25 | Time-bounded flags | 5 | Planned | A-11 |
| FR-26 | Global interpretation | 5 | Planned | |
| FR-27 | Metrics sliced by nuisance factor | 6 | Partial | Slice counts and positive rates per factor already in `/datasets/{id}/breakdown` and the dashboard |
| FR-28 | FPR disparity with CIs | 6 | Planned | Definition in METRICS.md (A-4) |
| FR-29 | Fairness mitigation | 6 | Planned | |
| FR-30 | Missing-channel test | 6 | Planned | Simulator already produces channel outages |
| FR-31 | +50% FP noise robustness | 6 | Planned | Simulator `pessimism` parameter supports it directly |
| FR-32 | Operating point by reviewer capacity | 4 | Planned | |
| FR-33 | Cost-sensitive threshold analysis | 6 | Planned | |
| FR-34 | POST /score | 7 | Planned | A-5 |
| FR-35 | GET /health, GET /model-info | 1 / 7 | Partial | `/health` done; `/model-info` returns only once a model exists |
| FR-36 | Batch scoring CLI | 7 | Planned | |
| FR-37 | Reviewer HTML report | 7 | Planned | Session timeline component already built |
| FR-38 | Experiment tracking | 4 | Planned | |
| FR-39 | One-command reproduction | 7 | Partial | `make simulate splits register` reproduces the data layer; content hash verified identical across runs |
| FR-40 | Fixed, recorded seeds | 1 | Done | Seed in config and manifest; per-session RNG streams |
| FR-41 | Drift monitor | 7 | Planned | |
| FR-42 | Model card | 7 | Planned | |
| NFR-1/2 | Performance | 3/7 | Partial | Simulator throughput measured; API list/overview ~40 ms on 5,000 sessions |
| NFR-3 | Scalability (streaming) | 1 | Done for simulator | Sharded generation, 250 sessions per shard |
| NFR-4/5 | Maintainability, test coverage | 1 | Done so far | 44 Python tests, 91% line coverage; ruff clean; strict TypeScript |
| NFR-6 | Portability | 1 | Partial | Dockerfiles and compose written but **not built here** (no Docker in the build environment) |
| NFR-7 | Usability | 1 | Done so far | Dashboard; responsive down to 390 px (no horizontal overflow, verified) |
| NFR-8 | Auditability | 1 | Done | `audit_logs` for every consent, status change, upload, deletion; status history per session |
| NFR-9 | Documentation | 1 | Done so far | README, SRS audit, plan, data dictionary, metrics, protocol |
| ETH-1 | Written consent, withdrawal, deletion | 1 | Done | All-items consent enforced by API and DB CHECK; withdrawal deletes media from disk (tested) |
| ETH-2 | No participants under 18 | 1 | Done | Operator attestation required; DB CHECK. Age itself is not stored |
| ETH-3 | Raw recordings deleted at project end | 1 | Partial | Per-recording `retention_until`, overwrite-then-unlink deletion; project-end purge command planned for Phase 7 |
| ETH-4 | Demographics never model inputs; stored separately | 1 | Done | Separate `demographics` table; nothing outside the fairness path reads it |
| ETH-5 | Harms section in final report | 7 | Planned | |
| ETH-6 | Output is a recommendation, never a verdict | 0 | Done | Contract has no VIOLATION recommendation (tested); UI wording checked |
| ETH-7 | Ethics approval before recording | 1 | Needs you | Consent records the approval reference |
| D1 | Repository | 0 | Done | |
| D2 | Simulator + config schema | 1 | Done | Pydantic schema, strict (unknown keys rejected) |
| D3 | Mock corpus + consent records | 1 | Needs you | De-identified register per A-22 |
| D4–D16 | Later deliverables | 2–7 | Planned | D12 container: Dockerfiles present, untested |
