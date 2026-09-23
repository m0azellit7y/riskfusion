# Traceability matrix (final)

**Status key**
- **Done**: implemented and verified (evidence given).
- **Done, target not met**: implemented and measured, but the SRS target was missed; the entry says why.
- **Needs you**: the software is complete, but completion needs recordings or approvals only you can provide.

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| FR-1 | Configurable simulator | Done | `simulator/`, `configs/simulator/v1*.yaml`; 5,000 sessions in 114–125 s; seeded, content-hashed |
| FR-2 | Mock recording with consent | Done (software) / Needs you (60 sessions) | Recorder, consent, uploads, coverage tracker; browser E2E test |
| FR-3 | Simulator validated against mock data | Done (code) / Needs you | `validation/mock.py` (KS per feature); `POST /corpus/validate`; runs once sessions are analysed |
| FR-4 | Pretrained detectors, ≥ 4 channels | Done | all 11 channels in one pass: 8 from video/audio (SCRFD, ArcFace, head pose, YOLOX, MediaPipe face mesh / pose / palm, audio VAD, DSP audio tagger) + 3 from the browser; automatic on upload; tests with real media for each |
| FR-5 | Browser telemetry | Done | tab, full screen, paste length, monitors, keystroke counts |
| FR-6 | Measured detector error | Done (code) / Needs you | `measure_detector_errors`, writes `v2_measured.yaml` at ≥ 20 sessions |
| FR-7 | Explicit UNKNOWN states | Done | all 11 channels; liveness UNKNOWN in the first minute, pose UNKNOWN when elbows are out of view, both audio channels UNKNOWN without an audio track |
| FR-8 | Validation + dead letter | Done | dict and vectorised validators agree; API and simulator dead-letter |
| FR-9 | 1 s grid with gaps | Done | `features/engine.build_grid` |
| FR-10 | Out-of-order within 30 s | Done | `reorder_with_watermark` (unit test) |
| FR-11/12/14 | Base, windowed, interaction features | Done | 273 features; windows 10/60/300 s + session |
| FR-13 | Baseline normalisation | Done | scoped per SRS_AUDIT A-9; ablation −0.007 PR-AUC when off |
| FR-15 | Versioned Parquet feature store | Done | `schema.json` with schema hash |
| FR-16 | Rule baseline | Done | `configs/models/baseline.yaml` |
| FR-17 | GBT +15% PR-AUC over baseline | Done | +151% on validation |
| FR-18 | Temporal model | Done | ROCKET (400 kernels) + logistic regression, PR-AUC 0.828 |
| FR-19 | Class imbalance | Done | `scale_pos_weight` |
| FR-20 | Calibration on dedicated split | Done | Platt; ECE 0.016 validation and holdout |
| FR-21 | Holdout exactly twice | Done | committed access log with 2 entries; third refused |
| FR-22 | Ablations | Done | channels, normalisation, windows, model families |
| FR-23 | SHAP, top 5 | Done | LightGBM TreeSHAP, distinct reasons |
| FR-24 | Plain-language explanations | Done | templates; no verdict language (unit test) |
| FR-25 | Flags with IoU ≥ 0.5 for ≥ 70% | Done | 84.4% |
| FR-26 | Global interpretation | Done | importance + partial dependence (Performance page) |
| FR-27 | Sliced metrics | Done | 6 factors, bootstrap CIs |
| FR-28 | FPR disparity < 1.3 | **Done, target not met** | holdout 1.15–2.05; head covering worst |
| FR-29 | ≥ 1 mitigation measured | Done | reweighting measured; not adopted (worsened lighting) |
| FR-30 | Missing channel never increases risk | Done | guaranteed (monotone model); 0% on validation and holdout |
| FR-31 | +50% FP robustness | Done | PR-AUC 0.915 → 0.889 |
| FR-32 | Operating point from capacity | Done | 3% clean-session budget; tiers 0.5% / 3% / 10% |
| FR-33 | Cost-sensitive analysis | Done | cost ratios 1–50 |
| FR-34 | POST /score, p95 < 500 ms | Done | p95 342 ms, 87-min session |
| FR-35 | /health, /model-info | Done | |
| FR-36 | Batch CLI | Done | `riskfusion score-dir IN OUT` |
| FR-37 | HTML reviewer report | Done | `/sessions/{id}/report`, batch CLI |
| FR-38 | Experiment tracking | Done | `runs/` + index; Performance page |
| FR-39 | One-command reproduction | Done | `make pipeline` |
| FR-40 | Seeds recorded | Done | configs, manifests, bundle |
| FR-41 | Drift monitor | Done | PSI, review rate, Wilson-bounded FPR alerts |
| FR-42 | Model card | Done | `docs/MODEL_CARD.md`, generated |
| NFR-1/2 | Performance | Done | features 0.035 s/session; scoring p95 342 ms |
| NFR-3 | Streaming/scale | Done | sharded simulator and feature builder |
| NFR-4/5 | Maintainability, tests | Done | 64 automated tests + browser E2E; ruff, mypy, strict TS |
| NFR-6 | Portability | Partial | Dockerfiles written, not built in this environment |
| NFR-7/8/9 | Usability, audit, docs | Done | responsive dashboard; audit log; docs set |
| ETH-1..4, 6 | Consent, adults, deletion, demographics, no verdicts | Done | see SRS_AUDIT and tests |
| ETH-5 | Harms section | Done | FINAL_REPORT §4 |
| ETH-7 | Ethics approval | Needs you | reference recorded on each consent |
| D1–D16 | Deliverables | Done except D3 (needs recordings), D16 (your presentation) | |
