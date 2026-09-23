# Implementation plan

The SRS phases are kept as the backbone. A product layer runs through all of them, so each phase ends with
something usable in the dashboard rather than only notebooks:
- a FastAPI service;
- PostgreSQL as the operational store;
- local object storage;
- a React dashboard.

## Architecture

```
 Browser (React dashboard)                    CLI / batch jobs
   consent, recording, telemetry,               simulate, splits, register,
   sessions, datasets, (later) reports          (later) extract, features, train, evaluate
          │  /api                                          │
          ▼                                                ▼
 FastAPI service ──────────── PostgreSQL (operational) ◄───┤
   lifecycle, uploads,          participants, consents,    │
   event ingest, scoring*       sessions, recordings,      │
          │                     labels (DR-4), splits,     │
          ▼                     events (mock), audit       │
 Object storage (local now,                                │
   S3-compatible later)       Parquet (analytical) ◄───────┘
   raw media, retention         simulated events, features*, predictions*
```
`*` = later phases. The contract `event.v1` is the only way data enters the pipeline, whether it comes from the
simulator, browser telemetry or real detectors. Parquet is the analytical store required by FR-15; PostgreSQL
holds everything with a lifecycle.

## Phases

| Phase | Scope | Exit criteria | State |
|---|---|---|---|
| 0 | Repo, pinned environment, contracts, validators, audit, ethics submission | Schemas + validators; audit written | **Done** (ethics submission: yours) |
| 1 | Simulator, consent + mock recording, label store, splits, sealed holdout, dashboard for all of it | 5,000 sessions < 10 min; label store queryable; holdout committed; recording works end to end | **Done in software**; 60 mock recordings need volunteers |
| 2 | Detectors on mock recordings (face, identity, liveness, head pose, objects, VAD/speaker), FR-5 extraction, FR-6 error measurement, FR-3 validation | Measured error report; simulator v2 with measured rates | Next |
| 3 | 1 s grid, out-of-order handling, base + windowed + interaction features, baseline normalisation, feature store | Feature store versioned; < 5 s per 90-min session | |
| 4 | Rule baseline, GBT model, imbalance handling, calibration, operating point, experiment tracking, holdout access 1 | +15% PR-AUC over baseline on validation; ECE < 0.05 | |
| 5 | Temporal model, SHAP, explanations, flags, global interpretation, ablations | Flags IoU target; explanation readability | |
| 6 | Sliced metrics, FPR disparity, mitigation, missing-channel and noise robustness, cost analysis, reviewer verdicts | Fairness report | |
| 7 | /score, /model-info, batch CLI, HTML reviewer report, drift monitor, model card, one-command reproduction, holdout access 2, final report | All D1–D16 | |

## How each later phase appears in the product

- **Phase 2:** "Process recording" on uploaded sessions moves them through PROCESSING. Detector events land in the existing `events` table and show on the session timeline next to ground truth.
- **Phase 4:** the Risk assessment panel on the session page fills in, and a Models page shows versions, calibration and the operating point.
- **Phase 5–6:** explanations and flags on the timeline, plus Evaluation and Fairness pages.
- **Phase 7:** a review queue sorted by calibrated risk, reviewer verdicts, the HTML report and drift charts.
