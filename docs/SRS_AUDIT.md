# SRS audit — RiskFusion SRS v2.0

This audit was carried out before implementation. Each finding has an ID (A-n), what the SRS says, why it is a
problem, and the resolution adopted in this repository. Resolutions marked *proposed* need your sign-off
or your supervisor's, because they interpret the requirements rather than implement them.

Severity levels:
- **Conflict**: two requirements cannot both hold as written.
- **Ambiguity**: the requirement can be read in more than one way.
- **Gap**: the requirement is missing something needed to build or verify it.

## Conflicts

**A-1. Reviewed labels versus script-only ground truth.**
- *SRS:* DR-4 allows `source = 'reviewed'`. DR-2 says labels "never from later human judgement". The labels table is keyed on `session_id`, so a scripted label and a reviewed label for the same session cannot coexist.
- *Resolution:* DR-4 is implemented exactly and holds scripted and simulated ground truth. Reviewer verdicts (entity E3) will live in a separate `review_verdicts` table (Phase 6). They are never written into `labels` and never used for training or evaluation.

**A-2. Sealed holdout "accessed exactly twice" versus which numbers are reported on it.**
- *SRS:* FR-21 allows two accesses (end of Phase 4 and Phase 7). FR-16 evaluates the baseline on the holdout and FR-17 compares on validation. For FR-20 (calibration), FR-28 (fairness) and FR-30 (missing channels), the SRS never says which split is used.
- *Resolution (proposed):*
  - Access 1 (Phase 4) reports PR-AUC and recall at the operating point for the baseline and the fusion model.
  - Access 2 (Phase 7) reports the full final evaluation: discrimination, calibration, fairness and missing-channel behaviour.
  - Everything else uses validation.

**A-3. "Git history evidences exactly two accesses" cannot prove the absence of reads.**
- *SRS:* FR-21. A file read leaves no trace in git.
- *Resolution:* the holdout IDs and their SHA-256 are committed (the data itself is 1.2 GB and is not). Holdout IDs are reachable only through `open_sealed_holdout()`. That function requires a stated purpose, appends to the committed `HOLDOUT_ACCESS_LOG.jsonl`, refuses a third access, and detects manifest tampering. `load_split_ids("holdout")` raises. The API and dashboard also hide labels and detector output for holdout sessions. Committing the log after each access is what makes git evidence meaningful.

**A-4. Positive rate versus fairness slice size.**
- *SRS:* ASM-2 sets 3–8% positives. R7 requires ≥300 sessions per slice.
- *Problem:* at 6% and 5,000 sessions, a 300-session slice holds about 18 violations, so FNR confidence intervals are extremely wide. The FPR ratio max/min (FR-28) is undefined when a slice has zero false positives.
- *Resolution (proposed):*
  - Report bootstrap 95% CIs for every slice metric.
  - Compute the FPR ratio with a +1 (add-one) correction, and report it as "undefined" when both counts are zero.
  - Generate a larger evaluation-only simulated set if CIs are too wide to support a conclusion. Its holdout is never used for training.

**A-5. Latency: FR-34 versus NFR-1.**
- *SRS:* `/score` must answer with p95 < 500 ms for a 90-minute session, while NFR-1 allows 5 s for feature computation. The SRS never says whether `/score` receives raw events or precomputed features.
- *Resolution (proposed):* `/score` accepts events and must compute features inside the 500 ms budget. NFR-1 then becomes a batch target that is easily met.

**A-6. Evidence links versus deletion of raw media.**
- *SRS:* FR-37 attaches `evidence_ref` clip links. ETH-3 deletes all raw media at project end.
- *Resolution:* evidence clips are raw media and are deleted with it. Reports render "evidence deleted on <date>" instead of a broken link. D3 ships derived events only.

**A-7. The prompt asks for a "REAL" session source; CON-1 forbids real candidate data.**
- *Resolution:* only `SIMULATED` and `MOCK` exist, enforced by a database CHECK constraint.

## Ambiguities

- **A-8. Is mock data used for training?** DR-2 and FR-2 imply mock data is a realism anchor. *Resolution (proposed):* models train on simulated data only. Mock data is used for FR-3, FR-6 and as an external validation set. A 15% participant holdout of 60 mock sessions (about 9 sessions) is too small to be informative.
- **A-9. FR-13 baseline normalisation.** Z-scoring against the candidate's own first 60 s:
  - removes any violation that happens in that window, including impersonation from the start;
  - divides by zero for constant signals;
  - is undefined when those seconds are UNKNOWN.

  *Resolution (proposed):* apply it only to continuous behavioural features (head pose, gaze probability, movement). Use a median/MAD baseline with a floor. Skip normalisation when fewer than 30 usable baseline seconds exist, and emit a flag feature saying so.
- **A-10. FR-12 window aggregation to session level.** Rolling 10/60/300-second statistics produce a series per second, but the model is session-level. *Resolution (proposed):* reduce each rolling feature to its session maximum, 95th percentile and time-above-threshold. `count_over_threshold` needs per-feature thresholds; these will be set on the training split only and versioned with the features.
- **A-11. FR-25 flag IoU.** The SRS does not say which split is used or how flags are produced from a session-level model. *Resolution (proposed):* a separate interval scorer over rolling features produces flags, evaluated on validation.
- **A-12. Identity and speaker enrolment are never specified**, yet the identity channel matches against "the enrolled candidate". *Resolution:* the recorder captures an enrolment photo before recording starts. The impersonation script swaps people after enrolment. Voice enrolment will use the first 30 s of candidate speech (Phase 2).
- **A-13. FR-5 monitor count.** Only Chromium exposes `screen.isExtended`; the full count needs a permission prompt. *Resolution:* where supported, emit a count of 1 or 2; otherwise emit `DEVICE_UNKNOWN (not_observable)`. This is honest missingness, exercised by FR-30.
- **A-14. FR-4 lists 4 channels; the contract defines 11.** *Resolution:* all 11 are implemented. Attention and pose come from video in Phase 2, and liveness is best-effort (see A-17).
- **A-15. Versioning of model and features** is shown by example (`fusion-v2.4.1`) but not defined. *Resolution:* semantic versions, plus a content hash of the training data and config recorded with every model.
- **A-16. `confidence_band` and the four recommendation tiers.**
  - FR-32 defines one operating point, but four tiers need three thresholds.
  - The method for `confidence_band` is not specified.

  *Resolution (proposed):*
  - `confidence_band` is a bootstrap interval over calibration-set resamples.
  - The tiers split at three thresholds chosen on the calibration split: routine review at 10% of clean sessions, human review at the FR-32 operating point (3%), and priority review at 0.5%. Implemented in `modeling/train.py`.

## Gaps

- **A-17. Liveness/PAD models.** Licences are restrictive: CelebA-Spoof is non-commercial and OULU-NPU requires an agreement. This is a Phase 2 risk. The simulator already models the channel.
- **A-18. DR-3 figures cannot be simulator parameters as written.** Precision depends on prevalence. *Resolution:* per-second FP rate = pessimism × reference prevalence × recall × (1 − precision) / precision, with the reference prevalence stated per detector in `configs/simulator/v1.yaml`.
- **A-19. FR-6 has no frame-level truth.** Scripts give interval truth only. *Resolution (proposed):* measure detector error at event level inside scripted intervals, and assume the clean state outside them (candidate present, alone, no phone). This assumption biases FP estimates upward, which errs on the safe side.
- **A-20. Security.** There is no authentication or authorisation requirement, although the system stores faces and voices. The prototype binds to localhost and records the operator name in an audit log. *Proposed:* add authentication before any shared deployment. The SRS also sets no encryption-at-rest requirement.
- **A-21. Retention periods.** "End of project" is the only retention rule. *Resolution:* every recording gets `retention_until` (configurable, default 180 days) plus a project-end purge.
- **A-22. D3 "consent records".** These contain names, so sharing them violates ETH-4. *Resolution:* D3 ships a de-identified consent register (code, form version, date, status).
- **A-23. Helpers on camera.** The second-person, remote-assistance and impersonation scripts need a second adult who is recorded. The SRS covers only the candidate's consent. *Resolution:* sessions using those scripts require a helper participant with their own active consent. A helper's withdrawal deletes the session.
- **A-24. Reviewer workflow.** Overturn is defined (E3) but no requirement captures reviewer verdicts or measures overturns. *Proposed:* add this in Phase 6.
- **A-25. Ethics approval timing.** Phase 0 only submits the ethics application, while Phase 1 collects data. Recording must wait for approval. The consent form records the approval reference (ETH-7).
- **A-26. Undefined numbers:**
  - FR-39 "±1%" (absolute or relative): treated as absolute percentage points.
  - FR-1 hardware: the 125 s result was measured on 1 vCPU.
  - Late events more than 30 s out of order (FR-10): routed to the dead-letter store with reason `late`.
  - ECE bins: 15 equal-width bins.
  - NFR-4 "fusion modules": taken to mean `src/riskfusion/fusion`, features and calibration.
