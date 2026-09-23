# Metric definitions

These are fixed now, before any model exists, so that later results cannot be tuned to the definition.
Computation arrives in Phases 4–6.

- **Positive class:** session label `violation = true` (scripted or simulated ground truth, never reviewer judgement).
- **PR-AUC:** average precision (step-wise interpolation) over session scores. This is the primary discrimination metric because positives are rare.
- **Recall at the operating point:** TP / (TP + FN) at the threshold chosen on the calibration split.
- **FPR:** FP / (FP + TN) among clean sessions. This is the false-accusation rate, the quantity the operating point constrains (FR-32).
- **Operating point:** the lowest threshold whose calibration-split FPR is ≤ the reviewer-capacity budget (default 3%). If recall < 0.80 at that point, this is reported as a failure; the threshold is not loosened.
- **ECE:** Σ_b (n_b/N)·|mean(p)_b − mean(y)_b| over 15 equal-width bins on [0, 1]. Also report the reliability diagram and the Brier score.
- **FPR disparity (FR-28):** max_s FPR_s / min_s FPR_s over the levels of one nuisance factor. Each FPR_s uses (FP_s + 1)/(N_s + 2) smoothing, and the ratio is reported as "undefined" if every slice has FP = 0. 95% CI by 2,000 stratified bootstrap resamples. Slices under 300 sessions are reported but marked underpowered.
- **Flag IoU (FR-25):** for each true interval, the maximum IoU with any predicted flag of a matching type. Success means IoU ≥ 0.5. The metric is the fraction of true episodes that succeed.
- **Missing-channel monotonicity (FR-30):** for every session and channel, score(channel removed) − score(full) ≤ 0.001. Report the violation rate and the maximum increase.
- **Overturn rate (later):** share of reviewed flagged sessions the reviewer finds clean.
