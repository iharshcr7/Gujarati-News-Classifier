# Phase 7 Recommendation — Retain or Replace the Selected Model

- Replacement threshold: **+0.005** absolute validation macro-F1.
- Baseline (Phase 5 selected): exp 8 `docvec:skipgram` / `gradient_boosting` macro-F1=0.8883, accuracy=0.8947.
- Best Phase 7 experiment: exp 107 `docvec:skipgram` / `gradient_boosting` macro-F1=0.8889, accuracy=0.8958 (improvement +0.0006).

**Decision: RETAIN the existing Phase 5 selected model.**

## Probability quality (validation)

- Score source: `predict_proba` — probability estimates, not calibrated guarantees.
- Accuracy 0.8947, mean confidence 0.9301, log-loss 0.3644, Brier 0.1600, ECE 0.0353.
- Sigmoid calibration (cv=3, fit on training only): log-loss 0.3589, Brier 0.1653, ECE 0.0465 (fit 204.7s).

## Failures

- None. Every configured experiment completed.
