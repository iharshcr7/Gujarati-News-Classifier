# 08 — Error analysis and improvement (Phase 7)

**Modules:** `src/error_analysis.py`, `src/experiments.py` ·
**Entry points:** `run_error_analysis.py`, `run_model_improvement.py` ·
**Reports:** `reports/error_analysis/*`, `reports/experiments/*` ·
**Plan:** `docs/superpowers/plans/2026-10-18-gujarati-error-analysis-model-improvement.md`

## Why this phase exists

A single accuracy number does not tell you *where* or *why* a model fails. Phase 7
does two things:

1. **Error analysis** — study the model's mistakes on the **validation** split.
2. **Model improvement** — try targeted changes and decide **retain or replace**
   the selected model using an explicit, pre-registered threshold.

**cardinal rule:** all of this uses **validation only**. The test split is never
inspected here.

## Step 1 — Reproduce the baseline first

Before analysing errors, the project re-loads the frozen selected model
**read-only** and re-evaluates it on validation. If the numbers didn't match the
original Phase 5 run, any later conclusion would be suspect. They matched exactly
(`reports/error_analysis/baseline_reproduction.md`, tolerance 0.001):

| Metric | Reproduced | Recorded | Difference |
| --- | --- | --- | --- |
| Accuracy | 0.894742 | 0.894742 | 0.00e+00 |
| Macro precision | 0.895153 | 0.895153 | 0.00e+00 |
| Macro recall | 0.882586 | 0.882586 | 0.00e+00 |
| Macro F1 | 0.888274 | 0.888274 | 0.00e+00 |
| Weighted F1 | 0.894235 | 0.894235 | 0.00e+00 |

Verdict: **reproduction matches**. The baseline is trustworthy.

## Step 2 — What the errors look like

From `reports/error_analysis/validation_error_analysis.md`:

- Validation examples: **10,175**; misclassified: **1,071**; error rate **0.1053 (10.53%)**.

**Per-class validation metrics** — the weak spots are Nation and Crime:

| Category | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| State | 0.9210 | 0.9470 | 0.9338 | 1,624 |
| International | 0.8645 | 0.9183 | 0.8906 | 1,494 |
| Entertainment | 0.9342 | 0.9136 | 0.9238 | 1,134 |
| Business | 0.8845 | 0.8770 | 0.8807 | 1,179 |
| **Nation** | 0.8004 | 0.7671 | **0.7834** | 1,181 |
| Technology | 0.8731 | 0.9130 | 0.8926 | 874 |
| Sports | 0.9576 | 0.9511 | 0.9543 | 1,044 |
| Astro | 0.9600 | 0.9452 | 0.9526 | 839 |
| **Crime** | 0.8203 | 0.7432 | **0.7798** | 479 |
| Auto | 0.9360 | 0.8502 | 0.8910 | 327 |

**Top confusion pairs** (true → predicted):

| Rank | True → Predicted | Count | Share of all errors |
| --- | --- | --- | --- |
| 1 | Nation → International | 97 | 0.0906 |
| 2 | Crime → State | 61 | 0.0570 |
| 3 | Business → Nation | 50 | 0.0467 |
| 4 | Nation → Business | 50 | 0.0467 |
| 5 | Crime → Nation | 44 | 0.0411 |
| 6 | Business → Technology | 43 | 0.0401 |
| 7 | International → Nation | 43 | 0.0401 |
| 8 | Entertainment → Nation | 38 | 0.0355 |
| 9 | Business → International | 36 | 0.0336 |
| 10 | State → International | 31 | 0.0289 |

## Step 3 — Investigating the causes

`reports/error_analysis/error_cause_investigation.md` groups the evidence:

**Headline length vs. error rate.**

| Token bucket | Support | Errors | Error rate |
| --- | --- | --- | --- |
| 2–3 | 3 | 1 | 0.3333 |
| 4–7 | 104 | 6 | 0.0577 |
| 8–15 | 2,072 | 269 | 0.1298 |
| 16+ | 7,996 | 795 | 0.0994 |
| mixed script (ASCII) | 6,259 | 704 | 0.1125 |

Short headlines are riskier (little signal), but — interestingly — the 8–15 token
bucket has a *higher* error rate than very long ones, likely because medium-length
headlines are often just a headline (less context) whereas long ones are full
articles (more topic words).

**Vocabulary overlap.** The investigation computes Jaccard overlap between every
pair of categories' vocabularies. The most overlapping pairs are exactly the ones
that confuse the model:

| Class A | Class B | Shared tokens | Jaccard |
| --- | --- | --- | --- |
| International | Nation | 8,619 | 0.2132 |
| Business | Nation | 7,157 | 0.1959 |
| Business | Technology | 5,088 | 0.1904 |
| Entertainment | Nation | 7,151 | 0.1870 |
| State | Crime | 5,374 | 0.1828 |

The shared tokens are mostly generic Gujarati function words (છે, અને, કે, પર,
પણ …) plus, for State/Crime, the word પોલીસ ("police"). This is strong evidence
that the confusions are **intrinsic topic overlap**, not a bug.

**Representative mistakes** (real validation rows, abbreviated):

- Nation → International: a long piece about India's **hypersonic missile**
  programme (defence/international flavour).
- Crime → State: a **surat land-grabbing** case — "Crime" and "State" genuinely
  overlap when the news is about official/police action.
- Business → Nation: a **Tomato Price** story predicted as Nation, because
  national price news can read like policy news.

## Step 4 — Probability quality

Because the GUI shows scores, it matters whether the model's scores are
meaningful. From `reports/experiments/validation_probability_quality.json` /
`recommendation.md`:

| Measure | Value |
| --- | --- |
| Score source | `predict_proba` (probability *estimates*, not calibrated guarantees) |
| Mean confidence | 0.9301 |
| Log-loss | 0.3644 |
| Brier score | 0.1600 |
| Expected calibration error (ECE) | 0.0353 |
| Sigmoid calibration (cv=3, train only) | log-loss 0.3589, Brier 0.1653, ECE 0.0465 |

Interpretation: the model's confidences are **reasonably well calibrated**
(ECE ≈ 3.5%) and calibration did **not** improve log-loss enough to justify the
added complexity. So the project keeps the raw probabilities and (importantly)
**describes them honestly in the UI** as estimates, not certainties.

## Step 5 — The improvement experiments

`run_model_improvement.py` (`src/experiments.py`) ran **15 targeted experiments**,
exp100–exp114, on validation. They varied, among other things, the gradient
boosting **learning rate** and **tree leaves**, TF-IDF **n-gram ranges**
(unigram vs. unigram+bigram), and CBOW/skip-gram combinations. All 15 completed —
`experiment_failures.json` is empty.

The full ranking is in `reports/experiments/validation_comparison.md`. Top rows:

| Rank | Exp | Feature | Classifier | Macro-F1 | Accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | 107 | Skip-gram docvec | gradient_boosting | 0.8889 | 0.8958 |
| 2 | 106 | Skip-gram docvec | gradient_boosting | 0.8886 | 0.8944 |
| 3 | **8 (baseline)** | Skip-gram docvec | gradient_boosting | 0.8883 | 0.8947 |
| 4 | 100 | Skip-gram docvec | gradient_boosting | 0.8883 | 0.8947 |
| 5 | 102 | Skip-gram docvec | gradient_boosting | 0.8883 | 0.8947 |

## Step 6 — The retain/replace decision

`config.REPLACEMENT_THRESHOLD = 0.005` — a new model must beat the baseline by
**+0.005** absolute validation macro-F1 to justify the churn of replacing it.

- Baseline (exp 8): macro-F1 **0.8883**.
- Best candidate (exp 107): macro-F1 **0.8889** → improvement **+0.0006**.

Since **0.0006 < 0.005**, the decision (`reports/experiments/recommendation.md`)
is:

> **RETAIN the existing Phase 5 selected model.**

This is a disciplined outcome: the extra experiments demonstrated that the
selected model is already near the ceiling of what this feature set and these
models can achieve, and the tiny gains are within noise. Keeping the simpler,
already-frozen model is the right engineering call.

## What was learned (and what a future phase could try)

- The remaining errors are **mostly intrinsic** (overlapping categories), so most
  classical tweaks yield little.
- **Calibration is already good**; score post-processing is not the bottleneck.
- A future attempt would need *new information*, e.g. richer context than the
  headline (article body), Gujarati-specific sub-word tokenization, or a
  transformer — all explicitly out of scope for this classical pipeline.

Next: [Chapter 09 — The Streamlit GUI](09_gui.md).
