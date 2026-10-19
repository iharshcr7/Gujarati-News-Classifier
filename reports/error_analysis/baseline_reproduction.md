# Baseline Reproduction — Phase 5 Selected Model (Phase 7, Step C)

The frozen Phase 5 selected model was loaded read-only and re-evaluated on the existing validation features. No model is retrained here, and no test labels are used.

- Model: docvec:skipgram / gradient_boosting (learning_rate=0.1, max_iter=200, seed=42)
- Artifact: C:\Users\harsh\OneDrive\Documents\Desktop\Gujarati News Article\models\classifiers\selected_model.joblib
- Tolerance: 0.001 absolute

## Comparison

| Metric | Reproduced | Recorded | Abs. diff | Within tolerance |
|---|---|---|---|---|
| Accuracy | 0.894742 | 0.894742 | 0.00e+00 | yes |
| Macro precision | 0.895153 | 0.895153 | 0.00e+00 | yes |
| Macro recall | 0.882586 | 0.882586 | 0.00e+00 | yes |
| Macro F1 | 0.888274 | 0.888274 | 0.00e+00 | yes |
| Weighted F1 | 0.894235 | 0.894235 | 0.00e+00 | yes |

**Verdict: reproduction MATCHES the recorded Phase 5 validation metrics within tolerance.**
