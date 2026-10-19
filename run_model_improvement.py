"""Entry point: Phase 7 targeted experiments and selection (Steps D, E, F, 11).

Run:
    .venv\\Scripts\\python.exe run_model_improvement.py

Runs a small, justified hyperparameter search on the existing Skip-gram document
vectors, a comparable CBOW experiment, and the existing TF-IDF word baseline as
unigram vs unigram+bigram with Logistic Regression and Linear SVM. Candidates are
saved under ``models/experiments/``; the frozen Phase 5 selected model and reports
are never overwritten. Selection uses validation macro-F1 only.
"""

from __future__ import annotations

import sys

import config
from src.experiments import REPLACEMENT_THRESHOLD, run_phase7_experiments


def _write_recommendation(outcome) -> "Path":
    from pathlib import Path

    decision = outcome["decision"]
    baseline = decision["baseline"]
    candidate = decision["best_candidate"]
    probability = outcome["probability"]

    lines = [
        "# Phase 7 Recommendation — Retain or Replace the Selected Model",
        "",
        f"- Replacement threshold: **+{REPLACEMENT_THRESHOLD:.3f}** absolute "
        "validation macro-F1.",
        f"- Baseline (Phase 5 selected): exp {baseline['index']} "
        f"`{baseline['feature_type']}` / `{baseline['classifier']}` "
        f"macro-F1={baseline['macro_f1']:.4f}, accuracy={baseline['accuracy']:.4f}.",
    ]
    if candidate is not None:
        lines.append(
            f"- Best Phase 7 experiment: exp {candidate['index']} "
            f"`{candidate['feature_type']}` / `{candidate['classifier']}` "
            f"macro-F1={candidate['macro_f1']:.4f}, accuracy={candidate['accuracy']:.4f} "
            f"(improvement {candidate['improvement']:+.4f})."
        )
    verdict = (
        "REPLACE the selected model with the best candidate."
        if decision["replace"]
        else "RETAIN the existing Phase 5 selected model."
    )
    lines += ["", f"**Decision: {verdict}**", ""]

    if probability.get("available"):
        uncal = probability["uncalibrated"]
        lines.append("## Probability quality (validation)")
        lines.append("")
        if isinstance(uncal, dict) and uncal.get("available"):
            lines.append(
                f"- Score source: `{probability.get('score_source')}` — probability "
                "estimates, not calibrated guarantees."
            )
            lines.append(
                f"- Accuracy {uncal['accuracy']:.4f}, mean confidence "
                f"{uncal['mean_confidence']:.4f}, log-loss {uncal['log_loss']:.4f}, "
                f"Brier {uncal['brier']:.4f}, ECE {uncal['ece']:.4f}."
            )
        cal = probability.get("calibrated_sigmoid_cv3")
        if isinstance(cal, dict):
            if cal.get("available"):
                lines.append(
                    f"- Sigmoid calibration (cv=3, fit on training only): "
                    f"log-loss {cal['log_loss']:.4f}, Brier {cal['brier']:.4f}, "
                    f"ECE {cal['ece']:.4f} (fit {cal['fit_seconds']:.1f}s)."
                )
            else:
                lines.append(f"- Calibration not run/available: {cal.get('error') or cal.get('reason')}")
        lines.append("")

    lines += [
        "## Failures",
        "",
    ]
    if outcome["failures"]:
        for failure in outcome["failures"]:
            lines.append(f"- {failure['key']}: {failure['error']}")
    else:
        lines.append("- None. Every configured experiment completed.")
    lines.append("")

    out_dir = Path(config.EXPERIMENTS_REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "recommendation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    outcome = run_phase7_experiments(
        artifact_dir=config.EXPERIMENTS_DIR,
        out_dir=config.EXPERIMENTS_REPORT_DIR,
    )

    df = outcome["comparison"]
    print("\nValidation comparison (sorted by macro-F1):")
    print(
        df.head(20)[
            ["rank", "index", "source", "feature_type", "classifier",
             "macro_f1", "accuracy", "train_seconds"]
        ].to_string(index=False)
    )

    decision = outcome["decision"]
    print("\nDecision:", "REPLACE" if decision["replace"] else "RETAIN selected model")
    recommendation_path = _write_recommendation(outcome)
    print("\nArtifacts:")
    for name, path in outcome["paths"].items():
        print(f"  {name:12s}: {path}")
    print(f"  recommendation: {recommendation_path}")
    print(f"  candidates    : {outcome['artifact_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
