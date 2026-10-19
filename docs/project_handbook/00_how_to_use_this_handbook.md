# 00 — How to use this handbook

## Who this is for

- **New team members** who need to understand the project quickly.
- **Students / beginners** learning how a real NLP text-classification pipeline
  is assembled from classical (non-deep-learning) components.
- **Maintainers** who need to run, extend, or debug the system.

You should be able to read Python and use a terminal. No prior
knowledge of NLP, Gujarati, or machine learning is assumed — terms are explained
as they appear, and collected in [Chapter 11](11_glossary_and_references.md).

## How the handbook is organised

Chapters follow the *life of one headline* through the system:

```
raw CSV  ->  inspect  ->  preprocess  ->  Word2Vec  ->  document vector
        ->  classifier  ->  evaluation  ->  GUI prediction
```

- **Chapters 01–03** describe the goal, the repository, and the data.
- **Chapters 04–07** walk the pipeline stage by stage (the core of the book).
- **Chapter 08** covers the Phase 7 error analysis and improvement study.
- **Chapters 09–10** cover the GUI and day-to-day operations/testing.
- **Chapter 11** is a glossary, a module dependency diagram, and pointers to the
  repository's own generated reports.

## How facts in this handbook were verified

This handbook was written by **reading the repository**, not by assuming how it
"usually" works. The following sources were inspected:

- **Source code:** `config.py`, everything in `src/`, `app/`, and the root
  entry-point scripts.
- **Generated reports:** files under `reports/` and `reports/error_analysis/`,
  `reports/experiments/`.
- **Model and feature artifacts:** `.meta.json` files under `data/processed/`,
  file listings under `models/`.
- **Phase plans:** the implementation plans under `docs/superpowers/plans/`.
- **The dataset itself:** the raw CSVs under `data/raw/`, read in a read-only
  way to confirm row counts, columns, class distributions, and samples.

### Accuracy conventions used here

| Marker | Meaning |
| --- | --- |
| (plain text) | Verified directly from the repository. |
| *inferred* | A reasonable conclusion that follows from verified facts but is not stated verbatim anywhere. |
| *unverified* / *not confirmed* | Could not be confirmed from the repository; treat with caution. |
| illustrative example | A made-up example used only to explain a concept. |

All numeric results (accuracy, macro-F1, row counts, vocabulary sizes) are quoted
**exactly** as they appear in the repository's own reports, so they can be
cross-checked.

## A note on the development environment

The project's plans mention a Python virtual environment at `.venv\Scripts\`.
During this documentation pass, **no `.venv` folder was present** in the project
root; a system Python 3.11 was available with the project's dependencies
installed. Chapter 10 explains how to create the virtual environment and install
dependencies (`requirements.txt`), so this discrepancy is expected and easy to
resolve. It does **not** affect the correctness of the documentation.

## Running things safely

The pipeline is **not** a single command; it is a series of scripts, each of
which reads inputs and writes artifacts/reports. Several steps are **expensive**
(training Word2Vec, generating document vectors, running the 12 classification
experiments) and some tests **overwrite** report files as a side effect. If you
only want to *use* the model, run the GUI (Chapter 09). If you want to
*reproduce* results, read Chapter 10 first.

Ready? Continue to [Chapter 01 — Project overview and goals](01_overview_and_goals.md).
