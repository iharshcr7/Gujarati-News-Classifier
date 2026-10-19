# 10 — Testing and operations

This chapter covers how to set up the environment, run the tests, reproduce
results, and regenerate artifacts — plus the operational gotchas you must know
before running anything.

## Environment setup

### Dependencies

`requirements.txt` lists the project's libraries:

```
pandas
numpy
gensim
scikit-learn
joblib
matplotlib
seaborn
streamlit
pytest
```

### Option A — a virtual environment (recommended, matches the project docs)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Option B — a system Python that already has the dependencies

If your interpreter already has the packages installed (as was the case during
this documentation pass), you can run scripts directly with `python`.

> **Verified fact from the documentation pass:** no `.venv` folder was present in
> the project root; a system Python 3.11 had pandas, numpy, sklearn, gensim,
> streamlit, joblib, and matplotlib installed, but **not pytest**. So Option A is
> the reliable path if you want to run the test suite.

## Running the tests

```powershell
python -m pytest -q
```

There are **13 test modules** plus `conftest.py` and `tests/__init__.py`:

| Test file | Covers |
| --- | --- |
| `test_config.py` | paths, column names, category labels |
| `test_data_loader.py` | CSV loading, missing/invalid files, real dimensions |
| `test_dataset_inspection.py` | Phase 1 report and Gujarati detection |
| `test_preprocessing.py` | cleaning, tokens, Unicode, robustness |
| `test_word2vec_trainer.py` | training pipeline structure |
| `test_document_vectors.py` | mean pooling, OOV, save/load, alignment |
| `test_text_features.py` | docvec loading, leakage-safe TF-IDF, SVD |
| `test_evaluation.py` | metrics, tables, plots |
| `test_classification.py` | estimators, SVD path, reproducibility, artifacts |
| `test_prediction.py` | inference round-trip, empty-input error |
| `test_error_analysis.py` | Phase 7 error grouping |
| `test_experiments.py` | Phase 7 improvement experiments |
| `test_app_content.py` | GUI content/helpers + contrast checks |

The project was built test-first (see the plans in `docs/superpowers/plans/`).
A full-suite green run was recorded in the project's working notes as **212 tests
passing**; that count was **not re-verified during this documentation pass**
(pytest was not installed in the available interpreter), so treat the exact number
as *unverified* — the *suite exists and is comprehensive* is what is verified.

### Two tests have real side effects — read this before a full run

Two tests intentionally exercise the real report writers and therefore
**overwrite** report files when you run the suite:

- `tests/test_dataset_inspection.py::test_run_inspection_writes_real_report`
  → rewrites `reports/dataset_inspection_report.txt`.
- `tests/test_preprocessing.py::test_pipeline_run_does_not_modify_source_csvs`
  → rewrites `reports/preprocessing_report.txt`.

This is expected behaviour (they assert the *source CSVs* are unchanged, not the
reports). If you need the report files frozen, run those tests last or restore
them afterwards. `reports/test_report.txt` (a 35-byte smoke-test file) is not to
be touched.

## Reproduction and determinism

- **Seed:** `config.RANDOM_SEED = 42` is passed wherever the estimator supports it.
- **Read-only discipline:** the pipeline never modifies `data/raw/` CSVs, the
  Word2Vec models, or the document-vector artifacts while training/evaluating.
- **Baseline reproduction:** Phase 7 re-ran the selected model on validation and
  matched the recorded metrics to **0.00e+00** (Chapter 08).
- **Caveat:** identical results across different machines/thread counts (e.g.
  Word2Vec with different `workers`) is *likely but not guaranteed*; treat
  cross-machine bit-exactness as *unverified*.

## Regenerating artifacts (only if needed)

The pipeline is a sequence of scripts, each run from the project root. They are
**ordered** and earlier outputs feed later stages:

```powershell
# 1. Inspect the raw data (fast; writes dataset_inspection_report.txt)
python inspect_dataset.py

# 2. Preprocess (writes preprocessing_report.txt; may dump *.preprocessed.csv)
python preprocess_dataset.py

# 3. Train Word2Vec (EXPENSIVE; overwrites models/word2vec_*.model)
python train_word2vec.py

# 4. Build document vectors (loads both models; writes data/processed/*.npz)
python generate_document_vectors.py

# 5. Run the 12 classification experiments (EXPENSIVE; writes models/classifiers/*)
python run_classification.py

# 6. Phase 7 error analysis (validation only)
python run_error_analysis.py

# 7. Phase 7 improvement experiments + retain/replace decision
python run_model_improvement.py
```

> **Do not run steps 3–7 casually.** They overwrite models and reports, and the
> training steps (3, 4, 5, 7) take significant time. If you only want to *use* the
> classifier, run the GUI instead (Chapter 09) — it needs no regeneration.

### Note on the disk footprint

- The trained Word2Vec models are small (≈9.37 MB each), but several saved
  classifier artifacts are large: `models/classifiers/exp011...joblib` is **822 MB**,
  `exp003` 564 MB, `exp007` 485 MB. `models/experiments/` totals ≈96 MB.
- `data/processed/*.preprocessed.csv` (the cleaned-text dumps) total ≈652 MB and
  are **orphaned**: `generate_document_vectors()` re-derives tokens from
  `data/raw/`, so these CSVs are not read by any later stage. They are safe to
  delete and regenerate if disk space matters.

## .gitignore (what is intentionally not committed)

```
.venv/
__pycache__/
.pytest_cache/
models/*.model
models/*.joblib
data/raw/*
data/processed/*
```

So large binaries (models, data) are excluded from version control by design.
*(Note: this project was not a Git repository during the documentation pass.)*

## Common troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: config` or `src` | Run scripts from the project root; modules import each other relative to it. |
| `Word2Vec.load` fails | The `.wv.vectors.npy` / `.syn1neg.npy` sidecars are missing next to the `.model` file. |
| Garbled output / `UnicodeEncodeError` in a script | Set `$env:PYTHONIOENCODING="utf-8"`; the entry points also reconfigure stdout to UTF-8. |
| Running a temp script that imports `config` | Prepend the project root to `$env:PYTHONPATH`, because the current directory is not automatically on `sys.path`. |
| Report files changed unexpectedly | The two side-effect tests above rewrote them. |

## A safe way to explore the data without changing anything

If you want to inspect the dataset yourself, write a **read-only** script (e.g.
under the OS temp directory) that loads the CSVs with pandas and prints shapes,
null counts, and samples — and set `$env:PYTHONPATH` to the project root. Avoid
running pipeline scripts just to look at data.

Next: [Chapter 11 — Glossary, dependencies, references](11_glossary_and_references.md).
