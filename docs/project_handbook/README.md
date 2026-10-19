# Gujarati News Classifier — Project Handbook

A complete, accurate, beginner-friendly guide to the **Gujarati News Article
Classifier**: a classical machine-learning system that reads a Gujarati news
headline and predicts which of **10 news categories** it belongs to.

This handbook explains *what* the project does, *why* it is built the way it is,
*how* every stage works, how to run and troubleshoot it, and how to extend it.
It is written for a reader who is comfortable with basic Python but is new to
Natural Language Processing (NLP) in general and Gujarati text in particular.

---

## Table of contents

| # | Chapter | What you will learn |
| --- | --- | --- |
| 00 | [How to use this handbook](00_how_to_use_this_handbook.md) | Audience, conventions, how facts were verified |
| 01 | [Project overview and goals](01_overview_and_goals.md) | The problem, the 8 phases, headline results |
| 02 | [Repository map](02_repository_map.md) | Every folder and file, and what it is for |
| 03 | [The dataset](03_datasets.md) | Categories, splits, sizes, distributions, quirks |
| 04 | [Text preprocessing](04_preprocessing.md) | Cleaning and tokenizing Gujarati Unicode text |
| 05 | [Word2Vec embeddings](05_word2vec.md) | CBOW vs Skip-gram, training, artifacts, OOV |
| 06 | [Document vectors](06_document_vectors.md) | Turning a headline into a fixed 100-d vector |
| 07 | [Classification](07_classification.md) | Feature configs, 4 models, selection, metrics |
| 08 | [Error analysis and improvement](08_error_analysis_improvement.md) | Phase 7: where it fails and why we kept the model |
| 09 | [The Streamlit GUI](09_gui.md) | Running and using the interactive app |
| 10 | [Testing and operations](10_testing_and_operations.md) | Env setup, tests, reproducibility, regenerating |
| 11 | [Glossary, dependencies, references](11_glossary_and_references.md) | Key terms, module graph, source reports |

The eight architecture diagrams live in [`diagrams/`](diagrams/README.md) and are
also embedded in the relevant chapters.

---

## One-paragraph summary

This project takes the **SHC Gujarati news headline dataset** (about 100,000
labelled headlines across 10 categories), cleans and tokenizes the Gujarati text,
trains two **Word2Vec** embedding models (CBOW and Skip-gram), converts every
headline into a **100-dimensional document vector** by mean-pooling its word
vectors, and trains four classical classifiers per feature set. The best
combination — **Skip-gram document vectors + Histogram Gradient Boosting** — was
selected using **validation macro-F1**, then evaluated once on a held-out test
split, reaching **macro-F1 ≈ 0.891** and **accuracy ≈ 0.895**. A **Streamlit**
GUI lets a person type a headline and see the predicted category and per-class
scores. A final Phase 7 error-analysis study investigated the remaining mistakes
and confirmed the selected model should be **retained**.

---

## How to read this handbook

- **New to the project?** Read 01 → 02 → 03, then follow the pipeline order
  04 → 05 → 06 → 07.
- **Just want to run it?** Jump to [Chapter 09](09_gui.md) (GUI) and
  [Chapter 10](10_testing_and_operations.md) (setup).
- **Looking for why a decision was made?** Chapters 07 and 08 record the
  experiments and selection logic; Chapter 03 records the data quirks.
- **Reading the code?** Chapter 02 maps every file; Chapter 11 shows the
  import/dependency graph.

> **Grounding note.** Every number, path, and behaviour in this handbook was read
> from the repository itself (source files, reports, and metadata) during a
> documentation pass. Where something could not be verified it is explicitly
> marked *unverified* or *inferred*. See
> [Chapter 00](00_how_to_use_this_handbook.md) for the full method.
