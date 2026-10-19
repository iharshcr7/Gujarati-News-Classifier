# Diagrams

This folder holds the architecture and flow diagrams for the Gujarati News
Classifier handbook. Each diagram has a Mermaid source file (`.mmd`) and is also
embedded (inline, so it renders on GitHub and in most Markdown viewers) in the
chapter that discusses it.

| File | Diagram | Shown in |
| --- | --- | --- |
| `A_system_architecture.mmd` | End-to-end system architecture | Chapter 01, 02 |
| `B_directory_map.mmd` | Repository directory map | Chapter 02 |
| `C_preprocessing_pipeline.mmd` | Text preprocessing pipeline | Chapter 04 |
| `D_cbow_vs_skipgram.mmd` | CBOW vs Skip-gram | Chapter 05 |
| `E_document_vector_generation.mmd` | Headline (document) vector generation | Chapter 06 |
| `F_classification_and_evaluation.mmd` | Experiments, selection, evaluation | Chapter 07 |
| `G_gui_inference.mmd` | GUI inference sequence | Chapter 09 |
| `H_file_dependency_graph.mmd` | Python module dependency graph | Chapter 11 |

## How to view the diagrams

1. **Inline (no tools):** open any chapter `.md` file or this file on GitHub, or
   in VS Code with the "Markdown Preview Mermaid Support" extension. The
   ```` ```mermaid ```` blocks render directly.
2. **Browser preview (no install):** open `preview.html` in any modern browser.
   It loads Mermaid from a CDN and renders all eight diagrams on one page. This
   option needs an internet connection the first time.
3. **Render to PNG/SVG (optional, requires tools not installed here):**
   ```powershell
   npx -y @mermaid-js/mermaid-cli -i A_system_architecture.mmd -o A_system_architecture.svg
   ```
   `mermaid-cli` (`mmdc`) was **not** found in this project and pulls a headless
   Chromium download, so image export was intentionally skipped.

> Note: mermaid CDN URLs in `preview.html` are for viewing the local diagrams
> only; they are not part of the application runtime.

---

## A. End-to-end system architecture

```mermaid
flowchart TD
    subgraph DATA["1. Data (data/raw/)"]
        RAW["Gujarati_SHC_LDC_LPC_Train.csv<br/>79,416 rows"]
        VAL["...Valid.csv<br/>10,251 rows"]
        TST["...Test.csv<br/>10,041 rows"]
    end
    subgraph INSPECT["2. Inspection (Phase 1)"]
        DI["inspect_dataset.py<br/>src/dataset_inspection.py"]
    end
    subgraph PREP["3. Preprocessing (Phase 2)"]
        PP["preprocess_dataset.py<br/>src/preprocessing.py"]
        CLN["Cleaned splits<br/>78,806 / 10,175 / 9,972"]
    end
    subgraph W2V["4. Word2Vec training (Phase 3)"]
        CBOW["word2vec_cbow.model (sg=0)"]
        SKIP["word2vec_skipgram.model (sg=1)"]
    end
    subgraph DOCVEC["5. Document vectors (Phase 4)"]
        DV["generate_document_vectors.py<br/>mean-pool 100-d vectors"]
        NPZ["data/processed/*.docvecs.npz + .meta.json"]
    end
    subgraph CLS["6. Classification (Phase 5)"]
        FEAT["Feature configs A/B/C"]
        MODELS["4 classifiers x 3 configs"]
        SEL["Select by validation macro-F1"]
        TEST["Evaluate once on test split"]
        ART["models/classifiers/selected_model.joblib"]
    end
    subgraph P7["7. Error analysis + improvement (Phase 7)"]
        ERR["run_error_analysis.py"]
        IMP["run_model_improvement.py"]
        DEC["Decision: RETAIN selected model"]
    end
    subgraph GUI["8. GUI (Phase 6)"]
        APP["streamlit run app/streamlit_app.py"]
        PRED["src/prediction.NewsClassifier"]
    end
    RAW --> DI
    VAL --> DI
    TST --> DI
    DI --> PP
    PP --> CLN
    CLN --> W2V
    CBOW --> DV
    SKIP --> DV
    CLN --> DV
    DV --> NPZ
    NPZ --> FEAT
    CLN --> FEAT
    FEAT --> MODELS
    MODELS --> SEL
    SEL --> TEST
    SEL --> ART
    ART --> ERR
    ART --> IMP
    ART --> PRED
    ERR --> IMP
    IMP --> DEC
    DEC --> PRED
    PRED --> APP
```

## B. Repository directory map

```mermaid
flowchart LR
    ROOT["Gujarati News Article/"]
    ROOT --> CFG["config.py"]
    ROOT --> SRC["src/ library modules"]
    ROOT --> APP["app/ GUI"]
    ROOT --> DATA["data/ raw + processed"]
    ROOT --> MODELS["models/"]
    ROOT --> REPORTS["reports/"]
    ROOT --> TESTS["tests/"]
    ROOT --> DOCS["docs/"]
```

## C. Preprocessing pipeline

```mermaid
flowchart TD
    IN["Raw headline"] --> VALID{"is_valid_text?"}
    VALID -- "no" --> EMPTY["empty text, 0 tokens"]
    VALID -- "yes" --> URL["Remove URLs"] --> HTML["Remove HTML"] --> EMAIL["Remove emails"]
    EMAIL --> PUNCT["Replace non-token chars with space"]
    PUNCT --> WS["Collapse whitespace / strip"]
    WS --> TOK["tokenize (split whitespace)"]
    TOK --> STOP{"remove_stopwords? default off"}
    STOP -- "yes" --> REM["drop stopwords"]
    STOP -- "no" --> KEEP["keep tokens"]
    REM --> OUT["PreprocessedText"]
    KEEP --> OUT
    EMPTY --> OUT
```

## D. CBOW vs Skip-gram

```mermaid
flowchart LR
    subgraph CBOWG["CBOW (sg=0): context -> target"]
        CI1["w(i-2)"] --> CH["sum/average"] --> CO["predict w(i)"]
        CI2["w(i-1)"] --> CH
        CI3["w(i+1)"] --> CH
        CI4["w(i+2)"] --> CH
    end
    subgraph SKIPG["Skip-gram (sg=1): target -> context"]
        SI["w(i)"] --> SO1["predict w(i-2)"]
        SI --> SO2["predict w(i-1)"]
        SI --> SO3["predict w(i+1)"]
        SI --> SO4["predict w(i+2)"]
    end
```

## E. Document vector generation

```mermaid
flowchart TD
    H["Headline"] --> PP["preprocess -> tokens"]
    PP --> LOOP["For each token"]
    LOOP --> LOOK{"in vocabulary?"}
    LOOK -- yes --> ACC["add 100-d embedding"]
    LOOK -- no --> SKIP["skip (OOV)"]
    ACC --> DONE{"more tokens?"}
    SKIP --> DONE
    DONE -- yes --> LOOP
    DONE -- no --> HAS{"any in-vocab?"}
    HAS -- yes --> MEAN["mean of in-vocab vectors"]
    HAS -- no --> ZERO["zero vector + fallback count"]
    MEAN --> SAVE["*.docvecs.npz + *.meta.json"]
    ZERO --> SAVE
```

## F. Classification and evaluation

```mermaid
flowchart TD
    TRAIN["Cleaned train split"] --> A["Config A: skipgram docvecs"]
    TRAIN --> B["Config B: cbow docvecs"]
    TRAIN --> C["Config C: TF-IDF (train only)"]
    A --> E1["4 estimators"]
    B --> E1
    C --> E2["sparse: LogReg, LinearSVC"]
    C --> E3["SVD dense: RandomForest, HistGB"]
    E1 --> GRID["12 experiments"]
    E2 --> GRID
    E3 --> GRID
    GRID --> VALM["validation macro-F1 ranking"]
    VALM --> BEST["best = exp008 skipgram + HistGB"]
    BEST --> TESTONCE["test once: macro-F1 0.8910"]
    TESTONCE --> ART["selected_model.joblib"]
```

## G. GUI inference sequence

```mermaid
sequenceDiagram
    actor User
    participant UI as streamlit_app.py
    participant NC as NewsClassifier
    participant PP as preprocessing
    participant DV as document_vectors
    participant MD as selected_model.joblib
    User->>UI: type headline, click Predict
    UI->>NC: get_classifier (cached)
    NC->>MD: load artifact
    NC->>PP: preprocess(headline)
    PP-->>NC: cleaned tokens
    NC->>DV: mean_pool_tokens (skipgram)
    DV-->>NC: 100-d vector
    NC->>MD: predict_proba
    MD-->>NC: scores
    NC-->>UI: label + scores
    UI->>User: result card + score bars
```

## H. Python module dependency graph

```mermaid
flowchart TD
    CFG["config.py"]
    DL["src/data_loader.py"] --> CFG
    DI["src/dataset_inspection.py"] --> DL
    PP["src/preprocessing.py"] --> DL
    WT["src/word2vec_trainer.py"] --> PP
    WT --> DL
    DV["src/document_vectors.py"] --> PP
    DV --> CFG
    TF["src/text_features.py"] --> DV
    TF --> CFG
    EV["src/evaluation.py"] --> CFG
    CL["src/classification.py"] --> TF
    CL --> EV
    PR["src/prediction.py"] --> CL
    PR --> PP
    PR --> DV
    APPS["app/streamlit_app.py"] --> PR
    APPS --> CFG
```
