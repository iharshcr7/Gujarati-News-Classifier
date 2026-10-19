# Gujarati News Classifier

An end-to-end machine-learning project that classifies Gujarati news headlines
into one of ten categories. It includes data-inspection and preprocessing
pipelines, Word2Vec and TF-IDF features, scikit-learn classification
experiments, evaluation and error-analysis reports, and an interactive
Streamlit app for trying the trained classifier.

The project uses classical NLP and machine learning rather than deep-learning
models. Inference runs locally; no API key or paid service is required.

## Categories

The classifier predicts one of these categories:

**State, International, Entertainment, Business, Nation, Technology, Sports,
Astro, Crime, Auto**

## Results

The selected model uses 100-dimensional Skip-gram document vectors and a
gradient-boosting classifier. On the held-out test split, the project reports:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8953 |
| Macro precision | 0.8966 |
| Macro recall | 0.8862 |
| Macro F1 | 0.8910 |
| Weighted F1 | 0.8952 |

These results describe this dataset and are not a guarantee of performance on
other news sources. More details are in
[`docs/project_handbook/01_overview_and_goals.md`](docs/project_handbook/01_overview_and_goals.md)
and the evaluation files under `reports/`.

## Requirements

- Python 3.11 is the version documented and used for this project.
- pip
- The dataset and trained model artifacts described below, if you want to run
  the app or reproduce the experiments.

## Installation

Run these commands from the project root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell does not allow environment activation, invoke the environment's
Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Streamlit app

The app accepts a Gujarati headline and displays its predicted category and
class scores:

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit (usually
`http://localhost:8501`).

The app uses the existing inference pipeline in `src/prediction.py`. It requires
these trained artifacts in the project:

- `models/classifiers/selected_model.joblib`
- `models/word2vec_skipgram.model`
- The Word2Vec `.npy` sidecar files next to the model file

**Datasets and trained model artifacts are not included in this GitHub
repository.** They are excluded because some files are very large. Consequently,
a fresh clone needs those files supplied separately before the app can make
predictions. The project does not provide a dataset or model download URL.

For additional app details and troubleshooting, see
[`app/README.md`](app/README.md).

## Reproduce the pipeline

The data and model artifacts are also required for these steps. Place the raw
CSV files in `data/raw/` with the filenames expected by `config.py`:

- `Gujarati_SHC_LDC_LPC_Train.csv`
- `Gujarati_SHC_LDC_LPC_Valid.csv`
- `Gujarati_SHC_LDC_LPC_Test.csv`

Each CSV is expected to contain `text` and `labels` columns. With the files in
place, run the scripts in order from the project root:

```powershell
python inspect_dataset.py
python preprocess_dataset.py
python train_word2vec.py
python generate_document_vectors.py
python run_classification.py
python run_error_analysis.py
python run_model_improvement.py
```

Training and experiment steps can take significant time and write model and
report artifacts. Some scripts overwrite their existing outputs, so back up any
artifacts you want to keep before rerunning them. For an explanation of each
phase and its outputs, see
[`docs/project_handbook/10_testing_and_operations.md`](docs/project_handbook/10_testing_and_operations.md).

## Run tests

Install the project dependencies, then run:

```powershell
python -m pytest -q
```

The suite covers configuration, data loading, preprocessing, Word2Vec,
document vectors, classification, prediction, evaluation, error analysis,
experiments, and app content.

## Project structure

```text
app/                 Streamlit user interface
src/                 Data, feature, training, evaluation, and prediction code
tests/               Pytest test suite
docs/                Project handbook and architecture documentation
documentation/       PDF reports and diagrams
data/raw/            Local input datasets (not tracked)
data/processed/      Generated data (not tracked)
models/              Trained model artifacts (not tracked)
reports/             Evaluation and experiment reports
```
<img width="1578" height="877" alt="image" src="https://github.com/user-attachments/assets/43381c10-3ff2-4abe-b9e7-c5ea0baf9670" />

<img width="581" height="427" alt="image" src="https://github.com/user-attachments/assets/101608f9-761a-4f9b-bf81-736dc8304b5a" />

For a detailed architecture, dataset description, methodology, and glossary,
start with the
[project handbook](docs/project_handbook/README.md).
