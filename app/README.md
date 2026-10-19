# Gujarati News Classifier — Streamlit GUI (Phase 6)

An interactive web interface for the existing Gujarati News Classification
project. It loads the frozen Phase 5 selected model and predicts which of the ten
trained categories a Gujarati news headline belongs to, using the **same**
preprocessing and document-vector pipeline the model was trained with. Nothing is
retrained and no artifacts are modified.

## What the GUI does

- Accepts a Gujarati headline (UTF-8) and predicts its category.
- Shows the predicted category with an icon and a short description.
- Displays the per-class scores returned by the model, labelled by their real
  type (probability estimates vs. decision scores).
- Explains the model, its methodology, and the verified validation/test metrics.
- Offers sample Gujarati headlines for quick testing.

The GUI **calls the existing inference pipeline** (`src.prediction.NewsClassifier`)
and does not implement a separate feature pipeline.

## Main features

- Editorial/newsroom visual style: warm off-white background, navy typography,
  teal accents, rounded white cards.
- Gujarati-safe font stack with graceful fallbacks (no external font download).
- Cached model loading (`@st.cache_resource`) so the model is loaded once.
- Input validation for empty/whitespace-only headlines.
- Graceful error messages for missing artifacts and failed inference.
- All ten categories handled, with a safe fallback for unknown labels.
- Responsive layout for desktop and mobile.
- Theme-aware colours that follow Streamlit's own Light/Dark setting, so text
  and controls stay readable in either mode.

## Required artifacts

These files must already exist (produced by earlier phases). The GUI never
creates or modifies them:

- `models/classifiers/selected_model.joblib` — frozen selected classifier
  (Word2Vec Skip-gram document vectors + `HistGradientBoostingClassifier`).
- `models/word2vec_skipgram.model` (+ its `.npy` sidecars) — Skip-gram embeddings
  used to build the document vectors for inference.

## Setup

Use the existing project virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`streamlit`, `numpy`, `scikit-learn`, `joblib`, and `gensim` are already listed in
`requirements.txt`; no new dependencies are required.

## Run the application

From the **project root**:

```powershell
streamlit run app/streamlit_app.py
```

(or, using the project virtual environment explicitly:)

```powershell
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Then open the local URL printed by Streamlit (default `http://localhost:8501`).

## Troubleshooting

- **"The trained model could not be found."** — Confirm
  `models/classifiers/selected_model.joblib` and `models/word2vec_skipgram.model`
  exist relative to the project root.
- **"Input headline has no usable tokens after preprocessing."** — The headline
  contained only punctuation/URLs/etc. Enter normal Gujarati text.
- **Gujarati text shows boxes (tofu).** — Install a Gujarati-capable font such as
  Noto Sans Gujarati or Nirmala UI; the app already falls back through several
  system fonts.
- **ModuleNotFoundError: config / src** — Run the command from the project root.
  The app also adds the project root to `sys.path` automatically.
- **First prediction is slow.** — The Word2Vec model is loaded on first use and
  then cached for the session.

## Theming (light/dark)

Streamlit does not expose global theme CSS variables, but it does set the standard
`color-scheme` property (`light` or `dark`) on the app root. The GUI therefore
defines every custom colour as a pair via `build_theme_variables_css()`
(`app/content.py`), emitted as `--gnc-*` variables using the CSS `light-dark()`
function. The palette follows Streamlit's own **Light/Dark** selection (set under
the app menu → Settings → Theme); there is no separate in-app theme toggle.

Every styled control sets both its foreground and background colour, so native
theme colours cannot leak in and produce low-contrast text. A plain `:root`
fallback block (light values) is included for browsers without `light-dark()`
support, so the palette stays internally consistent there too. Pairwise contrast
is guarded by tests in `tests/test_app_content.py`.

## Notes and limitations

- Predictions are model estimates, not guarantees; the model can be wrong.
- Validation metrics were used for model selection; the test split was evaluated
  once after the configuration was frozen. Metrics describe this dataset only.
- The app is an educational NLP demonstration and runs fully offline.
