"""Premium Streamlit GUI for the Gujarati News Classifier (Phase 6).

This view layer never retrains or rewrites anything. It loads the frozen Phase 5
selected model through :class:`src.prediction.NewsClassifier` and reports the
category and class scores that the existing inference pipeline already produces.

Run from the project root::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Make the project root importable no matter where the app is launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

import config  # noqa: E402
from app.content import (  # noqa: E402
    CATEGORY_INFO,
    EXAMPLE_HEADLINES,
    METHODOLOGY_STEPS,
    MODEL_DISPLAY,
    build_theme_variables_css,
    classify_score_type,
    format_score,
    get_category_info,
    score_to_bar_fraction,
    sorted_scores,
)

SELECTED_MODEL_PATH = config.CLASSIFIERS_DIR / "selected_model.joblib"

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Gujarati News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Editorial / newsroom styling.
#
# Theme handling: Streamlit does not expose global theme CSS variables, but it
# sets the standard CSS ``color-scheme`` property (light/dark) on the app root.
# ``build_theme_variables_css()`` therefore defines every ``--gnc-*`` colour via
# ``light-dark(light, dark)`` so the palette follows Streamlit's own Light/Dark
# selection. Every visible control we style sets BOTH foreground and background
# explicitly, so no native theme colour can leak in and produce low-contrast
# (e.g. white-on-white) text.
#
# ``!important`` is used only where a Streamlit/BaseWeb generated style would
# otherwise win:
#   * app/view background + text colour (guarantees our surface/ink win);
#   * the textarea placeholder colour (BaseWeb sets its own);
#   * the workspace container border colour (Streamlit paints its own border).
# ---------------------------------------------------------------------------
CSS = """
/* ---- App surfaces ---- */
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: var(--gnc-bg) !important;
    color: var(--gnc-ink) !important;
}
html, body, [data-testid="stApp"], textarea, input, button, select {
    font-family: "Noto Sans Gujarati", "Nirmala UI", "Shruti",
        "Gujarati Sangam MN", "Segoe UI", system-ui, -apple-system, Arial,
        sans-serif;
}
.block-container {
    padding-top: 2.4rem;
    padding-bottom: 3rem;
    max-width: 1120px;
}
h1, h2, h3, h4 { color: var(--gnc-ink); }

/* ---- Hero ---- */
.gnc-hero {
    background: var(--gnc-hero-bg);
    border: 1px solid var(--gnc-border);
    border-radius: 22px;
    padding: 34px 36px 30px;
    box-shadow: 0 12px 34px var(--gnc-shadow);
}
.gnc-badge {
    display: inline-block;
    background: var(--gnc-badge-bg);
    color: var(--gnc-teal);
    border: 1px solid var(--gnc-badge-border);
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.gnc-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.55rem 0 0.15rem;
    color: var(--gnc-ink);
}
.gnc-subtitle {
    color: var(--gnc-muted);
    font-size: 1.05rem;
    margin: 0 0 0.75rem;
}
.gnc-hero-note {
    color: var(--gnc-muted);
    font-size: 0.92rem;
    max-width: 760px;
    line-height: 1.55;
}

/* ---- Workspace card (Streamlit bordered container) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--gnc-surface);
    border: 1px solid var(--gnc-border) !important;
    border-radius: 18px;
    box-shadow: 0 8px 24px var(--gnc-shadow);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.6rem 1.1rem 1rem;
}

/* ---- Generic card + section heading ---- */
.gnc-card {
    background: var(--gnc-surface);
    border: 1px solid var(--gnc-border);
    border-radius: 18px;
    padding: 24px 26px;
    box-shadow: 0 8px 24px var(--gnc-shadow);
}
.gnc-section-title {
    font-size: 1.3rem;
    font-weight: 750;
    margin: 0 0 0.25rem;
    color: var(--gnc-ink);
}
.gnc-section-sub {
    color: var(--gnc-muted);
    font-size: 0.92rem;
    margin: 0 0 1rem;
}

/* ---- Buttons ---- */
div[data-testid="stButton"] > button {
    background: var(--gnc-secondary-bg);
    color: var(--gnc-secondary-text);
    border: 1px solid var(--gnc-secondary-border);
    border-radius: 12px;
    font-weight: 650;
    padding: 0.5rem 1.1rem;
}
div[data-testid="stButton"] > button:hover {
    border-color: var(--gnc-teal-bright);
    color: var(--gnc-teal);
}
div[data-testid="stButton"] > button:focus-visible {
    border-color: var(--gnc-teal-bright);
    box-shadow: 0 0 0 3px var(--gnc-focus);
    outline: none;
}
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] {
    background: var(--gnc-primary-bg);
    border-color: var(--gnc-primary-bg);
    color: var(--gnc-primary-text);
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover {
    background: var(--gnc-primary-hover);
    border-color: var(--gnc-primary-hover);
    color: var(--gnc-primary-text);
}
div[data-testid="stButton"] > button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
}

/* ---- Gujarati headline text area ---- */
div[data-testid="stTextArea"] div[data-baseweb="base-input"],
div[data-testid="stTextArea"] div[data-baseweb="textarea"] {
    background: var(--gnc-input-bg);
    border-color: var(--gnc-border);
}
div[data-testid="stTextArea"] textarea {
    background: var(--gnc-input-bg);
    color: var(--gnc-input-text);
    caret-color: var(--gnc-input-text);
    border: 1px solid var(--gnc-border);
    border-radius: 14px;
    font-size: 1.08rem;
    line-height: 1.7;
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: var(--gnc-placeholder) !important;
    opacity: 1;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--gnc-teal-bright);
    box-shadow: 0 0 0 3px var(--gnc-focus);
    outline: none;
}
.gnc-count {
    color: var(--gnc-muted);
    font-size: 0.82rem;
    margin-top: 0.35rem;
}

/* ---- Selectbox ---- */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: var(--gnc-input-bg);
    border-color: var(--gnc-border);
    color: var(--gnc-input-text);
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
    fill: var(--gnc-input-text);
}

/* ---- Labels, captions ---- */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label {
    color: var(--gnc-ink);
}
div[data-testid="stCaptionContainer"] { color: var(--gnc-muted); }
div[data-testid="stCaptionContainer"] p { color: var(--gnc-muted); }

/* ---- Expander ---- */
div[data-testid="stExpander"] details {
    background: var(--gnc-surface);
    border: 1px solid var(--gnc-border);
    border-radius: 14px;
}
div[data-testid="stExpander"] summary { color: var(--gnc-ink); }
div[data-testid="stExpander"] summary svg { fill: var(--gnc-ink); }
div[data-testid="stExpander"] p { color: var(--gnc-ink); }

/* ---- Alerts (validation messages) ---- */
div[data-testid="stAlert"] {
    background: var(--gnc-surface-2);
    color: var(--gnc-ink);
    border-radius: 12px;
}
div[data-testid="stAlert"] p { color: var(--gnc-ink); }

code {
    background: var(--gnc-surface-2);
    color: var(--gnc-ink);
}

/* ---- Result card ---- */
.gnc-result {
    background: var(--gnc-result-bg);
    border: 1px solid var(--gnc-badge-border);
    border-radius: 20px;
    padding: 26px 28px;
    box-shadow: 0 12px 30px var(--gnc-shadow);
}
.gnc-result-icon { font-size: 2.6rem; line-height: 1; }
.gnc-result-label {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--gnc-teal);
}
.gnc-result-category {
    font-size: 2rem;
    font-weight: 800;
    margin: 0.1rem 0 0.2rem;
    color: var(--gnc-ink);
}
.gnc-result-desc { color: var(--gnc-muted); font-size: 0.98rem; }
.gnc-result-quote {
    margin-top: 1rem;
    padding: 12px 16px;
    border-left: 3px solid var(--gnc-teal-bright);
    background: var(--gnc-quote-bg);
    border-radius: 8px;
    font-size: 1.02rem;
    color: var(--gnc-ink);
}
.gnc-fineprint {
    color: var(--gnc-muted);
    font-size: 0.82rem;
    margin-top: 1rem;
}

/* ---- Score bars ---- */
.gnc-score-row { margin-bottom: 0.7rem; }
.gnc-score-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    color: var(--gnc-ink);
    margin-bottom: 0.28rem;
}
.gnc-score-val { color: var(--gnc-muted); font-variant-numeric: tabular-nums; }
.gnc-bar-track {
    background: var(--gnc-bar-track);
    border-radius: 999px;
    height: 9px;
    overflow: hidden;
}
.gnc-bar-fill {
    background: var(--gnc-bar-fill);
    height: 100%;
    border-radius: 999px;
}
.gnc-bar-top { background: var(--gnc-teal-bright); }

/* ---- Category grid ---- */
.gnc-cat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(196px, 1fr));
    gap: 14px;
}
.gnc-cat-card {
    background: var(--gnc-surface);
    border: 1px solid var(--gnc-border);
    border-radius: 16px;
    padding: 18px 18px 16px;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.gnc-cat-card:hover {
    border-color: var(--gnc-teal-bright);
    transform: translateY(-2px);
}
.gnc-cat-icon { font-size: 1.6rem; }
.gnc-cat-name {
    font-weight: 750;
    margin: 0.35rem 0 0.2rem;
    color: var(--gnc-ink);
}
.gnc-cat-desc { color: var(--gnc-muted); font-size: 0.86rem; line-height: 1.45; }

/* ---- Model info grid ---- */
.gnc-info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 12px;
    margin-bottom: 1rem;
}
.gnc-info-tile {
    background: var(--gnc-surface);
    border: 1px solid var(--gnc-border);
    border-radius: 14px;
    padding: 14px 16px;
}
.gnc-info-key {
    color: var(--gnc-muted);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.gnc-info-value {
    color: var(--gnc-ink);
    font-size: 1.28rem;
    font-weight: 750;
    margin-top: 0.2rem;
}

/* ---- Methodology sequence ---- */
.gnc-flow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
}
.gnc-flow-step {
    background: var(--gnc-surface);
    border: 1px solid var(--gnc-border);
    border-radius: 999px;
    padding: 7px 14px;
    font-size: 0.86rem;
    color: var(--gnc-ink);
}
.gnc-flow-arrow { color: var(--gnc-teal-bright); font-weight: 800; }

/* ---- Footer ---- */
.gnc-footer {
    margin-top: 2.2rem;
    padding-top: 1.3rem;
    border-top: 1px solid var(--gnc-border);
    text-align: center;
    color: var(--gnc-muted);
    font-size: 0.84rem;
    line-height: 1.6;
}
.gnc-footer strong { color: var(--gnc-ink); }
"""

STYLE_BLOCK = f"<style>\n{build_theme_variables_css()}\n{CSS}\n</style>"
st.markdown(STYLE_BLOCK, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading (cached, read-only)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_classifier():
    """Load the frozen Phase 5 selected model once per session (read-only)."""
    from src.prediction import NewsClassifier

    return NewsClassifier.load()


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("headline_text", "")
    st.session_state.setdefault("result", None)


def _reset_state() -> None:
    """Clear the input box and any displayed prediction (button callback)."""
    st.session_state.headline_text = ""
    st.session_state.result = None


def _load_example(text: str) -> None:
    """Fill the input box with a sample headline (never predicts)."""
    st.session_state.headline_text = text


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_hero() -> None:
    st.markdown(
        """
        <div class="gnc-hero">
            <span class="gnc-badge">Word2Vec + Machine Learning</span>
            <div class="gnc-title">Gujarati News Classifier</div>
            <div class="gnc-subtitle">
                AI-powered Gujarati news category prediction using Word2Vec and
                Machine Learning.
            </div>
            <div class="gnc-hero-note">
                Paste a Gujarati news headline below and the app predicts which of
                the ten trained categories it belongs to &mdash; using the same
                preprocessing and document-vector pipeline the model was trained
                with. No data leaves your machine.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_examples() -> None:
    options = list(range(len(EXAMPLE_HEADLINES)))

    def _label(index: int) -> str:
        example = EXAMPLE_HEADLINES[index]
        return f"{example['hint']}  ·  {example['text']}"

    with st.expander("Try a sample headline", expanded=False):
        st.caption(
            "Sample inputs only — the suggested category is a hint, not a "
            "verified ground-truth label."
        )
        choice = st.selectbox(
            "Sample headlines",
            options=options,
            format_func=_label,
            index=None,
            placeholder="Choose a sample headline…",
        )
        chosen_text = EXAMPLE_HEADLINES[choice]["text"] if choice is not None else ""
        st.button(
            "Load sample headline",
            on_click=_load_example,
            args=(chosen_text,),
            disabled=choice is None,
            key="load_sample",
        )


def render_result(result: Mapping[str, object]) -> None:
    label = str(result["label"])
    headline = str(result["headline"])
    info = get_category_info(label)
    icon = html.escape(info["icon"])
    description = html.escape(info["description"])
    safe_headline = html.escape(headline)
    safe_label = html.escape(label)

    st.markdown(
        f"""
        <div class="gnc-result">
            <div class="gnc-result-icon">{icon}</div>
            <div class="gnc-result-label">Predicted category</div>
            <div class="gnc-result-category">{safe_label}</div>
            <div class="gnc-result-desc">{description}</div>
            <div class="gnc-result-quote">“{safe_headline}”</div>
            <div class="gnc-fineprint">
                This is a model prediction, not a guarantee. The classifier can be
                wrong, especially on short, ambiguous, or out-of-domain headlines.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scores(scores: Mapping[str, float]) -> None:
    ranked = sorted_scores(scores)
    score_type = classify_score_type(list(scores.values()))
    max_value = max((value for _, value in ranked), default=0.0)
    top_label = ranked[0][0] if ranked else None

    if score_type == "probability":
        note = (
            "Values are <strong>probability estimates</strong> from the "
            "classifier's <code>predict_proba</code>. They sum to ~1.0 but are "
            "not calibrated guarantees of correctness."
        )
    else:
        note = (
            "Values are raw <strong>decision-function scores</strong> (margins), "
            "not probabilities or percentages."
        )

    rows: List[str] = []
    for label, value in ranked:
        info = get_category_info(label)
        fraction = score_to_bar_fraction(value, max_value)
        width = max(2.0, fraction * 100.0) if value > 0 else 0.0
        bar_class = "gnc-bar-fill gnc-bar-top" if label == top_label else "gnc-bar-fill"
        rows.append(
            '<div class="gnc-score-row">'
            f'<div class="gnc-score-head"><span>{html.escape(info["icon"])} '
            f'{html.escape(label)}</span>'
            f'<span class="gnc-score-val">{format_score(value, score_type)}</span>'
            "</div>"
            f'<div class="gnc-bar-track"><div class="{bar_class}" '
            f'style="width:{width:.2f}%"></div></div>'
            "</div>"
        )

    st.markdown(
        '<div class="gnc-card">'
        '<div class="gnc-section-title">Class scores</div>'
        f'<div class="gnc-section-sub">{note}</div>'
        f'{"".join(rows)}'
        "</div>",
        unsafe_allow_html=True,
    )


def render_workspace() -> Tuple[str, bool, bool]:
    """Render the input card and return (text, predict_clicked, clear_clicked)."""
    with st.container(border=True):
        st.markdown(
            '<div class="gnc-section-title">Predict a headline</div>'
            '<div class="gnc-section-sub">'
            "Enter a Gujarati news headline and predict its category."
            "</div>",
            unsafe_allow_html=True,
        )

        text = st.text_area(
            "Gujarati headline",
            key="headline_text",
            height=120,
            placeholder="ભારતે ઓસ્ટ્રેલિયા સામે ક્રિકેટ મેચ જીતી",
            label_visibility="collapsed",
        )
        stripped = (text or "").strip()
        st.markdown(
            f'<div class="gnc-count">{len(text or "")} characters · '
            f'{len(stripped.split())} words</div>',
            unsafe_allow_html=True,
        )

        col_predict, col_clear, _ = st.columns([1.3, 1, 3.2])
        with col_predict:
            predict_clicked = st.button(
                "Predict Category", type="primary", use_container_width=True
            )
        with col_clear:
            clear_clicked = st.button(
                "Clear", on_click=_reset_state, use_container_width=True
            )

    return text or "", predict_clicked, clear_clicked


def render_categories() -> None:
    cards = []
    for label, info in CATEGORY_INFO.items():
        cards.append(
            '<div class="gnc-cat-card">'
            f'<div class="gnc-cat-icon">{html.escape(info["icon"])}</div>'
            f'<div class="gnc-cat-name">{html.escape(label)}</div>'
            f'<div class="gnc-cat-desc">{html.escape(info["description"])}</div>'
            "</div>"
        )
    st.markdown(
        '<div class="gnc-section-title">Explore the 10 categories</div>'
        '<div class="gnc-section-sub">The ten labels this model was trained to '
        "distinguish.</div>"
        f'<div class="gnc-cat-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_model_info() -> None:
    info_tiles = [
        ("Embedding", str(MODEL_DISPLAY["embedding"])),
        ("Classifier", str(MODEL_DISPLAY["classifier"])),
        ("Document vectors", f'{MODEL_DISPLAY["vector_dimension"]} dims'),
        ("Categories", str(MODEL_DISPLAY["categories"])),
        ("Validation macro-F1", f'{MODEL_DISPLAY["validation_macro_f1"] * 100:.2f}%'),
        ("Test macro-F1", f'{MODEL_DISPLAY["test_macro_f1"] * 100:.2f}%'),
        ("Test accuracy", f'{MODEL_DISPLAY["test_accuracy"] * 100:.2f}%'),
    ]
    tiles_html = "".join(
        f'<div class="gnc-info-tile"><div class="gnc-info-key">{html.escape(key)}</div>'
        f'<div class="gnc-info-value">{html.escape(value)}</div></div>'
        for key, value in info_tiles
    )
    flow = '<span class="gnc-flow-arrow">→</span>'.join(
        f'<span class="gnc-flow-step">{html.escape(step)}</span>'
        for step in METHODOLOGY_STEPS
    )
    st.markdown(
        '<div class="gnc-card">'
        '<div class="gnc-section-title">Model information</div>'
        '<div class="gnc-section-sub">'
        "The selected configuration was chosen on the validation split; the test "
        "split was evaluated once after the configuration was frozen."
        "</div>"
        f'<div class="gnc-info-grid">{tiles_html}</div>'
        '<div class="gnc-section-title" style="font-size:1.05rem;">Methodology</div>'
        f'<div class="gnc-flow">{flow}</div>'
        '<div class="gnc-fineprint">'
        "Metrics describe performance on this dataset and protocol only; they do "
        "not imply perfect accuracy or guarantee real-world results."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        '<div class="gnc-footer">'
        "<strong>Gujarati News Classification | Word2Vec + Machine Learning</strong>"
        "<br/>An educational NLP demonstration. Predictions may be incorrect."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _predict(headline: str) -> Optional[Dict[str, object]]:
    """Validate input and run the existing model pipeline. Returns result dict.

    Displays a friendly message and returns ``None`` on any handled failure.
    """
    if not headline.strip():
        st.warning("Please enter a Gujarati headline before predicting.")
        return None

    try:
        classifier = get_classifier()
    except FileNotFoundError:
        st.error(
            "The trained model could not be found. Make sure "
            "`models/classifiers/selected_model.joblib` and "
            "`models/word2vec_skipgram.model` exist."
        )
        return None
    except Exception:
        st.error(
            "The model could not be loaded. Please check that the saved "
            "artifacts are present and intact."
        )
        return None

    try:
        label, scores = classifier.predict_with_scores(headline)
    except ValueError as exc:
        st.warning(str(exc))
        return None
    except Exception:
        st.error(
            "Prediction failed unexpectedly. Please try a different Gujarati "
            "headline."
        )
        return None

    return {"label": label, "scores": dict(scores), "headline": headline}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    _init_state()
    render_hero()
    st.write("")

    headline, predict_clicked, _clear_clicked = render_workspace()

    if predict_clicked:
        result = _predict(headline)
        if result is not None:
            st.session_state.result = result

    render_examples()

    if st.session_state.get("result") is not None:
        st.write("")
        result = st.session_state.result
        render_result(result)
        render_scores(result.get("scores", {}))

    st.write("")
    render_categories()
    st.write("")
    render_model_info()
    render_footer()


if __name__ == "__main__":
    main()
