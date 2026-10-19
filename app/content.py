"""Static UI content and pure presentation helpers for the Streamlit GUI.

This module is deliberately free of any Streamlit import so the category
mappings and score-formatting helpers can be unit tested quickly and reused by
the view layer (:mod:`app.streamlit_app`).

Nothing here performs inference or touches model artifacts; it only describes
the ten Phase 5 categories and formats values that the model already produced.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

# ---------------------------------------------------------------------------
# Category presentation metadata
# ---------------------------------------------------------------------------
# One entry per Phase 5 category (config.CATEGORY_LABELS). ``description`` is a
# short editorial explanation for the UI -- it is NOT a model output.
CATEGORY_INFO: Dict[str, Dict[str, str]] = {
    "State": {
        "icon": "🏙️",
        "description": "News about states, cities, and regional developments.",
    },
    "International": {
        "icon": "🌍",
        "description": "International affairs and events.",
    },
    "Entertainment": {
        "icon": "🎬",
        "description": "Cinema, television, celebrities, and entertainment.",
    },
    "Business": {
        "icon": "💼",
        "description": "Business, finance, markets, and the economy.",
    },
    "Nation": {
        "icon": "🏛️",
        "description": "National news and developments.",
    },
    "Technology": {
        "icon": "💻",
        "description": "Technology, gadgets, software, and innovation.",
    },
    "Sports": {
        "icon": "🏏",
        "description": "Sports events, teams, and athletes.",
    },
    "Astro": {
        "icon": "🔮",
        "description": "Astrology and horoscope-related news.",
    },
    "Crime": {
        "icon": "🚨",
        "description": "Crime, investigations, and law-enforcement news.",
    },
    "Auto": {
        "icon": "🚗",
        "description": "Automobiles, vehicles, and the automotive industry.",
    },
}

# Used when a label is not one of the ten known categories.
DEFAULT_CATEGORY_ICON = "🏷️"
DEFAULT_CATEGORY_DESCRIPTION = (
    "This category is not one of the ten labels the model was trained on."
)

# ---------------------------------------------------------------------------
# Verified project details shown in the model-information section
# ---------------------------------------------------------------------------
# Values mirror the frozen Phase 5 selected configuration (validation used for
# selection; test evaluated once afterwards). Do not invent new numbers here.
MODEL_DISPLAY: Dict[str, object] = {
    "embedding": "Word2Vec Skip-gram",
    "classifier": "HistGradientBoostingClassifier",
    "vector_dimension": 100,
    "categories": len(CATEGORY_INFO),
    "validation_macro_f1": 0.8883,
    "test_macro_f1": 0.8910,
    "test_accuracy": 0.8953,
}

# Human-readable methodology chain (presentation only).
METHODOLOGY_STEPS: List[str] = [
    "Gujarati headline",
    "Existing preprocessing",
    "Skip-gram document vector (100d)",
    "Trained classifier",
    "Predicted category",
]

# ---------------------------------------------------------------------------
# Theme tokens (light / dark palette)
# ---------------------------------------------------------------------------
# Streamlit applies its active theme via generated Emotion classes and does NOT
# expose global theme CSS variables -- but it DOES set the standard CSS
# ``color-scheme`` property (``light`` or ``dark``) on the app root
# (``[data-testid="stApp"]``). We therefore define every custom colour as a pair
# using the CSS ``light-dark()`` function, so our palette follows Streamlit's own
# Light/Dark selection with no separate, conflicting theme mechanism.
#
# Each token maps to ``(light_value, dark_value)``. Values must be real CSS
# *colours* (``light-dark()`` only accepts colours, not gradients/images).
THEME_TOKENS: Dict[str, Tuple[str, str]] = {
    # Surfaces
    "gnc-bg": ("#faf8f4", "#0f1720"),
    "gnc-surface": ("#ffffff", "#1a2433"),
    "gnc-surface-2": ("#f4f1ea", "#223041"),
    "gnc-hero-bg": ("#f7fdfb", "#18272b"),
    "gnc-result-bg": ("#f6fdfb", "#16292c"),
    # Text
    "gnc-ink": ("#14213d", "#e8edf5"),
    "gnc-muted": ("#5b6478", "#9aa7b8"),
    # Borders / shadow
    "gnc-border": ("#e7e2d8", "#2b3a4d"),
    "gnc-shadow": ("rgba(20, 33, 61, 0.06)", "rgba(0, 0, 0, 0.45)"),
    # Inputs
    "gnc-input-bg": ("#ffffff", "#16202e"),
    "gnc-input-text": ("#172033", "#f1f5f9"),
    "gnc-placeholder": ("#64748b", "#8b9bb0"),
    "gnc-focus": ("rgba(13, 148, 136, 0.30)", "rgba(45, 212, 191, 0.40)"),
    # Accents
    "gnc-teal": ("#0f766e", "#2dd4bf"),
    "gnc-teal-bright": ("#0d9488", "#14b8a6"),
    "gnc-badge-bg": ("rgba(13, 148, 136, 0.10)", "rgba(45, 212, 191, 0.14)"),
    "gnc-badge-border": ("rgba(13, 148, 136, 0.28)", "rgba(45, 212, 191, 0.34)"),
    "gnc-quote-bg": ("rgba(13, 148, 136, 0.06)", "rgba(45, 212, 191, 0.10)"),
    # Buttons: teal primary with white text; neutral secondary.
    "gnc-primary-bg": ("#0f766e", "#0f766e"),
    "gnc-primary-hover": ("#115e59", "#0d9488"),
    "gnc-primary-text": ("#ffffff", "#ffffff"),
    "gnc-secondary-bg": ("#f1f5f9", "#1f2b3a"),
    "gnc-secondary-text": ("#172033", "#e8edf5"),
    "gnc-secondary-border": ("#e2e8f0", "#33455a"),
    # Score bars
    "gnc-bar-track": ("#efeae0", "#22303f"),
    "gnc-bar-fill": ("#9ad3cd", "#134e4a"),
}

# Opaque (foreground, background) token pairs that must stay readable in BOTH
# themes. Used by the contrast tests to guard against regressions. All values
# referenced here are opaque hex colours.
CONTRAST_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("gnc-ink", "gnc-bg"),
    ("gnc-ink", "gnc-surface"),
    ("gnc-muted", "gnc-surface"),
    ("gnc-input-text", "gnc-input-bg"),
    ("gnc-primary-text", "gnc-primary-bg"),
    ("gnc-secondary-text", "gnc-secondary-bg"),
)


def build_theme_variables_css() -> str:
    """Build the CSS that defines every ``--gnc-*`` variable for both themes.

    Two blocks are emitted:

    1. A plain fallback block using the light values (so browsers without
       ``light-dark()`` still get a fully consistent, readable light palette
       instead of inheriting mismatched native colours).
    2. A ``@supports`` block that swaps in ``light-dark(light, dark)`` so the
       palette follows Streamlit's active ``color-scheme``.
    """
    fallback = [":root {"]
    for name, (light, _dark) in THEME_TOKENS.items():
        fallback.append(f"    --{name}: {light};")
    fallback.append("}")

    adaptive = [
        "@supports (color: light-dark(#ffffff, #000000)) {",
        "    :root {",
    ]
    for name, (light, dark) in THEME_TOKENS.items():
        adaptive.append(f"        --{name}: light-dark({light}, {dark});")
    adaptive.append("    }")
    adaptive.append("}")

    return "\n".join(fallback + [""] + adaptive)


# ---------------------------------------------------------------------------
# Color contrast helpers (used by tests; deterministic and dependency-free)
# ---------------------------------------------------------------------------

def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    """Parse ``#rgb``/``#rrggbb`` into an ``(r, g, b)`` tuple."""
    text = str(value).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"Not a hex colour: {value!r}")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def relative_luminance(value: str) -> float:
    """WCAG relative luminance of an opaque hex colour."""

    def channel(component: int) -> float:
        srgb = component / 255.0
        if srgb <= 0.03928:
            return srgb / 12.92
        return ((srgb + 0.055) / 1.055) ** 2.4

    r, g, b = _hex_to_rgb(value)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio between two opaque hex colours (1.0 to 21.0)."""
    lum_fg = relative_luminance(foreground)
    lum_bg = relative_luminance(background)
    lighter, darker = max(lum_fg, lum_bg), min(lum_fg, lum_bg)
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# Sample headlines (sample inputs, NOT verified ground-truth labels)
# ---------------------------------------------------------------------------
# Genuine Gujarati Unicode, short and natural. ``hint`` is a suggestion only and
# must not be presented as a guaranteed/verified label.
EXAMPLE_HEADLINES: List[Dict[str, str]] = [
    {
        "text": "ભારતે ઓસ્ટ્રેલિયા સામે ક્રિકેટ મેચ જીતી",
        "hint": "Sports",
    },
    {
        "text": "નવી સ્માર્ટફોન ટેકનોલોજી બજારમાં લોન્ચ થઈ",
        "hint": "Technology",
    },
    {
        "text": "શેર બજારમાં આજે જોરદાર તેજી જોવા મળી",
        "hint": "Business",
    },
    {
        "text": "બોલિવૂડ ફિલ્મે બોક્સ ઓફિસ પર રેકોર્ડ કમાણી કરી",
        "hint": "Entertainment",
    },
    {
        "text": "અમેરિકાના રાષ્ટ્રપતિએ નવી વિદેશ નીતિની જાહેરાત કરી",
        "hint": "International",
    },
]


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------

def get_category_info(label: str) -> Dict[str, str]:
    """Return ``{"icon", "description"}`` for ``label`` with a safe fallback."""
    info = CATEGORY_INFO.get(str(label))
    if info is None:
        return {
            "icon": DEFAULT_CATEGORY_ICON,
            "description": DEFAULT_CATEGORY_DESCRIPTION,
        }
    return {"icon": info["icon"], "description": info["description"]}


def category_icon(label: str) -> str:
    """Return the icon for ``label`` (fallback icon when unknown)."""
    return get_category_info(label)["icon"]


def category_description(label: str) -> str:
    """Return the one-line description for ``label`` (fallback when unknown)."""
    return get_category_info(label)["description"]


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def sorted_scores(scores: Mapping[str, float]) -> List[Tuple[str, float]]:
    """Return ``(label, score)`` pairs sorted by score descending.

    Sorting is presentation-only and deterministic: ties are broken by label so
    the order never flickers between reruns. The label-to-score mapping itself
    is never changed.
    """
    return sorted(
        ((str(label), float(value)) for label, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )


def classify_score_type(values: Sequence[float]) -> str:
    """Classify a score vector as ``"probability"`` or ``"decision"``.

    A vector is a probability estimate when every value is non-negative and the
    values sum to roughly one (as produced by ``predict_proba``). Anything else
    -- e.g. ``decision_function`` margins, which may be negative and do not sum
    to one -- is a ``"decision"`` score. Used so the UI labels the values by
    what they actually are rather than inventing percentages.
    """
    floats = [float(v) for v in values]
    if not floats:
        return "decision"
    if all(v >= 0.0 for v in floats) and abs(sum(floats) - 1.0) <= 1e-6:
        return "probability"
    return "decision"


def format_score(value: float, score_type: str) -> str:
    """Format one score for display according to its type."""
    value = float(value)
    if score_type == "probability":
        return f"{value * 100:.1f}%"
    return f"{value:+.3f}"


def score_to_bar_fraction(value: float, max_value: float) -> float:
    """Return ``value / max_value`` clamped to ``[0, 1]`` for bar widths."""
    max_value = float(max_value)
    if max_value <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(value) / max_value))
