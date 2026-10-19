"""Tests for app/content.py (Phase 6 GUI static content + helpers)."""

from __future__ import annotations

import config
from app.content import (
    CATEGORY_INFO,
    CONTRAST_PAIRS,
    DEFAULT_CATEGORY_DESCRIPTION,
    DEFAULT_CATEGORY_ICON,
    EXAMPLE_HEADLINES,
    METHODOLOGY_STEPS,
    MODEL_DISPLAY,
    THEME_TOKENS,
    build_theme_variables_css,
    category_description,
    category_icon,
    classify_score_type,
    contrast_ratio,
    format_score,
    get_category_info,
    relative_luminance,
    score_to_bar_fraction,
    sorted_scores,
)


def test_all_ten_categories_covered():
    assert set(CATEGORY_INFO) == set(config.CATEGORY_LABELS)
    assert len(CATEGORY_INFO) == 10


def test_every_category_has_icon_and_description():
    for label, info in CATEGORY_INFO.items():
        assert info["icon"].strip(), label
        assert info["description"].strip(), label


def test_unknown_label_returns_safe_fallback():
    info = get_category_info("NotACategory")
    assert info["icon"] == DEFAULT_CATEGORY_ICON
    assert info["description"] == DEFAULT_CATEGORY_DESCRIPTION
    assert category_icon("NotACategory") == DEFAULT_CATEGORY_ICON
    assert category_description("NotACategory") == DEFAULT_CATEGORY_DESCRIPTION


def test_known_label_helpers_match_mapping():
    assert category_icon("Sports") == CATEGORY_INFO["Sports"]["icon"]
    assert category_description("Sports") == CATEGORY_INFO["Sports"]["description"]


def test_sorted_scores_descending_and_preserves_keys():
    scores = {"State": 0.2, "Sports": 0.7, "Crime": 0.1}
    ranked = sorted_scores(scores)
    assert [label for label, _ in ranked] == ["Sports", "State", "Crime"]
    assert {label for label, _ in ranked} == set(scores)
    assert sum(value for _, value in ranked) == 0.2 + 0.7 + 0.1


def test_sorted_scores_is_deterministic_on_ties():
    scores = {"B": 0.5, "A": 0.5}
    assert [label for label, _ in sorted_scores(scores)] == ["A", "B"]


def test_sorted_scores_empty():
    assert sorted_scores({}) == []


def test_classify_score_type_probability():
    assert classify_score_type([0.7, 0.3]) == "probability"
    assert classify_score_type([0.2, 0.3, 0.5]) == "probability"


def test_classify_score_type_decision_for_margins():
    assert classify_score_type([1.2, -0.4, 0.1]) == "decision"
    assert classify_score_type([2.0, 1.0]) == "decision"


def test_classify_score_type_empty_is_decision():
    assert classify_score_type([]) == "decision"


def test_format_score_probability_and_decision():
    assert format_score(0.889, "probability") == "88.9%"
    assert format_score(0.889, "decision") == "+0.889"
    assert format_score(-0.25, "decision") == "-0.250"


def test_score_to_bar_fraction_clamps():
    assert score_to_bar_fraction(0.5, 1.0) == 0.5
    assert score_to_bar_fraction(2.0, 1.0) == 1.0
    assert score_to_bar_fraction(-1.0, 1.0) == 0.0
    assert score_to_bar_fraction(0.5, 0.0) == 0.0


def test_example_headlines_are_gujarati_and_labelled_with_known_hints():
    assert EXAMPLE_HEADLINES
    for example in EXAMPLE_HEADLINES:
        text = example["text"]
        assert text.strip()
        assert any("\u0A80" <= ch <= "\u0AFF" for ch in text), text
        assert example["hint"] in CATEGORY_INFO


def test_model_display_matches_frozen_metrics():
    assert MODEL_DISPLAY["classifier"] == "HistGradientBoostingClassifier"
    assert MODEL_DISPLAY["embedding"] == "Word2Vec Skip-gram"
    assert MODEL_DISPLAY["vector_dimension"] == 100
    assert MODEL_DISPLAY["categories"] == 10
    assert MODEL_DISPLAY["validation_macro_f1"] == 0.8883
    assert MODEL_DISPLAY["test_macro_f1"] == 0.8910
    assert MODEL_DISPLAY["test_accuracy"] == 0.8953


def test_methodology_steps_non_empty():
    assert len(METHODOLOGY_STEPS) >= 3
    assert all(step.strip() for step in METHODOLOGY_STEPS)


# ---------------------------------------------------------------------------
# Theme tokens + contrast
# ---------------------------------------------------------------------------
def test_theme_tokens_define_light_and_dark_pairs():
    assert THEME_TOKENS
    for name, pair in THEME_TOKENS.items():
        assert name.startswith("gnc-"), name
        assert isinstance(pair, tuple) and len(pair) == 2, name
        light, dark = pair
        assert light.strip() and dark.strip(), name


def test_theme_css_has_fallback_and_light_dark_supports_block():
    css = build_theme_variables_css()
    assert ":root {" in css
    assert "@supports (color: light-dark(" in css
    assert "light-dark(" in css
    for name in THEME_TOKENS:
        # Defined once in the fallback block and once inside @supports.
        assert css.count(f"--{name}:") == 2, name


def test_theme_css_fallback_uses_light_values_only():
    css = build_theme_variables_css()
    fallback = css.split("@supports", 1)[0]
    for name, (light, _dark) in THEME_TOKENS.items():
        assert f"--{name}: {light};" in fallback, name


def test_contrast_pairs_reference_known_tokens():
    for foreground, background in CONTRAST_PAIRS:
        assert foreground in THEME_TOKENS, foreground
        assert background in THEME_TOKENS, background


def test_contrast_pairs_meet_wcag_aa_in_both_themes():
    for foreground, background in CONTRAST_PAIRS:
        light_fg, dark_fg = THEME_TOKENS[foreground]
        light_bg, dark_bg = THEME_TOKENS[background]
        ratio_light = contrast_ratio(light_fg, light_bg)
        ratio_dark = contrast_ratio(dark_fg, dark_bg)
        assert ratio_light >= 4.5, (foreground, background, "light", ratio_light)
        assert ratio_dark >= 4.5, (foreground, background, "dark", ratio_dark)


def test_primary_button_text_is_readable_on_primary_fill():
    for theme_index in (0, 1):
        fg = THEME_TOKENS["gnc-primary-text"][theme_index]
        bg = THEME_TOKENS["gnc-primary-bg"][theme_index]
        assert contrast_ratio(fg, bg) >= 4.5


def test_contrast_ratio_bounds_and_formula():
    assert contrast_ratio("#000000", "#ffffff") == 21.0
    assert contrast_ratio("#ffffff", "#ffffff") == 1.0
    assert contrast_ratio("#ffffff", "#000000") == 21.0


def test_relative_luminance_is_monotonic_and_bounded():
    assert relative_luminance("#000000") == 0.0
    assert relative_luminance("#ffffff") == 1.0
    assert relative_luminance("#808080") < relative_luminance("#ffffff")


def test_hex_parsing_accepts_short_form_and_rejects_garbage():
    assert contrast_ratio("#fff", "#000") == 21.0
    for bad in ("not-a-colour", "#12", "#12345", "#gggggg"):
        try:
            relative_luminance(bad)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {bad!r}")
