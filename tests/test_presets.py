import pytest

from bertgen.presets import PRESET_MODELS, Preset, Size, parse_model_choice, resolve_base_model


@pytest.mark.parametrize(
    ("choice", "language", "expected"),
    [
        ("base", "ja", "llm-jp/llm-jp-modernbert-base"),
        ("llm-jp-base", "en", "llm-jp/llm-jp-modernbert-base"),
        ("ja-130m", "ja", "sbintuitions/modernbert-ja-130m"),
        ("large", "ja-JP", "sbintuitions/modernbert-ja-310m"),
        ("base", "en", "answerdotai/ModernBERT-base"),
        ("large", "EN_us", "answerdotai/ModernBERT-large"),
        ("base", "fr", "jhu-clsp/mmBERT-small"),
        ("large", "de", "jhu-clsp/mmBERT-base"),
        ("ja-70m", "en", "sbintuitions/modernbert-ja-70m"),
        ("mm-base", "ja", "jhu-clsp/mmBERT-base"),
        ("someorg/some-model", "ja", "someorg/some-model"),
    ],
)
def test_resolve(choice: str, language: str, expected: str) -> None:
    assert resolve_base_model(choice, language) == expected


def test_unknown_choice() -> None:
    with pytest.raises(ValueError, match="unknown"):
        resolve_base_model("huge", "en")


def test_parse_model_choice() -> None:
    assert parse_model_choice("large") is Size.LARGE
    assert parse_model_choice("ja-70m") is Preset.JA_70M
    assert parse_model_choice("llm-jp-base") is Preset.LLM_JP_BASE
    assert parse_model_choice("org/model") == "org/model"
    with pytest.raises(ValueError, match="ja-130m"):
        parse_model_choice("ja-130")


def test_every_preset_has_model() -> None:
    assert set(PRESET_MODELS) == set(Preset)
