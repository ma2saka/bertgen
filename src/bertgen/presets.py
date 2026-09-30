"""Base model presets and language-aware resolution."""

from enum import StrEnum


class Preset(StrEnum):
    EN_BASE = "en-base"
    EN_LARGE = "en-large"
    JA_70M = "ja-70m"
    JA_130M = "ja-130m"
    JA_310M = "ja-310m"
    LLM_JP_BASE = "llm-jp-base"
    MM_SMALL = "mm-small"
    MM_BASE = "mm-base"


class Size(StrEnum):
    BASE = "base"
    LARGE = "large"


PRESET_MODELS: dict[Preset, str] = {
    Preset.EN_BASE: "answerdotai/ModernBERT-base",
    Preset.EN_LARGE: "answerdotai/ModernBERT-large",
    Preset.JA_70M: "sbintuitions/modernbert-ja-70m",
    Preset.JA_130M: "sbintuitions/modernbert-ja-130m",
    Preset.JA_310M: "sbintuitions/modernbert-ja-310m",
    Preset.LLM_JP_BASE: "llm-jp/llm-jp-modernbert-base",
    Preset.MM_SMALL: "jhu-clsp/mmBERT-small",
    Preset.MM_BASE: "jhu-clsp/mmBERT-base",
}

_BY_LANGUAGE: dict[str, dict[Size, Preset]] = {
    "ja": {Size.BASE: Preset.LLM_JP_BASE, Size.LARGE: Preset.JA_310M},
    "en": {Size.BASE: Preset.EN_BASE, Size.LARGE: Preset.EN_LARGE},
}
_MULTILINGUAL = {Size.BASE: Preset.MM_SMALL, Size.LARGE: Preset.MM_BASE}


def parse_model_choice(choice: str) -> Preset | Size | str:
    """Classify `choice` as a preset, a size or a Hugging Face id (contains '/').

    Raises ValueError for anything else.
    """
    if "/" in choice:
        return choice
    for kind in (Preset, Size):
        try:
            return kind(choice)
        except ValueError:
            pass
    choices = ", ".join([*Size, *Preset])
    raise ValueError(f"unknown base model choice {choice!r}: use {choices} or a Hugging Face id")


def resolve_base_model(choice: str, language: str) -> str:
    """Map `base`/`large`, a preset key or an HF id to a Hugging Face model id."""
    match parse_model_choice(choice):
        case Preset() as preset:
            return PRESET_MODELS[preset]
        case Size() as size:
            primary = language.lower().split("-", 1)[0].split("_", 1)[0]
            return PRESET_MODELS[_BY_LANGUAGE.get(primary, _MULTILINGUAL)[size]]
        case model_id:
            return model_id
