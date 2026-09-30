"""Stage 2: plan the synthetic dataset."""

import logging
import re
from typing import Annotated

from pydantic import BaseModel, Field

from bertgen.llm.base import LLM, LLMError, WebResearcher
from bertgen.stages.prompts import (
    EXAMPLES_MARKER,
    MAX_SEEDS,
    PLAN_SYSTEM,
    plan_user,
    research_question,
)
from bertgen.text import normalize
from bertgen.types import NONE_KEY, Axis, DataPlan, TaskKind, TaskSpec

log = logging.getLogger(__name__)

_MIN_AXIS_VALUES = 3
_MAX_AXIS_VALUES = 8
MAX_SEED_CHARS = 200
_EXAMPLES_LINE = re.compile(rf"^\W*{EXAMPLES_MARKER}\W*$", re.MULTILINE)


class PlanDraft(BaseModel):
    """The plan as the LLM writes it, before normalization."""

    label_weights: dict[str, float]
    axes: list[Axis]
    hard_cases: Annotated[list[str], Field(description="near-miss or confusable cases")]
    style_notes: str
    seeds: list[str] = []


def make_plan(spec: TaskSpec, llm: LLM, researcher: WebResearcher | None) -> DataPlan:
    """Plan label mix, variation axes and hard cases, optionally informed by web research."""
    research = _research(spec, researcher)
    notes = _EXAMPLES_LINE.split(research, maxsplit=1)[0].strip()
    draft = llm.ask(PLAN_SYSTEM, plan_user(spec, research), PlanDraft)
    return DataPlan(
        label_weights=normalize_weights(spec, draft.label_weights),
        axes=_clean_axes(draft.axes),
        hard_cases=[case.strip() for case in draft.hard_cases if case.strip()],
        style_notes=draft.style_notes.strip(),
        research_notes=notes,
        seeds=_clean_seeds(draft.seeds) if research else [],
    )


def _research(spec: TaskSpec, researcher: WebResearcher | None) -> str:
    if researcher is None:
        return ""
    try:
        return researcher.research(research_question(spec)).strip()
    except LLMError as error:
        log.warning("research failed, planning without it: %s", error)
        return ""


def _clean_seeds(seeds: list[str]) -> list[str]:
    """Trimmed, deduplicated seeds within the length cap, at most `MAX_SEEDS`."""
    kept: dict[str, str] = {}
    for seed in seeds:
        kept.setdefault(normalize(seed), seed.strip())
    return [seed for seed in kept.values() if 0 < len(seed) <= MAX_SEED_CHARS][:MAX_SEEDS]


def normalize_weights(spec: TaskSpec, raw: dict[str, float]) -> dict[str, float]:
    """Keep known labels, replace missing or invalid weights by the mean, scale to sum 1."""
    keys = spec.label_names
    if spec.kind in (TaskKind.SPAN, TaskKind.MULTILABEL) and raw.get(NONE_KEY, 0.0) > 0:
        keys = [*keys, NONE_KEY]
    given = {key: raw[key] for key in keys if raw.get(key, 0.0) > 0}
    if not given:
        return {key: 1.0 / len(keys) for key in keys}
    fill = sum(given.values()) / len(given)
    weights = {key: given.get(key, fill) for key in keys}
    total = sum(weights.values())
    return {key: weight / total for key, weight in weights.items()}


def _clean_axes(axes: list[Axis]) -> list[Axis]:
    cleaned: list[Axis] = []
    for axis in axes:
        values = list(dict.fromkeys(value.strip() for value in axis.values if value.strip()))
        if len(values) >= _MIN_AXIS_VALUES:
            cleaned.append(Axis(name=axis.name, values=values[:_MAX_AXIS_VALUES]))
    return cleaned
