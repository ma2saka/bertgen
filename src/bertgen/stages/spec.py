"""Stage 1: turn a natural-language rule into a TaskSpec."""

import logging

from bertgen.llm.base import LLM, LLMError
from bertgen.stages.prompts import SPEC_SYSTEM, spec_user
from bertgen.types import TaskKind, TaskSpec

log = logging.getLogger(__name__)

_MAX_ATTEMPTS = 2


def define_spec(rule: str, llm: LLM) -> TaskSpec:
    """Ask the LLM for a TaskSpec; the returned spec always carries the given rule."""
    user = spec_user(rule)
    problem: str | None = None
    for _ in range(_MAX_ATTEMPTS):
        prompt = user if problem is None else f"{user}\n\nFix this problem: {problem}"
        spec = llm.ask(SPEC_SYSTEM, prompt, TaskSpec).model_copy(update={"rule": rule})
        problem = _problem(spec)
        if problem is None:
            return spec
        log.warning("spec rejected: %s", problem)
    raise LLMError(f"could not obtain a valid spec: {problem}")


def _problem(spec: TaskSpec) -> str | None:
    names = spec.label_names
    if len(set(names)) != len(names):
        return "label names must be unique"
    if spec.kind is TaskKind.BINARY and len(names) != 2:
        return "binary tasks need exactly two labels, including an explicit negative class"
    if spec.kind is TaskKind.MULTICLASS and len(names) < 2:
        return "multiclass tasks need at least two labels"
    return None
