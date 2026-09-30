"""Stage 6: decide whether to continue after an evaluation round."""

from collections.abc import Sequence

from bertgen.llm.base import LLM
from bertgen.stages.prompts import JUDGE_SYSTEM, describe_example, judge_user
from bertgen.types import (
    DataPlan,
    Decision,
    Prediction,
    RoundResult,
    Split,
    TaskSpec,
    Verdict,
)

_MAX_ERRORS = 20


def judge(
    spec: TaskSpec,
    plan: DataPlan,
    history: Sequence[RoundResult],
    counts: dict[Split, dict[str, int]],
    errors: Sequence[Prediction],
    llm: LLM,
    target: float,
    *,
    warm_start: bool,
) -> Verdict:
    """Accept when any round reached `target`; otherwise let the LLM choose the next step.

    `history` holds every evaluated round, oldest first, with test metrics. `errors` should come
    from a split other than test, since the LLM's focus items feed back into data generation.
    `warm_start` tells the LLM whether the next round continues from the latest model.
    """
    assert history, "judge needs at least one evaluated round"
    best = max(result.metrics.primary for result in history)
    if best >= target:
        return Verdict(
            decision=Decision.ACCEPT, reason=f"best primary {best:.4f} reached target {target}"
        )
    user = judge_user(
        spec,
        plan,
        history,
        {split.value: per_label for split, per_label in counts.items()},
        _error_sample(errors),
        target,
        warm_start,
    )
    return llm.ask(JUDGE_SYSTEM, user, Verdict)


def _error_sample(errors: Sequence[Prediction]) -> list[tuple[str, str, str]]:
    stride = max(1, len(errors) // _MAX_ERRORS)
    return [
        (p.example.text, describe_example(p.example), describe_example(p.predicted))
        for p in errors[::stride][:_MAX_ERRORS]
    ]
