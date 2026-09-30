import random

import pytest
from fakes import FakeResearcher, ScriptedLLM
from pydantic import BaseModel, ValidationError

from bertgen.llm.base import LLMError
from bertgen.stages.generate import (
    ClassBatch,
    ClassItem,
    ClassVerdict,
    ClassVerdictBatch,
    SpanBatch,
    SpanItem,
    SpanText,
    SpanVerdict,
    SpanVerdictBatch,
    class_examples,
    generate,
    locate_spans,
    sample_cell,
    span_examples,
    verify,
)
from bertgen.stages.judge import judge
from bertgen.stages.plan import MAX_SEED_CHARS, PlanDraft, make_plan
from bertgen.stages.spec import define_spec
from bertgen.types import (
    NONE_KEY,
    Axis,
    DataPlan,
    Decision,
    Example,
    HParamPatch,
    LabelDef,
    Metrics,
    Prediction,
    RoundResult,
    Span,
    TaskKind,
    TaskSpec,
    TrainConfig,
    Verdict,
)


def make_spec(kind: TaskKind, *names: str) -> TaskSpec:
    return TaskSpec(
        rule="rule",
        kind=kind,
        language="en",
        labels=[LabelDef(name=n, description=n) for n in names],
        input_description="short text",
        decision_criteria="criteria",
    )


BINARY = make_spec(TaskKind.BINARY, "harm", "none")
MULTILABEL = make_spec(TaskKind.MULTILABEL, "a", "b")
SPAN = make_spec(TaskKind.SPAN, "haiku")


def plan_for(spec: TaskSpec) -> DataPlan:
    weights = {name: 1.0 / len(spec.labels) for name in spec.label_names}
    return DataPlan(
        label_weights=weights,
        axes=[Axis(name="tone", values=["a", "b", "c"])],
        hard_cases=["h1", "h2"],
        style_notes="casual",
    )


def batch(*items: tuple[str, list[str]]) -> ClassBatch:
    return ClassBatch(items=[ClassItem(text=t, labels=ls) for t, ls in items])


def test_class_examples_cardinality_and_unknown_labels() -> None:
    result = class_examples(
        BINARY,
        batch(
            ("ok", ["harm"]),
            ("two", ["harm", "none"]),
            ("zero", []),
            ("unknown", ["other"]),
            ("  ", ["none"]),
        ),
    )
    assert [e.text for e in result] == ["ok"]


def test_multilabel_allows_empty_and_multiple() -> None:
    result = class_examples(MULTILABEL, batch(("x", []), ("y", ["a", "b", "a"]), ("z", ["c"])))
    assert [(e.text, e.labels) for e in result] == [("x", []), ("y", ["a", "b"])]


def test_locate_spans_uses_first_unused_occurrence() -> None:
    spans = locate_spans("ab ab", [SpanText(text="ab", label="haiku")] * 2)
    assert spans == [Span(start=0, end=2, label="haiku"), Span(start=3, end=5, label="haiku")]


def test_locate_spans_missing_returns_none() -> None:
    assert locate_spans("hello", [SpanText(text="xyz", label="haiku")]) is None


def test_span_examples_drop_ambiguous_occurrences() -> None:
    result = span_examples(
        SPAN,
        SpanBatch(
            items=[
                SpanItem(text="ab x ab", spans=[SpanText(text="ab", label="haiku")]),
                SpanItem(text="ab x ab", spans=[SpanText(text="ab", label="haiku")] * 2),
            ]
        ),
    )
    assert len(result) == 1
    assert [(s.start, s.end) for s in result[0].spans] == [(0, 2), (5, 7)]


def test_span_examples_drop_unlocatable_and_unknown_label() -> None:
    result = span_examples(
        SPAN,
        SpanBatch(
            items=[
                SpanItem(text="an old pond", spans=[SpanText(text="old pond", label="haiku")]),
                SpanItem(text="an old pond", spans=[SpanText(text="frog", label="haiku")]),
                SpanItem(text="an old pond", spans=[SpanText(text="old", label="nope")]),
                SpanItem(text="nothing here", spans=[]),
            ]
        ),
    )
    assert [e.text for e in result] == ["an old pond", "nothing here"]
    assert result[0].spans == [Span(start=3, end=11, label="haiku")]


def test_generate_dedups_and_respects_count_and_avoid() -> None:
    llm = ScriptedLLM(
        responder=lambda s, u, schema: batch(
            ("Same", ["harm"]), ("same  ", ["none"]), ("known", ["none"]), ("fresh", ["harm"])
        )
    )
    out = list(
        generate(
            BINARY,
            plan_for(BINARY),
            llm,
            2,
            focus=[],
            seen={"KNOWN"},
            shown=[],
            workers=1,
            seed=1,
        )
    )
    assert [e.text for e in out] == ["Same", "fresh"]
    assert "KNOWN" not in llm.calls[0][1]


def test_generate_quotes_only_shown_texts() -> None:
    llm = ScriptedLLM(responder=lambda s, u, schema: batch(("new", ["harm"])))
    list(
        generate(
            BINARY,
            plan_for(BINARY),
            llm,
            1,
            focus=[],
            seen={"secret test item", "train item"},
            shown=["train item"],
            workers=1,
        )
    )
    user = llm.calls[0][1]
    assert "train item" in user
    assert "secret test item" not in user


def test_generate_prompt_lists_one_cell_per_example() -> None:
    llm = ScriptedLLM(responder=lambda s, u, schema: batch(("t", ["harm"])))
    list(
        generate(
            BINARY,
            plan_for(BINARY),
            llm,
            1,
            focus=[],
            seen=set(),
            shown=[],
            workers=1,
            batch_size=6,
        )
    )
    user = llm.calls[0][1]
    assert user.startswith("Write 5 examples")
    assert [f"{i}. target:" in user for i in range(1, 6)] == [True] * 5
    assert "tone: " in user


def test_generate_stops_when_budget_exhausted(caplog: pytest.LogCaptureFixture) -> None:
    llm = ScriptedLLM(responder=lambda s, u, schema: batch(("only", ["harm"])))
    out = list(
        generate(
            BINARY,
            plan_for(BINARY),
            llm,
            10,
            focus=[],
            seen=set(),
            shown=[],
            workers=2,
            max_calls=4,
        )
    )
    assert len(out) == 1
    assert len(llm.calls) == 4
    assert "budget exhausted" in caplog.text


def test_generate_stops_after_failed_waves(caplog: pytest.LogCaptureFixture) -> None:
    def fail(system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        raise LLMError("boom")

    llm = ScriptedLLM(responder=fail)
    out = list(
        generate(
            BINARY,
            plan_for(BINARY),
            llm,
            3,
            focus=[],
            seen=set(),
            shown=[],
            workers=1,
            max_calls=10,
        )
    )
    assert out == []
    assert len(llm.calls) == 2
    assert "consecutive waves failed" in caplog.text


def test_generate_prompt_carries_focus_and_language() -> None:
    llm = ScriptedLLM(responder=lambda s, u, schema: batch(("t", ["harm"])))
    list(
        generate(
            BINARY, plan_for(BINARY), llm, 1, focus=["sarcasm"], seen=set(), shown=[], workers=1
        )
    )
    system, user, _ = llm.calls[0]
    assert "sarcasm" in user
    assert "'en'" in system


def test_generate_is_deterministic_for_seed() -> None:
    def run() -> list[str]:
        llm = ScriptedLLM(responder=lambda s, u, schema: batch(("t", ["harm"])))
        list(
            generate(
                BINARY,
                plan_for(BINARY),
                llm,
                3,
                focus=[],
                seen=set(),
                shown=[],
                workers=1,
                seed=7,
            )
        )
        return [user for _, user, _ in llm.calls]

    assert run() == run()


def test_sample_cell_follows_weights() -> None:
    plan = plan_for(BINARY).model_copy(update={"label_weights": {"harm": 1.0, "none": 0.0}})
    rng = random.Random(0)
    cells = [sample_cell(plan, rng) for _ in range(50)]
    assert {cell.label for cell in cells} == {"harm"}
    assert 0 < sum(cell.lookalike for cell in cells) < 50


def test_sample_cell_never_lookalike_for_none() -> None:
    plan = plan_for(SPAN).model_copy(update={"label_weights": {NONE_KEY: 1.0}})
    rng = random.Random(0)
    assert not any(sample_cell(plan, rng).lookalike for _ in range(20))


def test_verify_keeps_matching_classification() -> None:
    examples = [
        Example(text="a", labels=["harm"]),
        Example(text="b", labels=["none"]),
        Example(text="c", labels=["harm"]),
    ]
    verdict = ClassVerdictBatch(
        items=[
            ClassVerdict(index=0, labels=["harm"]),
            ClassVerdict(index=1, labels=["harm"]),
        ]
    )
    llm = ScriptedLLM([verdict])
    kept = verify(BINARY, examples, llm, workers=1)
    assert [e.text for e in kept] == ["a"]
    assert "harm" not in llm.calls[0][1].replace('"a"', "")


def test_verify_span_compares_substring_and_label() -> None:
    examples = [
        Example(text="an old pond", spans=[Span(start=3, end=11, label="haiku")]),
        Example(text="another one", spans=[Span(start=0, end=7, label="haiku")]),
    ]
    verdict = SpanVerdictBatch(
        items=[
            SpanVerdict(index=0, spans=[SpanText(text="old pond", label="haiku")]),
            SpanVerdict(index=1, spans=[]),
        ]
    )
    kept = verify(SPAN, examples, ScriptedLLM([verdict]), workers=1)
    assert [e.text for e in kept] == ["an old pond"]


def test_verify_span_counts_repeated_occurrences() -> None:
    twice = [Span(start=0, end=2, label="haiku"), Span(start=3, end=5, label="haiku")]
    examples = [Example(text="ab ab", spans=twice)]
    verdict = SpanVerdictBatch(
        items=[SpanVerdict(index=0, spans=[SpanText(text="ab", label="haiku")])]
    )
    assert verify(SPAN, examples, ScriptedLLM([verdict]), workers=1) == []


def round_result(n: int, primary: float, verdict: Verdict | None = None) -> RoundResult:
    metrics = Metrics(primary=primary, accuracy=primary, per_label=[], errors=[], n_test=10)
    return RoundResult(
        round=n, config=TrainConfig(base_model="m"), metrics=metrics, verdict=verdict
    )


def test_judge_accepts_at_target_without_calling_llm() -> None:
    llm = ScriptedLLM()
    history = [round_result(0, 0.95)]
    verdict = judge(BINARY, plan_for(BINARY), history, {}, [], llm, target=0.9, warm_start=True)
    assert verdict.decision is Decision.ACCEPT
    assert llm.calls == []


@pytest.mark.parametrize(
    ("warm_start", "phrase"),
    [(True, "extra passes on top of that model"), (False, "starts from the base model")],
)
def test_judge_delegates_below_target_and_shows_history(warm_start: bool, phrase: str) -> None:
    error = Prediction(
        example=Example(text="bad text", labels=["harm"]),
        predicted=Example(text="bad text", labels=["none"]),
    )
    earlier = Verdict(decision=Decision.TUNE, reason="underfit", hparams=HParamPatch(epochs=5))
    history = [round_result(0, 0.5, earlier), round_result(1, 0.6)]
    llm = ScriptedLLM([Verdict(decision=Decision.MORE_DATA, reason="r", focus=["x"])])
    verdict = judge(
        BINARY, plan_for(BINARY), history, {}, [error], llm, target=0.9, warm_start=warm_start
    )
    assert verdict.decision is Decision.MORE_DATA
    user = llm.calls[0][1]
    assert phrase in user
    assert "bad text" in user
    assert "verdict: tune (underfit)" in user
    assert '"epochs": 5.0' in user


@pytest.mark.parametrize("patch", [{"batch_size": 0}, {"epochs": 0}, {"learning_rate": -1e-5}])
def test_hparam_patch_rejects_invalid_values(patch: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        Verdict.model_validate({"decision": "tune", "reason": "r", "hparams": patch})


def test_hparam_patch_applies_over_config() -> None:
    config = HParamPatch(epochs=4).apply(TrainConfig(base_model="m", epochs=1))
    assert config == TrainConfig(base_model="m", epochs=4)


def test_define_spec_overrides_rule_and_keeps_kind() -> None:
    llm = ScriptedLLM([SPAN.model_copy(update={"rule": "echoed wrongly"})])
    spec = define_spec("find haiku", llm)
    assert spec.rule == "find haiku"
    assert spec.kind is TaskKind.SPAN


def test_define_spec_retries_invalid_binary() -> None:
    bad = BINARY.model_copy(update={"labels": [LabelDef(name="x", description="x")]})
    llm = ScriptedLLM([bad, BINARY])
    assert define_spec("r", llm).kind is TaskKind.BINARY
    assert len(llm.calls) == 2


def test_make_plan_normalizes_and_uses_research() -> None:
    draft = PlanDraft(
        label_weights={"harm": 3.0, "bogus": 5.0},
        axes=[
            Axis(name="ok", values=[str(i) for i in range(12)]),
            Axis(name="thin", values=["x", "y"]),
        ],
        hard_cases=["case", " "],
        style_notes="s",
    )
    llm = ScriptedLLM([draft])
    researcher = FakeResearcher("slang notes")
    plan = make_plan(BINARY, llm, researcher)
    assert plan.label_weights == pytest.approx({"harm": 0.5, "none": 0.5})
    assert [a.name for a in plan.axes] == ["ok"]
    assert len(plan.axes[0].values) == 8
    assert plan.hard_cases == ["case"]
    assert plan.research_notes == "slang notes"
    assert "slang notes" in llm.calls[0][1]
    assert len(researcher.questions) == 1


def test_make_plan_span_keeps_none_key() -> None:
    draft = PlanDraft(
        label_weights={"haiku": 0.8, "__none__": 0.2}, axes=[], hard_cases=[], style_notes=""
    )
    plan = make_plan(SPAN, ScriptedLLM([draft]), None)
    assert plan.label_weights == pytest.approx({"haiku": 0.8, "__none__": 0.2})


def _seed_draft(seeds: list[str]) -> PlanDraft:
    return PlanDraft(
        label_weights={"harm": 1.0, "none": 1.0},
        axes=[],
        hard_cases=[],
        style_notes="",
        seeds=seeds,
    )


def test_make_plan_splits_notes_from_examples_and_cleans_seeds() -> None:
    research = "slang notes\nEXAMPLES\nfoo | http://a\nbar | http://b"
    seeds = ["  foo ", "FOO", "", "x" * (MAX_SEED_CHARS + 1), "bar"] + [f"s{i}" for i in range(60)]
    llm = ScriptedLLM([_seed_draft(seeds)])
    plan = make_plan(BINARY, llm, FakeResearcher(research))
    assert plan.research_notes == "slang notes"
    assert "foo | http://a" in llm.calls[0][1]
    assert plan.seeds[:3] == ["foo", "bar", "s0"]
    assert len(plan.seeds) == 40


def test_make_plan_ignores_seeds_without_research() -> None:
    plan = make_plan(BINARY, ScriptedLLM([_seed_draft(["invented"])]), None)
    assert plan.seeds == []


def test_research_question_asks_for_examples() -> None:
    researcher = FakeResearcher("notes")
    make_plan(BINARY, ScriptedLLM([_seed_draft([])]), researcher)
    assert "EXAMPLES" in researcher.questions[0]
    assert "verbatim" in researcher.questions[0]


def _run_generate(plan: DataPlan, llm: ScriptedLLM, seed: int) -> None:
    list(
        generate(
            BINARY, plan, llm, 3, focus=[], seen=set(), shown=[], workers=1, seed=seed, max_calls=6
        )
    )


def test_generate_prompts_carry_at_most_three_seeds() -> None:
    llm = ScriptedLLM(responder=lambda s, u, schema: batch(("t", ["harm"])))
    plan = plan_for(BINARY).model_copy(update={"seeds": [f"SEEDTEXT{i}" for i in range(10)]})
    _run_generate(plan, llm, 1)
    counts = [user.count("SEEDTEXT") for _, user, _ in llm.calls]
    assert max(counts) <= 3
    assert any(counts)
    assert all("Real-world material" in user for _, user, _ in llm.calls if "SEEDTEXT" in user)


def test_generate_seed_sampling_is_deterministic() -> None:
    plan = plan_for(BINARY).model_copy(update={"seeds": [f"SEEDTEXT{i}" for i in range(10)]})
    users: list[list[str]] = []
    for _ in range(2):
        llm = ScriptedLLM(responder=lambda s, u, schema: batch(("t", ["harm"])))
        _run_generate(plan, llm, 5)
        users.append(sorted(user for _, user, _ in llm.calls))
    assert users[0] == users[1]


def test_generate_without_seeds_has_no_seed_section() -> None:
    llm = ScriptedLLM(responder=lambda s, u, schema: batch(("t", ["harm"])))
    _run_generate(plan_for(BINARY), llm, 1)
    assert all("Real-world material" not in user for _, user, _ in llm.calls)


def test_round_rank_breaks_test_ties_on_validation() -> None:
    def scored(n: int, test: float, valid: float) -> RoundResult:
        result = round_result(n, test)
        return result.model_copy(update={"valid_metrics": round_result(n, valid).metrics})

    rounds = [scored(0, 0.0, 0.1), scored(1, 0.0, 0.3), scored(2, 0.0, 0.2)]
    assert max(rounds, key=lambda r: r.rank).round == 1
    assert max([*rounds, scored(3, 0.1, 0.0)], key=lambda r: r.rank).round == 3
