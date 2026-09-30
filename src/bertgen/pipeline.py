"""Resumable orchestration of the six stages: spec, plan, generate, train, evaluate, judge."""

import logging
import math
import random
import shutil
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from bertgen.corpus import (
    STALL_LIMIT,
    CorpusError,
    Pool,
    bucket_of,
    corpus_format,
    corpus_weights,
    label_quotas,
    read_passages,
    sample,
)
from bertgen.llm import LLM, LLMError, WebResearcher, parse_llm, web_researcher
from bertgen.llm.registry import parse_llm_spec
from bertgen.oracles import Oracle, load_oracle, parse_oracle_ref
from bertgen.presets import parse_model_choice, resolve_base_model
from bertgen.stages.evaluate import evaluate
from bertgen.stages.generate import consult_oracle, generate, verify
from bertgen.stages.judge import judge
from bertgen.stages.plan import make_plan
from bertgen.stages.prompts import describe_example
from bertgen.stages.spec import define_spec
from bertgen.stages.train import train
from bertgen.store import ExampleStore
from bertgen.text import normalize
from bertgen.types import (
    DataPlan,
    Decision,
    Draft,
    HParamPatch,
    Metrics,
    OracleAgreement,
    OracleOutcome,
    RoundInit,
    RoundResult,
    Split,
    Stage,
    TaskKind,
    TaskSpec,
    TrainConfig,
    Verdict,
)

log = logging.getLogger(__name__)

DEFAULT_LLM = "anthropic:claude-sonnet-5-5"
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"

_TOP_UP_FACTOR = 1.5
_TOP_UP_THRESHOLD = 0.05
_REPORT_ERRORS = 5
_UNLABELED = (None, OracleOutcome.ABSTAINED)

LLMFactory = Callable[[str], LLM]
type Agreement = dict[Split, OracleAgreement]


class RunConfig(BaseModel):
    """Everything needed to (re)start a run; stored in `run.json`."""

    name: Annotated[str, Field(pattern=NAME_PATTERN)]
    rule: Annotated[str, Field(min_length=1)]
    runs_dir: Path = Path("runs")
    model: str = "base"
    llm: str = DEFAULT_LLM
    eval_llm: str | None = None
    planner_llm: str | None = None
    web: bool = False
    verify: bool = True
    warm_start: Annotated[
        bool, Field(description="initialize each round after the first from the previous model")
    ] = True
    oracle: Annotated[
        str | None, Field(description="'package.module:attr' of an oracle that relabels data")
    ] = None
    corpus: Annotated[
        Path | None,
        Field(description="shuffled .txt/.jsonl(.gz) file labeled by the oracle instead of an LLM"),
    ] = None
    corpus_max_chars: Annotated[int, Field(ge=20)] = 300
    examples: Annotated[int, Field(ge=2)] = 400
    test_examples: Annotated[int, Field(ge=1)] = 200
    examples_per_round: Annotated[int, Field(ge=1)] = 200
    max_rounds: Annotated[int, Field(ge=1)] = 3
    target: Annotated[float, Field(gt=0)] = 0.9
    valid_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.15
    workers: Annotated[int, Field(ge=1)] = 4
    seed: int = 42
    hparams: HParamPatch = HParamPatch()

    @field_validator("model")
    @classmethod
    def _model_choice(cls, value: str) -> str:
        parse_model_choice(value)
        return value

    @field_validator("oracle")
    @classmethod
    def _oracle_ref(cls, value: str | None) -> str | None:
        if value is not None:
            parse_oracle_ref(value)
        return value

    @field_validator("llm", "eval_llm", "planner_llm")
    @classmethod
    def _llm_spec(cls, value: str | None) -> str | None:
        if value is not None:
            parse_llm_spec(value)
        return value

    @field_validator("corpus")
    @classmethod
    def _corpus_path(cls, value: Path | None) -> Path | None:
        if value is not None:
            corpus_format(value)
        return value

    @model_validator(mode="after")
    def _corpus_needs_oracle(self) -> Self:
        if self.corpus is not None and self.oracle is None:
            raise ValueError("corpus requires an oracle to label its passages")
        return self

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.name


class RunState(BaseModel):
    """`oracle_agreement` mirrors, per split, how the oracle judged the stored examples' LLM
    labels in `data.db`; it is empty for runs without an oracle."""

    config: RunConfig
    round: int = 0
    oracle_agreement: Agreement = {}


class Providers:
    """LLM roles, each built on first use: `generator` writes train data, `evaluator` writes test
    data and each verifies the other's output, `planner` defines the spec and plan and judges."""

    def __init__(self, config: RunConfig, factory: LLMFactory = parse_llm) -> None:
        self._config = config
        self._factory = factory

    @cached_property
    def generator(self) -> LLM:
        return self._factory(self._config.llm)

    @cached_property
    def evaluator(self) -> LLM:
        spec = self._config.eval_llm
        return self._factory(spec) if spec else self.generator

    @cached_property
    def planner(self) -> LLM:
        spec = self._config.planner_llm
        return self._factory(spec) if spec else self.generator

    @cached_property
    def researcher(self) -> WebResearcher | None:
        if not self._config.web:
            return None
        researcher = web_researcher(self.planner)
        if researcher is None:
            log.warning(
                "planner %s cannot search the web; continuing without research",
                self.planner.name,
            )
        return researcher


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def state(self) -> Path:
        return self.root / "run.json"

    @property
    def spec(self) -> Path:
        return self.root / "spec.json"

    @property
    def plan(self) -> Path:
        return self.root / "plan.json"

    @property
    def db(self) -> Path:
        return self.root / "data.db"

    @property
    def model(self) -> Path:
        return self.root / "model"

    def round(self, n: int) -> Path:
        return self.root / "rounds" / str(n)


def load_state(run_dir: Path) -> RunState:
    """Read `run.json` of an existing run."""
    return RunState.model_validate_json(RunPaths(run_dir).state.read_text())


def load_spec(run_dir: Path) -> TaskSpec:
    return TaskSpec.model_validate_json(RunPaths(run_dir).spec.read_text())


def load_rounds(run_dir: Path) -> list[RoundResult]:
    """Evaluated rounds in order, with their verdict when one was recorded."""
    rounds: list[RoundResult] = []
    paths = RunPaths(run_dir)
    n = 0
    while (metrics_path := paths.round(n) / "metrics.json").exists():
        round_dir = paths.round(n)
        verdict_path = round_dir / "verdict.json"
        valid_path = round_dir / "valid_metrics.json"
        init_path = round_dir / "init.json"
        rounds.append(
            RoundResult(
                round=n,
                config=TrainConfig.model_validate_json(
                    (round_dir / "train_config.json").read_text()
                ),
                metrics=Metrics.model_validate_json(metrics_path.read_text()),
                valid_metrics=Metrics.model_validate_json(valid_path.read_text())
                if valid_path.exists()
                else None,
                verdict=Verdict.model_validate_json(verdict_path.read_text())
                if verdict_path.exists()
                else None,
                warm_from=RoundInit.model_validate_json(init_path.read_text()).warm_from
                if init_path.exists()
                else None,
            )
        )
        n += 1
    return rounds


def run(
    config: RunConfig, factory: LLMFactory = parse_llm, *, stop_after: Stage | None = None
) -> Path:
    """Run or resume `config`; returns the directory of the best model.

    Every stage whose artifact already exists on disk is skipped. With `stop_after` set to SPEC,
    PLAN or GENERATE, returns that stage's artifact instead, so it can be reviewed or edited
    before the run is resumed.
    """
    paths = RunPaths(config.run_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    llms = Providers(config, factory)
    stored = load_state(paths.root).oracle_agreement if paths.state.exists() else {}
    state = RunState(config=config, oracle_agreement=stored)
    _write(paths.state, state)
    if config.corpus is None and config.verify and config.eval_llm in (None, config.llm):
        log.warning(
            "generator and evaluator are the same LLM (%s): verification only checks "
            "self-consistency; set --eval-llm for an independent cross-check",
            config.llm,
        )

    if config.corpus is not None and not config.corpus.is_file():
        raise CorpusError(f"corpus file not found: {config.corpus}")
    spec = _cached(paths.spec, TaskSpec, lambda: define_spec(config.rule, llms.planner))
    log.info("spec: %s, labels %s, language %s", spec.kind, spec.label_names, spec.language)
    base_model = resolve_base_model(config.model, spec.language)
    if stop_after is Stage.SPEC:
        return paths.spec
    oracle = load_oracle(config.oracle, spec) if config.oracle else None
    plan = _cached(paths.plan, DataPlan, lambda: make_plan(spec, llms.planner, llms.researcher))
    log.info(
        "plan: %d axes, %d hard cases, %d seeds",
        len(plan.axes),
        len(plan.hard_cases),
        len(plan.seeds),
    )
    if stop_after is Stage.PLAN:
        return paths.plan

    with ExampleStore(paths.db) as store:
        _initial_data(config, spec, plan, llms, store, oracle)
        state = _sync_agreement(paths, state, store)
        if stop_after is Stage.GENERATE:
            return paths.db
        config_next = config.hparams.apply(TrainConfig(base_model=base_model, seed=config.seed))

        for n in range(config.max_rounds):
            state = state.model_copy(update={"round": n})
            _write(paths.state, state)
            round_dir = paths.round(n)
            train_config = _cached(
                round_dir / "train_config.json", TrainConfig, config_next.model_copy
            )
            warm_from = _cached(
                round_dir / "init.json",
                RoundInit,
                partial(RoundInit, warm_from=n - 1 if config.warm_start and n > 0 else None),
            ).warm_from
            init = None if warm_from is None else paths.round(warm_from) / "model"
            model_dir = _train_round(spec, store, train_config, round_dir, init)
            metrics = _cached(
                round_dir / "metrics.json",
                Metrics,
                partial(evaluate, spec, model_dir, store.load(Split.TEST)),
            )
            valid_metrics = _cached(
                round_dir / "valid_metrics.json",
                Metrics,
                partial(evaluate, spec, model_dir, store.load(Split.VALID)),
            )
            log.info(
                "round %d: primary %.4f, accuracy %.4f (valid primary %.4f)",
                n,
                metrics.primary,
                metrics.accuracy,
                valid_metrics.primary,
            )
            history = load_rounds(paths.root)
            _link_best(paths, history)

            verdict_path = round_dir / "verdict.json"
            if verdict_path.exists():
                verdict = Verdict.model_validate_json(verdict_path.read_text())
            else:
                verdict = judge(
                    spec,
                    plan,
                    history[: n + 1],
                    store.counts(),
                    valid_metrics.errors,
                    llms.planner,
                    config.target,
                    warm_start=config.warm_start,
                )
                _write(verdict_path, verdict)
            log.info("round %d verdict: %s (%s)", n, verdict.decision, verdict.reason)
            (round_dir / "report.md").write_text(
                render_report(
                    n,
                    spec,
                    train_config,
                    metrics,
                    valid_metrics,
                    verdict,
                    state.oracle_agreement,
                    warm_from=warm_from,
                )
            )

            if verdict.decision in (Decision.ACCEPT, Decision.STOP):
                break
            if n + 1 >= config.max_rounds:
                log.info("reached max_rounds=%d", config.max_rounds)
                break
            if verdict.decision is Decision.MORE_DATA:
                _more_data(config, spec, plan, llms, store, oracle, n + 1, verdict.focus)
                state = _sync_agreement(paths, state, store)
            config_next = verdict.hparams.apply(train_config)
            if config_next == train_config and not store.load(Split.TRAIN, round_=n + 1):
                log.warning("round %d would repeat round %d unchanged; stopping", n + 1, n)
                break

    log.info("best model: %s", paths.model.resolve())
    return paths.model


def _initial_data(
    config: RunConfig,
    spec: TaskSpec,
    plan: DataPlan,
    llms: Providers,
    store: ExampleStore,
    oracle: Oracle | None,
) -> None:
    """Generate the missing test and round-0 train/valid data, or sample it from the corpus.

    The test writer sees no plan seeds, so test items share no raw material with training data.
    """
    if config.corpus is not None:
        _initial_corpus_data(config, spec, plan, store, oracle)
        return
    if not store.load(Split.TEST):
        log.info("generating %d test examples", config.test_examples)
        test = _synthesize(
            config,
            spec,
            plan.model_copy(update={"seeds": []}),
            llms.evaluator,
            llms.generator,
            config.test_examples,
            focus=[],
            store=store,
            oracle=oracle,
            seed=config.seed + 1,
        )
        rows = [(draft, Split.TEST) for draft in test]
        store.add(rows, 0, llms.evaluator.name)
        _log_agreement(rows)
    if not store.load(Split.TRAIN, round_=0):
        log.info("generating %d train/valid examples", config.examples)
        examples = _synthesize(
            config,
            spec,
            plan,
            llms.generator,
            llms.evaluator,
            config.examples,
            focus=[],
            store=store,
            oracle=oracle,
            seed=config.seed + 2,
        )
        _add_train_valid(config, store, examples, 0, llms.generator.name)
    _log_counts(store)


def _initial_corpus_data(
    config: RunConfig, spec: TaskSpec, plan: DataPlan, store: ExampleStore, oracle: Oracle | None
) -> None:
    if not store.load(Split.TEST):
        test = _from_corpus(config, spec, plan, store, oracle, Pool.TEST, config.test_examples)
        store.add([(draft, Split.TEST) for draft in test], 0, _corpus_source(config))
    if not store.load(Split.TRAIN, round_=0):
        examples = _from_corpus(config, spec, plan, store, oracle, Pool.TRAIN, config.examples)
        _add_train_valid(config, store, examples, 0, _corpus_source(config))
    _log_counts(store)


def _more_data(
    config: RunConfig,
    spec: TaskSpec,
    plan: DataPlan,
    llms: Providers,
    store: ExampleStore,
    oracle: Oracle | None,
    round_: int,
    focus: list[str],
) -> None:
    if store.load(Split.TRAIN, round_=round_):
        return
    if config.corpus is not None:
        if focus:
            log.info("judge focus is ignored for corpus data: %s", "; ".join(focus))
        examples = _from_corpus(
            config,
            spec,
            plan,
            store,
            oracle,
            Pool.TRAIN,
            config.examples_per_round,
            required=False,
        )
        _add_train_valid(config, store, examples, round_, _corpus_source(config))
        _log_counts(store)
        return
    log.info("generating %d more examples for round %d", config.examples_per_round, round_)
    examples = _synthesize(
        config,
        spec,
        plan,
        llms.generator,
        llms.evaluator,
        config.examples_per_round,
        focus=focus,
        store=store,
        oracle=oracle,
        seed=config.seed + 1000 * round_,
    )
    _add_train_valid(config, store, examples, round_, llms.generator.name)
    _log_counts(store)


def _synthesize(
    config: RunConfig,
    spec: TaskSpec,
    plan: DataPlan,
    writer: LLM,
    checker: LLM,
    count: int,
    *,
    focus: list[str],
    store: ExampleStore,
    oracle: Oracle | None,
    seed: int,
) -> list[Draft]:
    """Generate `count` examples with `writer` and keep those `checker` agrees with.

    With an `oracle`, examples it labels take its labels and skip verification; each draft
    records the oracle's outcome (None without an oracle). New texts differ from every stored one;
    prompts quote train/valid texts only, so test items never reach a generation prompt. One
    top-up batch is generated when verification drops more than 5%. Raises LLMError when nothing
    usable comes back.
    """
    seen = set(store.texts())
    shown = store.texts(Split.TRAIN, Split.VALID)

    def batch(n: int, batch_seed: int) -> list[Draft]:
        drafted = list(
            generate(
                spec,
                plan,
                writer,
                n,
                focus=focus,
                seen=seen,
                shown=shown,
                workers=config.workers,
                seed=batch_seed,
            )
        )
        seen.update(example.text for example in drafted)
        consulted = (
            list(consult_oracle(oracle, drafted)) if oracle else list(map(Draft.plain, drafted))
        )
        pending = [d.example for d in consulted if d.outcome in _UNLABELED]
        if not config.verify or not pending:
            return consulted
        verified = {ex.text for ex in verify(spec, pending, checker, workers=config.workers)}
        return [d for d in consulted if d.outcome not in _UNLABELED or d.example.text in verified]

    kept = batch(count, seed)
    missing = count - len(kept)
    if config.verify and missing > count * _TOP_UP_THRESHOLD:
        kept += batch(math.ceil(missing * _TOP_UP_FACTOR), seed + 500)
    if not kept:
        raise LLMError(f"no usable examples from {writer.name}; see the warnings above")
    if len(kept) < count:
        log.warning("synthesized %d of %d requested examples", len(kept), count)
    return kept[:count]


def _from_corpus(
    config: RunConfig,
    spec: TaskSpec,
    plan: DataPlan,
    store: ExampleStore,
    oracle: Oracle | None,
    pool: Pool,
    count: int,
    *,
    required: bool = True,
) -> list[Draft]:
    """Sample `count` oracle-labeled passages of `pool` not yet stored, split by the plan's label
    weights plus a negative share (see `corpus_weights`).

    Raises CorpusError when `required` and nothing was found, and for the test pool when any
    bucket stays empty.
    """
    assert config.corpus is not None and oracle is not None
    log.info("sampling %d %s examples from %s", count, pool, config.corpus)
    wanted = label_quotas(corpus_weights(spec, plan.label_weights), count)
    seen = {normalize(text) for text in store.texts()}
    with closing(read_passages(config.corpus, config.corpus_max_chars)) as passages:
        sampled = sample(passages, oracle, pool, wanted, seen)
    examples = sampled.examples
    filled = Counter(map(bucket_of, examples))
    breakdown = ", ".join(f"{key} {filled[key]}/{n}" for key, n in wanted.items())
    if len(examples) < count:
        reason = (
            f"no progress in the last {STALL_LIMIT} passages" if sampled.stalled else "end of file"
        )
        log.warning(
            "corpus ran out: %d of %d %s examples after %d passages, %s (%s)",
            len(examples),
            count,
            pool,
            sampled.scanned,
            reason,
            breakdown,
        )
    if required and not examples:
        raise CorpusError(f"no {pool} passage of {config.corpus} was labeled by the oracle")
    if pool is Pool.TEST and any(n and not filled[key] for key, n in wanted.items()):
        raise CorpusError(f"{config.corpus} yields a test set with an empty bucket: {breakdown}")
    return [Draft.plain(example) for example in examples]


def _corpus_source(config: RunConfig) -> str:
    assert config.corpus is not None
    return f"corpus:{config.corpus.name}"


def _add_train_valid(
    config: RunConfig, store: ExampleStore, drafts: list[Draft], round_: int, source: str
) -> None:
    shuffled = drafts.copy()
    random.Random(config.seed + round_).shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * config.valid_ratio))
    rows = [(d, Split.VALID if i < n_valid else Split.TRAIN) for i, d in enumerate(shuffled)]
    store.add(rows, round_, source)
    _log_agreement(rows)


def _log_agreement(rows: list[tuple[Draft, Split]]) -> None:
    for split in Split:
        outcomes = [d.outcome for d, s in rows if s is split and d.outcome is not None]
        if outcomes:
            stats = OracleAgreement.tally(outcomes)
            log.info("  oracle on new %s: %s", split, describe_agreement(stats))


def _sync_agreement(paths: RunPaths, state: RunState, store: ExampleStore) -> RunState:
    state = state.model_copy(update={"oracle_agreement": store.agreement()})
    _write(paths.state, state)
    return state


def describe_agreement(stats: OracleAgreement) -> str:
    """One-line summary such as 'LLM label agreed with oracle 92.0% (46/50), 3 abstained'."""
    rate = "n/a" if stats.rate is None else f"{stats.rate:.1%}"
    return (
        f"LLM label agreed with oracle {rate} ({stats.agreed}/{stats.labeled}), "
        f"{stats.abstained} abstained"
    )


def _train_round(
    spec: TaskSpec, store: ExampleStore, config: TrainConfig, round_dir: Path, init: Path | None
) -> Path:
    model_dir = round_dir / "model"
    if model_dir.exists():
        return model_dir
    staging = round_dir / "model.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    train_set, valid_set = store.load(Split.TRAIN), store.load(Split.VALID)
    log.info(
        "training %s on %d examples (%d valid)",
        init or config.base_model,
        len(train_set),
        len(valid_set),
    )
    train(spec, train_set, valid_set, config, staging, init)
    staging.rename(model_dir)
    log.info("saved model to %s", model_dir)
    return model_dir


def _link_best(paths: RunPaths, history: list[RoundResult]) -> None:
    best = max(history, key=lambda result: result.rank)
    if paths.model.is_symlink():
        paths.model.unlink()
    paths.model.symlink_to(Path("rounds") / str(best.round) / "model", target_is_directory=True)


def _log_counts(store: ExampleStore) -> None:
    for split, per_label in store.counts().items():
        log.info("  %s: %s", split, dict(sorted(per_label.items())) or "{}")


def _cached[T: BaseModel](path: Path, schema: type[T], compute: Callable[[], T]) -> T:
    if path.exists():
        return schema.model_validate_json(path.read_text())
    value = compute()
    _write(path, value)
    return value


def _write(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value.model_dump_json(indent=2) + "\n")
    tmp.replace(path)


def render_report(
    n: int,
    spec: TaskSpec,
    config: TrainConfig,
    metrics: Metrics,
    valid_metrics: Metrics,
    verdict: Verdict,
    agreement: Agreement | None = None,
    *,
    warm_from: int | None = None,
) -> str:
    """Markdown summary of one round; scores and errors are from the test split.

    `agreement` adds a table of oracle agreement over all data stored so far; `warm_from` names
    the round whose model this round started from.
    """
    lines = [
        f"# Round {n}",
        "",
        f"Rule: {spec.rule}",
        f"Task: {spec.kind}, base model `{config.base_model}`"
        + ("" if warm_from is None else f", initialized from round {warm_from}"),
        "",
        "| metric | value |",
        "|---|---|",
        f"| primary | {metrics.primary:.4f} |",
        f"| accuracy | {metrics.accuracy:.4f} |",
        f"| test examples | {metrics.n_test} |",
        *(
            [f"| unreachable gold spans | {metrics.unreachable_spans} |"]
            if spec.kind is TaskKind.SPAN
            else []
        ),
        f"| validation primary | {valid_metrics.primary:.4f} |",
        "",
        "| label | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
        *(
            f"| {s.label} | {s.precision:.3f} | {s.recall:.3f} | {s.f1:.3f} | {s.support} |"
            for s in metrics.per_label
        ),
        "",
        f"Training: epochs {config.epochs}, lr {config.learning_rate}, "
        f"batch {config.batch_size}, max_length {config.max_length}",
        "",
        f"## Verdict: {verdict.decision}",
        "",
        verdict.reason,
    ]
    if verdict.focus:
        lines += ["", "Focus:", *(f"- {item}" for item in verdict.focus)]
    if verdict.hparams.model_dump(exclude_none=True):
        lines += ["", f"Hyperparameters: `{verdict.hparams.model_dump_json(exclude_none=True)}`"]
    if metrics.errors:
        lines += ["", f"## Errors ({len(metrics.errors)} total)", ""]
        for p in metrics.errors[:_REPORT_ERRORS]:
            lines += [
                f"- {p.example.text[:200]!r}",
                f"  - gold: {describe_example(p.example)}",
                f"  - predicted: {describe_example(p.predicted)}",
            ]
    if agreement:
        lines += [
            "",
            "## Oracle agreement with LLM labels",
            "",
            "| split | agreed | disagreed | abstained | rate |",
            "|---|---|---|---|---|",
            *(
                f"| {split} | {a.agreed} | {a.disagreed} | {a.abstained} | "
                f"{'-' if a.rate is None else f'{a.rate:.3f}'} |"
                for split, a in agreement.items()
            ),
        ]
    return "\n".join(lines) + "\n"
