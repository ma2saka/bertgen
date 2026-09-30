"""The `bgen` command: run or resume a run, `predict` with its model, show its `status`."""

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from itertools import batched
from pathlib import Path
from typing import TextIO

from transformers.utils import logging as transformers_logging

from bertgen.corpus import CorpusError
from bertgen.llm import LLMError
from bertgen.oracles import OracleError
from bertgen.pipeline import (
    DEFAULT_LLM,
    NAME_PATTERN,
    RunConfig,
    RunPaths,
    describe_agreement,
    load_rounds,
    load_spec,
    load_state,
    run,
)
from bertgen.stages.evaluate import BATCH_SIZE, Predictor
from bertgen.types import Scored, Stage

_RUN_FIELDS = (
    "rule",
    "model",
    "llm",
    "eval_llm",
    "planner_llm",
    "web",
    "verify",
    "warm_start",
    "oracle",
    "corpus",
    "corpus_max_chars",
    "examples",
    "test_examples",
    "examples_per_round",
    "max_rounds",
    "target",
    "valid_ratio",
    "workers",
    "seed",
)
_STOP_STAGES = (Stage.SPEC, Stage.PLAN, Stage.GENERATE)
_HPARAM_OPTIONS = {
    "epochs": "epochs",
    "lr": "learning_rate",
    "batch_size": "batch_size",
    "max_length": "max_length",
}


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    for noisy in ("httpx", "httpx2", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    match args[:1]:
        case ["predict"]:
            parser = _predict_parser()
            _predict(parser, parser.parse_args(args[1:]))
        case ["status"]:
            parser = _status_parser()
            _status(parser, parser.parse_args(args[1:]))
        case _:
            parser = _run_parser()
            _run(parser, parser.parse_args(args))


def _run_name(value: str) -> str:
    if not re.fullmatch(NAME_PATTERN, value):
        raise argparse.ArgumentTypeError(f"invalid run name {value!r}: use letters, digits, ._-")
    return value


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--name", required=True, type=_run_name, help="run name; data lives in RUNS_DIR/NAME"
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bgen",
        description="Build a ModernBERT classifier from a natural-language rule. "
        "Re-running with the same --name resumes; other subcommands: predict, status.",
    )
    _common(parser)
    add = parser.add_argument
    add("--rule", help="what the classifier should do (required for a new run)")
    add("--model", help="base, large, a preset key or a Hugging Face model id (default: base)")
    add("--llm", help=f"generator LLM spec (default: {DEFAULT_LLM})")
    add("--eval-llm", help="LLM that writes the test set and verifies train data (default: --llm)")
    add("--planner-llm", help="LLM for spec, plan and judge (default: --llm)")
    add("--web", action=argparse.BooleanOptionalAction, default=None, help="web research")
    add("--verify", action=argparse.BooleanOptionalAction, default=None, help="cross-check data")
    add(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="start each round after the first from the previous round's model (default: on)",
    )
    add(
        "--oracle",
        metavar="REF",
        help="'package.module:attr' of a rule oracle whose labels replace the LLM's, "
        "e.g. bertgen.oracles.haiku:HaikuOracle; fixed once data is generated",
    )
    add(
        "--corpus",
        type=Path,
        metavar="PATH",
        help="shuffled .txt or .jsonl(.gz) file whose passages the --oracle labels, "
        "replacing LLM-written data; fixed once data exists",
    )
    add("--corpus-max-chars", type=int, help="maximum passage length (default: 300)")
    add("--examples", type=int, help="initial train+valid examples")
    add("--test-examples", type=int, help="test examples")
    add("--examples-per-round", type=int, help="examples added on a more_data verdict")
    add("--max-rounds", type=int, help="upper bound on training rounds")
    add("--target", type=float, help="primary score that ends the run")
    add("--valid-ratio", type=float, help="fraction of generated data held out for validation")
    add("--workers", type=int, help="concurrent LLM calls")
    add("--seed", type=int)
    add("--epochs", type=float, help="first-round training settings; later rounds follow the judge")
    add("--lr", type=float, help="learning rate")
    add("--batch-size", type=int)
    add("--max-length", type=int, help="max tokens per input during training")
    add(
        "--stop-after",
        type=Stage,
        choices=_STOP_STAGES,
        help="stop after this stage and print its artifact, e.g. to review spec.json; "
        "re-run without it to continue",
    )
    return parser


def _predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bgen predict", description="Classify texts.")
    _common(parser)
    parser.add_argument(
        "texts", nargs="*", metavar="TEXT", help="texts to classify; one per line on stdin if none"
    )
    return parser


def _status_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bgen status", description="Show rounds and metrics.")
    _common(parser)
    return parser


def _given(args: argparse.Namespace, names: dict[str, str]) -> dict[str, object]:
    values: dict[str, object] = vars(args)
    return {field: values[opt] for opt, field in names.items() if values[opt] is not None}


def _run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    name: str = args.name
    runs_dir: Path = args.runs_dir
    options = _given(args, {field: field for field in _RUN_FIELDS})
    hparams = _given(args, _HPARAM_OPTIONS)
    state_path = runs_dir / name / "run.json"

    try:
        if state_path.exists():
            stored = load_state(runs_dir / name).config
            if options.get("rule", stored.rule) != stored.rule:
                parser.error(f"run {name!r} exists with a different rule; choose another --name")
            if (
                options.get("oracle", stored.oracle) != stored.oracle
                and RunPaths(runs_dir / name).db.exists()
            ):
                parser.error(f"run {name!r} already has data labeled with oracle {stored.oracle!r}")
            if RunPaths(runs_dir / name).db.exists() and (
                not _same_file(options.get("corpus", stored.corpus), stored.corpus)
                or options.get("corpus_max_chars", stored.corpus_max_chars)
                != stored.corpus_max_chars
            ):
                parser.error(
                    f"run {name!r} already has data from corpus {stored.corpus} "
                    f"split at {stored.corpus_max_chars} characters"
                )
            merged = {**stored.hparams.model_dump(exclude_none=True), **hparams}
            config = RunConfig.model_validate(
                {**stored.model_dump(), **options, "runs_dir": runs_dir, "hparams": merged}
            )
        elif "rule" not in options:
            parser.error("--rule is required for a new run")
        else:
            config = RunConfig.model_validate(
                {**options, "name": name, "runs_dir": runs_dir, "hparams": hparams}
            )
    except ValueError as error:
        parser.error(str(error))
    stop_after: Stage | None = args.stop_after
    try:
        print(run(config, stop_after=stop_after))
    except (LLMError, OracleError, CorpusError) as error:
        logging.getLogger(__name__).error("%s", error)
        raise SystemExit(1) from None


def _same_file(given: object, stored: Path | None) -> bool:
    if isinstance(given, Path) and stored is not None:
        return given.resolve() == stored.resolve()
    return given == stored


def _existing_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> RunPaths:
    paths = RunPaths(args.runs_dir / args.name)
    if not paths.state.exists():
        parser.error(f"no run named {args.name!r} in {args.runs_dir}")
    return paths


def _predict(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    paths = _existing_run(parser, args)
    if not paths.model.exists():
        parser.error(f"run {args.name!r} has no trained model yet")
    predictor = Predictor(load_spec(paths.root), paths.model)
    texts: list[str] = args.texts
    batches: Iterable[list[str]] = [texts] if texts else _stdin_batches(sys.stdin)
    for batch in batches:
        for scored in predictor.scored(batch):
            print(json.dumps(_prediction(scored), ensure_ascii=False), flush=True)


def _stdin_batches(stream: TextIO) -> Iterator[list[str]]:
    """Non-empty lines of `stream`, one per batch when interactive so each answer comes back
    immediately, otherwise in batches of BATCH_SIZE."""
    size = 1 if stream.isatty() else BATCH_SIZE
    texts = (line.rstrip("\r\n") for line in stream if line.strip())
    for batch in batched(texts, size, strict=False):
        yield list(batch)


def _prediction(scored: Scored) -> dict[str, object]:
    example = scored.example
    if example.spans:
        spans = [
            {
                "text": example.text[s.start : s.end],
                "label": s.label,
                "start": s.start,
                "end": s.end,
                "score": round(score, 4),
            }
            for s, score in zip(example.spans, scored.span_scores, strict=True)
        ]
        return {"text": example.text, "spans": spans}
    result: dict[str, object] = {"text": example.text, "labels": example.labels}
    if scored.scores:
        result["scores"] = {label: round(p, 4) for label, p in scored.scores.items()}
    return result


def _status(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    paths = _existing_run(parser, args)
    state = load_state(paths.root)
    print(f"{state.config.name}: {state.config.rule}")
    if paths.spec.exists():
        spec = load_spec(paths.root)
        print(f"task: {spec.kind} {spec.label_names} ({spec.language})")
    for split, stats in state.oracle_agreement.items():
        print(f"oracle on {split}: {describe_agreement(stats)}")
    best = paths.model.resolve().parent.name if paths.model.exists() else None
    for result in load_rounds(paths.root):
        metrics = result.metrics
        decision = result.verdict.decision if result.verdict else "-"
        marker = " *" if str(result.round) == best else ""
        print(
            f"round {result.round}: primary {metrics.primary:.4f}  "
            f"accuracy {metrics.accuracy:.4f}  n_test {metrics.n_test}  -> {decision}{marker}"
        )
