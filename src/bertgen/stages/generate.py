"""Stages 3 and 4: synthesize labeled examples and cross-check them with a second LLM."""

import json
import logging
import math
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from bertgen.llm.base import LLM, LLMError
from bertgen.oracles import Oracle
from bertgen.stages.prompts import (
    Cell,
    describe_example,
    generate_system,
    generate_user,
    verify_system,
    verify_user,
)
from bertgen.text import normalize
from bertgen.types import (
    NONE_KEY,
    DataPlan,
    Draft,
    Example,
    OracleOutcome,
    Span,
    TaskKind,
    TaskSpec,
)

log = logging.getLogger(__name__)

_MIN_BATCH = 5
_LOOKALIKE_RATE = 0.3
_MAX_FAILED_WAVES = 2
_HARD_CASES_PER_CALL = 4
_MAX_SEEDS_PER_CALL = 3
_AVOID_SAMPLES = 4
_AVOID_RECENT = 4
_AVOID_CHARS = 80


class ClassItem(BaseModel):
    text: str
    labels: list[str]


class ClassBatch(BaseModel):
    items: list[ClassItem]


class SpanText(BaseModel):
    text: str
    label: str


class SpanItem(BaseModel):
    text: str
    spans: list[SpanText]


class SpanBatch(BaseModel):
    items: list[SpanItem]


class ClassVerdict(BaseModel):
    index: int
    labels: list[str]


class ClassVerdictBatch(BaseModel):
    items: list[ClassVerdict]


class SpanVerdict(BaseModel):
    index: int
    spans: list[SpanText]


class SpanVerdictBatch(BaseModel):
    items: list[SpanVerdict]


def sample_cell(plan: DataPlan, rng: random.Random) -> Cell:
    """Draw a label by plan weight, a value per axis, and occasionally a look-alike target."""
    labels = [label for label, weight in plan.label_weights.items() if weight > 0]
    weights = [plan.label_weights[label] for label in labels]
    label = rng.choices(labels, weights)[0]
    axes = {axis.name: rng.choice(axis.values) for axis in plan.axes}
    lookalike = label != NONE_KEY and rng.random() < _LOOKALIKE_RATE
    return Cell(label, axes, lookalike)


def generate(
    spec: TaskSpec,
    plan: DataPlan,
    llm: LLM,
    count: int,
    *,
    focus: Sequence[str],
    seen: set[str],
    shown: Sequence[str],
    workers: int,
    seed: int = 0,
    batch_size: int = 12,
    max_calls: int | None = None,
) -> Iterator[Example]:
    """Yield up to `count` examples whose normalized text is not in `seen` or already yielded.

    Prompts quote short snippets of `shown` and of this call's own output to push for diversity.
    Cells and prompts are drawn from a seeded RNG in submission order. Stops early with a warning
    when `max_calls` LLM calls (default: three times the nominal need) are used up or when two
    consecutive waves of calls all fail.
    """
    rng = random.Random(seed)
    budget = max_calls or max(workers, 3 * math.ceil(count / batch_size))
    system = generate_system(spec)
    keys = {normalize(text) for text in seen}
    pool_texts = list(shown)
    recent: list[str] = []
    emitted = 0
    calls = 0
    failed_waves = 0

    with ThreadPoolExecutor(workers) as pool:
        while emitted < count and calls < budget and failed_waves < _MAX_FAILED_WAVES:
            wave = min(workers, budget - calls)
            n = min(batch_size, max(_MIN_BATCH, math.ceil((count - emitted) * 1.3 / wave)))
            users = []
            for _ in range(wave):
                cells = [sample_cell(plan, rng) for _ in range(n)]
                snippets = _avoid_snippets(rng, pool_texts, recent)
                hard = rng.sample(plan.hard_cases, min(_HARD_CASES_PER_CALL, len(plan.hard_cases)))
                users.append(
                    generate_user(
                        spec,
                        plan,
                        cells=cells,
                        hard_cases=hard,
                        focus=focus,
                        avoid=snippets,
                        seeds=_sample_seeds(rng, plan.seeds),
                    )
                )
            futures = [pool.submit(_request_batch, spec, llm, system, user) for user in users]
            calls += wave
            results = [future.result() for future in futures]
            failed_waves = failed_waves + 1 if all(r is None for r in results) else 0
            for result in results:
                for example in result or []:
                    key = normalize(example.text)
                    if key in keys:
                        continue
                    keys.add(key)
                    recent.append(example.text)
                    emitted += 1
                    yield example
                    if emitted >= count:
                        return

    if failed_waves >= _MAX_FAILED_WAVES:
        log.warning("generation stopped: every call in %d consecutive waves failed", failed_waves)
    else:
        log.warning(
            "generation budget exhausted: %d/%d examples after %d calls", emitted, count, calls
        )


def _avoid_snippets(rng: random.Random, pool: Sequence[str], recent: list[str]) -> list[str]:
    picked = rng.sample(pool, min(_AVOID_SAMPLES, len(pool)))
    picked += recent[-_AVOID_RECENT:]
    return [text[:_AVOID_CHARS] for text in picked]


def _sample_seeds(rng: random.Random, seeds: Sequence[str]) -> list[str]:
    if not seeds:
        return []
    return rng.sample(list(seeds), rng.randint(0, min(_MAX_SEEDS_PER_CALL, len(seeds))))


def _request_batch(spec: TaskSpec, llm: LLM, system: str, user: str) -> list[Example] | None:
    """Examples parsed from one call, or None when the call failed."""
    try:
        if spec.kind is TaskKind.SPAN:
            return span_examples(spec, llm.ask(system, user, SpanBatch))
        return class_examples(spec, llm.ask(system, user, ClassBatch))
    except LLMError as error:
        log.warning("generation call failed: %s", error)
        return None


def class_examples(spec: TaskSpec, batch: ClassBatch) -> list[Example]:
    """Map a classification batch to examples, dropping malformed items."""
    known = set(spec.label_names)
    exclusive = spec.kind in (TaskKind.BINARY, TaskKind.MULTICLASS)
    examples: list[Example] = []
    for item in batch.items:
        text = item.text.strip()
        labels = list(dict.fromkeys(item.labels))
        if not text or not known.issuperset(labels):
            continue
        if exclusive and len(labels) != 1:
            continue
        examples.append(Example(text=text, labels=labels))
    return examples


def span_examples(spec: TaskSpec, batch: SpanBatch) -> list[Example]:
    """Locate span substrings in their texts.

    Items are dropped when a span is unlocatable, has an unknown label, or quotes a substring
    that occurs in the text a different number of times than it is listed (its position would
    be ambiguous).
    """
    known = set(spec.label_names)
    examples: list[Example] = []
    for item in batch.items:
        text = item.text.strip()
        if not text or not all(s.label in known for s in item.spans):
            continue
        if not _occurrences_match(text, item.spans):
            continue
        spans = locate_spans(text, item.spans)
        if spans is not None:
            examples.append(Example(text=text, spans=spans))
    return examples


def _occurrences_match(text: str, items: list[SpanText]) -> bool:
    listed = Counter(item.text for item in items)
    return all(text.count(sub) == n for sub, n in listed.items())


def locate_spans(text: str, items: list[SpanText]) -> list[Span] | None:
    """Offsets of each substring at its first occurrence not overlapping an earlier span."""
    spans: list[Span] = []
    for item in items:
        start = _free_occurrence(text, item.text, spans)
        if start is None:
            return None
        spans.append(Span(start=start, end=start + len(item.text), label=item.label))
    return sorted(spans, key=lambda span: span.start)


def _free_occurrence(text: str, sub: str, taken: list[Span]) -> int | None:
    if not sub:
        return None
    start = text.find(sub)
    while start >= 0:
        end = start + len(sub)
        if all(end <= span.start or start >= span.end for span in taken):
            return start
        start = text.find(sub, start + 1)
    return None


def consult_oracle(oracle: Oracle, examples: Iterable[Example]) -> Iterator[Draft]:
    """Pair each example with the oracle's outcome; labeled ones carry the oracle's labels.

    Abstained examples keep the LLM's label. Disagreements are logged at INFO. The oracle is
    called sequentially.
    """
    for example in examples:
        gold = oracle.label(example.text)
        assert gold is None or gold.text == example.text, "oracle must not rewrite the text"
        if gold is None:
            yield Draft(example, example, OracleOutcome.ABSTAINED)
        elif _signature(gold) == _signature(example):
            yield Draft(gold, example, OracleOutcome.AGREED)
        else:
            log.info(
                "oracle relabeled %r: LLM %s, oracle %s",
                example.text[:200],
                describe_example(example),
                describe_example(gold),
            )
            yield Draft(gold, example, OracleOutcome.DISAGREED)


def verify(
    spec: TaskSpec,
    examples: list[Example],
    llm: LLM,
    *,
    workers: int,
    batch_size: int = 20,
) -> list[Example]:
    """Keep the examples whose label an independent annotation reproduces exactly."""
    chunks = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]
    system = verify_system(spec)
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(_verify_chunk, spec, llm, system, chunk) for chunk in chunks]
        results = [future.result() for future in futures]
    kept = [example for chunk in results for example in chunk]
    rate = len(kept) / len(examples) if examples else 1.0
    log.info("verifier agreement %.1f%% (%d/%d)", 100 * rate, len(kept), len(examples))
    return kept


def _verify_chunk(spec: TaskSpec, llm: LLM, system: str, chunk: list[Example]) -> list[Example]:
    user = verify_user([example.text for example in chunk])
    try:
        annotations = _annotate(spec, llm, system, user)
    except LLMError as error:
        log.warning("verification call failed, dropping %d examples: %s", len(chunk), error)
        return []
    return [
        example
        for i, example in enumerate(chunk)
        if i in annotations and annotations[i] == _signature(example)
    ]


type Signature = tuple[str, ...]


def _annotate(spec: TaskSpec, llm: LLM, system: str, user: str) -> dict[int, Signature]:
    result: dict[int, Signature] = {}
    if spec.kind is TaskKind.SPAN:
        for span_item in llm.ask(system, user, SpanVerdictBatch).items:
            result.setdefault(span_item.index, _span_key(span_item.spans))
    else:
        for class_item in llm.ask(system, user, ClassVerdictBatch).items:
            result.setdefault(class_item.index, _label_key(class_item.labels))
    return result


def _span_key(spans: list[SpanText]) -> Signature:
    """Multiset of (substring, label) pairs, so repeated occurrences must agree in number."""
    return tuple(sorted(json.dumps([s.text, s.label], ensure_ascii=False) for s in spans))


def _label_key(labels: list[str]) -> Signature:
    return tuple(sorted(set(labels)))


def _signature(example: Example) -> Signature:
    if example.spans:
        return _span_key(
            [SpanText(text=example.text[s.start : s.end], label=s.label) for s in example.spans]
        )
    return _label_key(example.labels)
