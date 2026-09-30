"""Real text labeled by a rule oracle, as an alternative to LLM-synthesized examples.

A corpus file is read in file order, so it should be shuffled beforehand
(see examples/haiku575/prepare_corpus.sql).
"""

import gzip
import hashlib
import logging
import re
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple, TextIO

from pydantic import BaseModel

from bertgen.oracles import Oracle
from bertgen.text import normalize
from bertgen.types import NONE_KEY, Example, TaskKind, TaskSpec

MIN_PASSAGE_CHARS = 10
TEST_SHARE = 0.2
NONE_SHARE = 0.25
PROGRESS_EVERY = 100_000
STALL_LIMIT = 1_000_000

_TERMINATORS = "。．！？!?"
_SENTENCE_END = re.compile(f"(?<=[{_TERMINATORS}])(?![{_TERMINATORS}」』）)\\]])|(?<=\\.)(?=\\s)")

log = logging.getLogger(__name__)


class CorpusError(ValueError):
    """The corpus file has an unknown format or yields no usable passage."""


class CorpusFormat(StrEnum):
    TXT = ".txt"
    JSONL = ".jsonl"


class Pool(StrEnum):
    """Disjoint halves of a corpus: test rows come from TEST, train/valid rows from TRAIN."""

    TEST = "test"
    TRAIN = "train"


class _Record(BaseModel):
    text: str


def corpus_format(path: Path) -> CorpusFormat:
    """The document format of `path` by suffix: .txt or .jsonl, optionally followed by .gz."""
    suffixes = path.suffixes[:-1] if path.suffixes[-1:] == [".gz"] else path.suffixes
    try:
        return CorpusFormat(suffixes[-1] if suffixes else "")
    except ValueError:
        raise CorpusError(f"corpus must be .txt or .jsonl (optionally .gz): {path}") from None


class Passage(NamedTuple):
    pool: Pool
    text: str


def read_passages(path: Path, max_chars: int) -> Generator[Passage]:
    """Passages of at most `max_chars` characters from every document of `path`, in file order.

    A .txt file holds one document per line, a .jsonl file one object with a "text" field per
    line. Documents are split on newlines, then sentences are packed into passages. Every
    passage carries the pool of its whole document.
    """
    fmt = corpus_format(path)
    with _open(path) as lines:
        for line in lines:
            if not line.strip():
                continue
            document = _Record.model_validate_json(line).text if fmt is CorpusFormat.JSONL else line
            pool = pool_of(document)
            yield from (Passage(pool, text) for text in split_passages(document, max_chars))


def split_passages(document: str, max_chars: int) -> Iterator[str]:
    """Pack consecutive sentences of each line into passages of at most `max_chars`; a longer
    sentence is cut hard. Passages shorter than MIN_PASSAGE_CHARS are dropped."""
    for line in document.splitlines():
        passage = ""
        for sentence in _SENTENCE_END.split(line.strip()):
            if len(passage) + len(sentence) <= max_chars:
                passage += sentence
                continue
            yield from _kept(passage)
            chunks = [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]
            yield from (p for chunk in chunks[:-1] for p in _kept(chunk))
            passage = chunks[-1]
        yield from _kept(passage)


def pool_of(document: str) -> Pool:
    """The pool of `document`, fixed by a hash of its normalized form."""
    digest = hashlib.blake2b(normalize(document).encode(), digest_size=8).digest()
    return Pool.TEST if int.from_bytes(digest) / 2**64 < TEST_SHARE else Pool.TRAIN


def corpus_weights(spec: TaskSpec, weights: dict[str, float]) -> dict[str, float]:
    """`weights` with a NONE_KEY share of NONE_SHARE added for SPAN and MULTILABEL tasks that
    lack one, so span-less and label-less passages are sampled as negatives."""
    if spec.kind not in (TaskKind.SPAN, TaskKind.MULTILABEL) or weights.get(NONE_KEY, 0.0) > 0:
        return weights
    total = sum(weights.values())
    return {key: w / total * (1 - NONE_SHARE) for key, w in weights.items()} | {
        NONE_KEY: NONE_SHARE
    }


def bucket_of(example: Example) -> str:
    """The quota bucket of `example`: its first span label, else its first label, else NONE_KEY."""
    if example.spans:
        return example.spans[0].label
    return example.labels[0] if example.labels else NONE_KEY


def label_quotas(weights: dict[str, float], count: int) -> dict[str, int]:
    """Split `count` in proportion to `weights` with largest-remainder rounding."""
    total = sum(weights.values())
    assert total > 0, "label quotas need a positive weight"
    raw = {key: weight / total * count for key, weight in weights.items()}
    result = {key: int(value) for key, value in raw.items()}
    by_remainder = sorted(raw, key=lambda key: raw[key] - result[key], reverse=True)
    for key in by_remainder[: count - sum(result.values())]:
        result[key] += 1
    return result


@dataclass(frozen=True)
class Sampled:
    """Examples collected by `sample`; `stalled` when it stopped after STALL_LIMIT passages
    without filling any bucket rather than at the end of the passages."""

    examples: list[Example]
    scanned: int
    stalled: bool


def sample(
    passages: Iterable[Passage],
    oracle: Oracle,
    pool: Pool,
    quotas: dict[str, int],
    seen: set[str],
) -> Sampled:
    """Oracle-labeled passages of `pool`, collected until every bucket's quota is filled.

    Passages whose normalized text is in `seen` are skipped before the oracle is called, and
    the normalized text of every collected passage is added to `seen`. Abstentions and
    examples whose bucket is full or has no quota are dropped. Returns fewer examples than
    requested when the passages run out or STALL_LIMIT passages in a row add nothing.
    """
    open_ = {key: n for key, n in quotas.items() if n > 0}
    collected: list[Example] = []
    scanned = idle = 0
    for passage in passages:
        if not open_:
            break
        if idle >= STALL_LIMIT:
            return Sampled(collected, scanned, stalled=True)
        scanned += 1
        idle += 1
        if not scanned % PROGRESS_EVERY:
            log.info("scanned %d passages, %d %s examples", scanned, len(collected), pool)
        if passage.pool is not pool:
            continue
        key = normalize(passage.text)
        if key in seen:
            continue
        example = oracle.label(passage.text)
        if example is None:
            continue
        assert example.text == passage.text, "oracle must not rewrite the text"
        bucket = bucket_of(example)
        if bucket not in open_:
            continue
        collected.append(example)
        seen.add(key)
        idle = 0
        open_[bucket] -= 1
        if not open_[bucket]:
            del open_[bucket]
    return Sampled(collected, scanned, stalled=False)


def _open(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _kept(passage: str) -> Iterator[str]:
    passage = passage.strip()
    if len(passage) >= MIN_PASSAGE_CHARS:
        yield passage
