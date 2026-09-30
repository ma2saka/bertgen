import gzip
import json
from pathlib import Path

import pytest
from fakes import FakeOracle, FakeSpanOracle

import bertgen.corpus
from bertgen.corpus import (
    MIN_PASSAGE_CHARS,
    NONE_SHARE,
    CorpusError,
    Passage,
    Pool,
    corpus_format,
    corpus_weights,
    label_quotas,
    pool_of,
    read_passages,
    sample,
    split_passages,
)
from bertgen.text import normalize
from bertgen.types import NONE_KEY, LabelDef, TaskKind, TaskSpec

SPEC = TaskSpec(
    rule="r",
    kind=TaskKind.BINARY,
    language="en",
    labels=[LabelDef(name="pos", description="p"), LabelDef(name="neg", description="n")],
    input_description="text",
    decision_criteria="c",
)


def in_pool(pool: Pool, texts: list[str]) -> list[str]:
    return [t for t in texts if pool_of(t) is pool]


def documents(texts: list[str]) -> list[Passage]:
    return [Passage(pool_of(t), t) for t in texts]


def test_split_packs_sentences_per_line() -> None:
    document = "一つ目の文です。二つ目の文です！三つ目？\n短い\n「かぎ括弧の文。」と言った。"
    assert list(split_passages(document, 16)) == [
        "一つ目の文です。二つ目の文です！",
        "「かぎ括弧の文。」と言った。",
    ]
    assert list(split_passages(document, 100)) == [
        "一つ目の文です。二つ目の文です！三つ目？",
        "「かぎ括弧の文。」と言った。",
    ]


def test_split_cuts_long_sentences_and_drops_short_passages() -> None:
    sentence = "あ" * 25 + "。"
    passages = list(split_passages(f"{sentence}短い文。", 10))
    assert passages == ["あ" * 10, "あ" * 10, "あああああ。短い文。"]
    assert all(len(p) >= MIN_PASSAGE_CHARS for p in split_passages("abc. def.\n" * 3, 40))


def test_split_packs_ascii_sentences_but_keeps_decimals() -> None:
    document = "Pi is about 3.14 here. The next sentence follows. The last one!"
    assert list(split_passages(document, 30)) == [
        "Pi is about 3.14 here.",
        "The next sentence follows.",
        "The last one!",
    ]


@pytest.mark.parametrize("name", ["c.txt", "c.jsonl", "c.txt.gz", "c.jsonl.gz"])
def test_read_passages_formats(tmp_path: Path, name: str) -> None:
    documents = ["first document is long enough.", "second one\nhas two lines of text"]
    path = tmp_path / name
    body = (
        "\n".join(json.dumps({"text": d, "source": "x"}) for d in documents)
        if ".jsonl" in name
        else "\n".join(d.replace("\n", " ") for d in documents)
    ) + "\n\n"
    if name.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(body)
    else:
        path.write_text(body, encoding="utf-8")
    passages = list(read_passages(path, 300))
    assert passages[0] == Passage(pool_of(documents[0]), "first document is long enough.")
    assert len(passages) == (3 if ".jsonl" in name else 2)
    if ".jsonl" in name:
        assert passages[1].pool is passages[2].pool is pool_of(documents[1])


def test_passages_of_one_document_share_its_pool(tmp_path: Path) -> None:
    lines = [". ".join(f"sentence {i} of document {d}" for i in range(6)) for d in range(200)]
    path = tmp_path / "c.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    passages = list(read_passages(path, 40))
    assert len(passages) > 2 * len(lines)
    assert all(p.pool is pool_of(next(li for li in lines if p.text in li)) for p in passages)
    assert {p.pool for p in passages} == set(Pool)


@pytest.mark.parametrize("name", ["c.csv", "c.gz", "corpus"])
def test_unknown_format_is_rejected(name: str) -> None:
    with pytest.raises(CorpusError):
        corpus_format(Path(name))


def test_pools_are_deterministic_disjoint_and_about_one_fifth_test() -> None:
    texts = [f"passage number {i}" for i in range(2000)]
    assert [pool_of(t) for t in texts] == [pool_of(t) for t in texts]
    assert all(pool_of(t) is pool_of(t.upper() + "  ") for t in texts[:50])
    share = len(in_pool(Pool.TEST, texts)) / len(texts)
    assert 0.15 < share < 0.25


def test_quotas_sum_to_count() -> None:
    assert label_quotas({"a": 0.5, "b": 0.5}, 5) in ({"a": 3, "b": 2}, {"a": 2, "b": 3})
    assert label_quotas({"a": 0.75, NONE_KEY: 0.25}, 8) == {"a": 6, NONE_KEY: 2}
    assert sum(label_quotas({"a": 0.3, "b": 0.3, "c": 0.4}, 7).values()) == 7
    assert label_quotas({"a": 0.6, "b": 0.6}, 10) == {"a": 5, "b": 5}


def test_corpus_weights_add_negatives_to_span_and_multilabel_tasks() -> None:
    assert corpus_weights(SPEC, {"pos": 0.5, "neg": 0.5}) == {"pos": 0.5, "neg": 0.5}
    for kind in (TaskKind.SPAN, TaskKind.MULTILABEL):
        spec = SPEC.model_copy(update={"kind": kind})
        weights = corpus_weights(spec, {"pos": 1.0})
        assert weights == pytest.approx({"pos": 1 - NONE_SHARE, NONE_KEY: NONE_SHARE})
        given = {"pos": 0.9, NONE_KEY: 0.1}
        assert corpus_weights(spec, given) == given


def test_sample_fills_quotas_from_its_pool_and_skips_seen() -> None:
    texts = [f"{'good' if i % 4 == 0 else 'bad'} text {i}" for i in range(400)]
    texts += ["apple is good 1", "apple is good 2"]
    train = in_pool(Pool.TRAIN, texts)
    seen = {normalize(train[0])}
    before = set(seen)

    examples = sample(documents(texts), FakeOracle(SPEC), Pool.TRAIN, {"pos": 5, "neg": 3}, seen)
    examples = examples.examples

    assert [e.labels[0] for e in examples].count("pos") == 5
    assert [e.labels[0] for e in examples].count("neg") == 3
    assert all(pool_of(e.text) is Pool.TRAIN for e in examples)
    assert train[0] not in {e.text for e in examples}
    assert not any("apple" in e.text for e in examples)
    assert seen == before | {normalize(e.text) for e in examples}

    again = sample(documents(texts), FakeOracle(SPEC), Pool.TRAIN, {"pos": 5, "neg": 3}, seen)
    assert not {e.text for e in again.examples} & {e.text for e in examples}


def test_sample_returns_short_when_passages_run_out() -> None:
    texts = [f"bad text {i}" for i in range(50)] + ["good text", "good text"]
    sampled = sample(documents(texts * 2), FakeOracle(SPEC), Pool.TEST, {"pos": 3, "neg": 2}, set())
    assert not sampled.stalled
    assert sampled.scanned == len(texts) * 2
    labels = [e.labels[0] for e in sampled.examples]
    assert labels.count("neg") == 2
    assert labels.count("pos") == (1 if pool_of("good text") is Pool.TEST else 0)


def test_sample_buckets_spans_by_label_or_none() -> None:
    span_spec = SPEC.model_copy(update={"kind": TaskKind.SPAN, "labels": SPEC.labels[:1]})
    texts = [f"line {i} {'river' if i % 2 else 'x'}" for i in range(200)]
    quotas = {"NUM": 4, NONE_KEY: 2}
    examples = sample(documents(texts), FakeSpanOracle(span_spec), Pool.TRAIN, quotas, set())
    examples = examples.examples
    assert sum(1 for e in examples if e.spans) == 4
    assert sum(1 for e in examples if not e.spans) == 2


def test_sample_stops_after_a_run_of_passages_without_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bertgen.corpus, "STALL_LIMIT", 30)
    texts = [f"bad text {i}" for i in range(300)] + ["good text"]
    sampled = sample(documents(texts), FakeOracle(SPEC), Pool.TRAIN, {"pos": 1, "neg": 1}, set())
    assert sampled.stalled
    assert [e.labels for e in sampled.examples] == [["neg"]]
    assert sampled.scanned < len(texts)
