import json
from pathlib import Path

import pytest
from tiny_model import build_tokenizer, save_tiny_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    pipeline,
)

from bertgen.stages import train as train_module
from bertgen.stages._encoding import (
    IGNORE_INDEX,
    encode,
    label_maps,
    split_characters,
    unreachable_spans,
)
from bertgen.stages.evaluate import decode_bio, evaluate, predict
from bertgen.stages.train import train
from bertgen.types import Example, LabelDef, Span, TaskKind, TaskSpec, TrainConfig


def make_spec(kind: TaskKind, names: list[str]) -> TaskSpec:
    return TaskSpec(
        rule="r",
        kind=kind,
        language="en",
        labels=[LabelDef(name=n, description=n) for n in names],
        input_description="text",
        decision_criteria="c",
    )


def quick_config(base: str) -> TrainConfig:
    return TrainConfig(
        base_model=base, epochs=2, batch_size=4, max_length=64, learning_rate=1e-3, warmup_ratio=0.0
    )


@pytest.fixture(scope="module")
def base_model(tmp_path_factory: pytest.TempPathFactory) -> str:
    return save_tiny_model(tmp_path_factory.mktemp("base"))


def test_label_maps() -> None:
    seq = label_maps(make_spec(TaskKind.MULTICLASS, ["a", "b", "c"]))
    assert seq.id2label == {0: "a", 1: "b", 2: "c"}
    span = label_maps(make_spec(TaskKind.SPAN, ["X", "Y"]))
    assert list(span.label2id) == ["O", "B-X", "I-X", "B-Y", "I-Y"]


def test_span_encoding_aligns_bio() -> None:
    spec = make_spec(TaskKind.SPAN, ["NUM"])
    maps = label_maps(spec)
    ex = Example(text="ab 123 c", spans=[Span(start=3, end=6, label="NUM")])
    labels = encode(spec, build_tokenizer(), ex, maps, 64)["labels"]
    assert isinstance(labels, list)
    tags = [None if i == IGNORE_INDEX else maps.id2label[int(i)] for i in labels]
    assert tags == [None, *["O"] * 3, "B-NUM", "I-NUM", "I-NUM", "O", "O", None]


def test_encoding_truncates_and_multilabel_vector() -> None:
    spec = make_spec(TaskKind.MULTILABEL, ["a", "b", "c"])
    ex = Example(text="x" * 100, labels=["a", "c"])
    features = encode(spec, build_tokenizer(), ex, label_maps(spec), 16)
    input_ids = features["input_ids"]
    assert isinstance(input_ids, list)
    assert len(input_ids) == 16
    assert features["labels"] == [1.0, 0.0, 1.0]


def test_decode_bio_merges_and_orphan_i_starts_span() -> None:
    text = "ab cd ef"
    offsets = [(0, 0), (0, 2), (3, 5), (6, 8), (0, 0)]
    special = [1, 0, 0, 0, 1]
    tags = ["O", "I-X", "I-X", "B-Y", "O"]
    assert decode_bio(text, tags, offsets, special) == [
        Span(start=0, end=5, label="X"),
        Span(start=6, end=8, label="Y"),
    ]


@pytest.mark.parametrize(
    ("kind", "names"),
    [
        (TaskKind.BINARY, ["neg", "pos"]),
        (TaskKind.MULTICLASS, ["a", "b", "c"]),
        (TaskKind.MULTILABEL, ["a", "b", "c"]),
    ],
)
def test_train_sequence_kinds(
    kind: TaskKind, names: list[str], base_model: str, tmp_path: Path
) -> None:
    spec = make_spec(kind, names)

    def make(i: int) -> Example:
        chosen = [names[i % len(names)]]
        if kind is TaskKind.MULTILABEL and i % 2:
            chosen.append(names[-1])
        return Example(text=f"sample text {i} {chosen[0]}", labels=sorted(set(chosen)))

    data = [make(i) for i in range(20)]
    out = train(spec, data, data[:6], quick_config(base_model), tmp_path / "model")
    assert not (out / ".checkpoints").exists()

    model = AutoModelForSequenceClassification.from_pretrained(out)
    assert model.config.id2label == dict(enumerate(names))
    if kind is TaskKind.MULTILABEL:
        assert model.config.problem_type == "multi_label_classification"

    pipe = pipeline("text-classification", model=str(out), top_k=None)
    scores = pipe("sample text 1")
    assert isinstance(scores, list)
    first = scores[0]
    assert isinstance(first, list)
    assert all(f"'label': '{name}'" in str(first) for name in names)

    predictions = predict(spec, out, [e.text for e in data])
    assert len(predictions) == len(data)
    metrics = evaluate(spec, out, data)
    assert metrics.n_test == 20
    assert 0.0 <= metrics.primary <= 1.0
    assert [s.label for s in metrics.per_label] == names
    assert sum(s.support for s in metrics.per_label) >= 20
    assert len(metrics.errors) <= 20


def test_train_span_and_evaluate(base_model: str, tmp_path: Path) -> None:
    spec = make_spec(TaskKind.SPAN, ["NUM"])
    data = []
    for i in range(20):
        text = f"call {i}{i} now"
        start = 5
        data.append(
            Example(text=text, spans=[Span(start=start, end=start + len(f"{i}{i}"), label="NUM")])
        )
    out = train(spec, data, data[:6], quick_config(base_model), tmp_path / "model")

    model = AutoModelForTokenClassification.from_pretrained(out)
    assert model.config.id2label[1] == "B-NUM"
    pipe = pipeline("token-classification", model=str(out))
    assert isinstance(pipe("call 11 now"), list)

    metrics = evaluate(spec, out, data)
    assert metrics.n_test == 20
    assert [s.label for s in metrics.per_label] == ["NUM"]
    assert metrics.per_label[0].support == 20
    assert 0.0 <= metrics.accuracy <= 1.0


def test_perfect_span_prediction_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    from bertgen.stages import evaluate as ev

    spec = make_spec(TaskKind.SPAN, ["X", "Y"])
    test = [
        Example(text="aa bb", spans=[Span(start=0, end=2, label="X")]),
        Example(text="cc dd", spans=[Span(start=3, end=5, label="Y")]),
    ]
    guesses = [
        Example(text="aa bb", spans=[Span(start=0, end=2, label="X")]),
        Example(text="cc dd", spans=[]),
    ]

    def fake_predict(*_args: object) -> list[Example]:
        return guesses

    monkeypatch.setattr(ev, "predict", fake_predict)

    def fake_tokenizer(*_args: object) -> PreTrainedTokenizerFast:
        return build_tokenizer()

    monkeypatch.setattr(ev.AutoTokenizer, "from_pretrained", fake_tokenizer)
    metrics = ev.evaluate(spec, Path("."), test)
    assert metrics.accuracy == 0.5
    assert metrics.primary == pytest.approx(2 / 3)
    assert len(metrics.errors) == 1
    assert metrics.unreachable_spans == 0


def test_macro_f1_ignores_labels_absent_from_gold_and_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bertgen.stages import evaluate as ev

    spec = make_spec(TaskKind.MULTICLASS, ["a", "b", "other"])
    test = [Example(text="x", labels=["a"]), Example(text="y", labels=["b"])]

    def fake_predict(*_args: object) -> list[Example]:
        return test

    monkeypatch.setattr(ev, "predict", fake_predict)
    metrics = ev.evaluate(spec, Path("."), test)
    assert metrics.primary == 1.0
    assert [s.support for s in metrics.per_label] == [1, 1, 0]


def word_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


def test_unreachable_spans_counts_boundaries_inside_tokens() -> None:
    text = "hello world"
    examples = [
        Example(text=text, spans=[Span(start=0, end=5, label="X")]),
        Example(text=text, spans=[Span(start=0, end=3, label="X")]),
        Example(text=text, spans=[Span(start=7, end=11, label="X")]),
        Example(text=text),
    ]
    assert unreachable_spans(word_tokenizer(), examples, 16) == 2


def test_split_characters_makes_every_boundary_reachable(tmp_path: Path) -> None:
    text = "hello world"
    examples = [Example(text=text, spans=[Span(start=0, end=3, label="X")])]
    tokenizer = word_tokenizer()
    split_characters(tokenizer)
    assert unreachable_spans(tokenizer, examples, 32) == 0
    tokenizer.save_pretrained(tmp_path)
    reloaded = AutoTokenizer.from_pretrained(tmp_path)
    assert unreachable_spans(reloaded, examples, 32) == 0


def pre_tokenizer_of(model_dir: Path) -> dict[str, object]:
    pre = json.loads((model_dir / "tokenizer.json").read_text())["pre_tokenizer"]
    assert isinstance(pre, dict)
    return pre


def test_align_to_spans_splits_characters_once() -> None:
    text = "hello world " * 8
    examples = [Example(text=text, spans=[Span(start=90, end=93, label="X")])]
    tokenizer = word_tokenizer()
    train_module._align_to_spans(tokenizer, examples, 8)  # pyright: ignore[reportPrivateUsage]
    once = tokenizer.backend_tokenizer.to_str()
    assert unreachable_spans(tokenizer, examples, 8) == 1
    train_module._align_to_spans(tokenizer, examples, 8)  # pyright: ignore[reportPrivateUsage]
    assert tokenizer.backend_tokenizer.to_str() == once


def test_warm_start_continues_from_saved_model(
    base_model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_spec(TaskKind.SPAN, ["NUM"])
    data = [
        Example(text="x" * 70 + f"{i}{i}", spans=[Span(start=70, end=72, label="NUM")])
        if i % 2
        else Example(text=f"call {i}{i} now", spans=[Span(start=5, end=7, label="NUM")])
        for i in range(1, 10)
    ]
    sources: list[str] = []
    load_tokenizer = AutoTokenizer.from_pretrained

    def recording(source: str) -> object:
        sources.append(source)
        return load_tokenizer(source)

    monkeypatch.setattr(train_module.AutoTokenizer, "from_pretrained", recording)
    first = train(spec, data, data[:4], quick_config(base_model), tmp_path / "first")
    second = train(spec, data, data[:4], quick_config(base_model), tmp_path / "second", first)

    assert sources == [base_model, str(first)]
    assert pre_tokenizer_of(first) == pre_tokenizer_of(second) == pre_tokenizer_of(Path(base_model))
    assert pre_tokenizer_of(second)["type"] == "Split"
    assert AutoModelForTokenClassification.from_pretrained(second).config.id2label[1] == "B-NUM"
