"""Label maps and example-to-feature encoding shared by training and inference."""

from collections.abc import Sequence
from dataclasses import dataclass

from tokenizers import Regex, pre_tokenizers
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from bertgen.types import Example, Span, TaskKind, TaskSpec

IGNORE_INDEX = -100
OUTSIDE = "O"

Features = dict[str, list[int] | list[float] | int]


@dataclass(frozen=True)
class LabelMaps:
    id2label: dict[int, str]
    label2id: dict[str, int]


def label_maps(spec: TaskSpec) -> LabelMaps:
    """Class names for sequence tasks; `O`, `B-X`, `I-X` tags for SPAN tasks."""
    if spec.kind is TaskKind.SPAN:
        names = [OUTSIDE]
        for name in spec.label_names:
            names += [f"B-{name}", f"I-{name}"]
    else:
        names = spec.label_names
    return LabelMaps(dict(enumerate(names)), {name: i for i, name in enumerate(names)})


def bio_labels(
    spans: Sequence[Span],
    offsets: Sequence[tuple[int, int]],
    special: Sequence[int],
    label2id: dict[str, int],
) -> list[int]:
    """Token-level BIO ids from character spans; special tokens get IGNORE_INDEX."""
    ordered = sorted(spans, key=lambda span: span.start)
    labels: list[int] = []
    previous: int | None = None
    for (start, end), is_special in zip(offsets, special, strict=True):
        if is_special:
            labels.append(IGNORE_INDEX)
            continue
        hit = next((i for i, s in enumerate(ordered) if start < s.end and end > s.start), None)
        if hit is None:
            labels.append(label2id[OUTSIDE])
        else:
            prefix = "I" if hit == previous else "B"
            labels.append(label2id[f"{prefix}-{ordered[hit].label}"])
        previous = hit
    return labels


def encode(
    spec: TaskSpec,
    tokenizer: PreTrainedTokenizerBase,
    example: Example,
    maps: LabelMaps,
    max_length: int,
) -> Features:
    """Tokenize one example (truncated to max_length) and attach its labels."""
    enc = tokenizer(
        example.text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    features: Features = {
        "input_ids": list(enc["input_ids"]),
        "attention_mask": list(enc["attention_mask"]),
    }
    match spec.kind:
        case TaskKind.SPAN:
            features["labels"] = bio_labels(
                example.spans,
                [(int(s), int(e)) for s, e in enc["offset_mapping"]],
                enc["special_tokens_mask"],
                maps.label2id,
            )
        case TaskKind.MULTILABEL:
            chosen = {maps.label2id[name] for name in example.labels}
            features["labels"] = [float(i in chosen) for i in range(len(maps.label2id))]
        case TaskKind.BINARY | TaskKind.MULTICLASS:
            assert len(example.labels) == 1, f"expected exactly one label: {example.labels}"
            features["labels"] = maps.label2id[example.labels[0]]
    return features


def trimmed(text: str, span: Span) -> Span:
    """`span` without leading and trailing whitespace."""
    start, end = span.start, span.end
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start=start, end=end, label=span.label)


def unreachable_spans(
    tokenizer: PreTrainedTokenizerBase, examples: Sequence[Example], max_length: int
) -> int:
    """Count gold spans that cannot be decoded exactly from token-level tags: a boundary falls
    inside a token, or the span lies beyond `max_length` tokens."""
    count = 0
    for example in examples:
        if not example.spans:
            continue
        enc = tokenizer(
            example.text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        starts: set[int] = set()
        ends: set[int] = set()
        for (start, end), is_special in zip(
            enc["offset_mapping"], enc["special_tokens_mask"], strict=True
        ):
            if is_special or start == end:
                continue
            token = trimmed(example.text, Span(start=start, end=end, label=""))
            starts.update((start, token.start))
            ends.update((end, token.end))
        count += sum(1 for s in example.spans if s.start not in starts or s.end not in ends)
    return count


def split_characters(tokenizer: PreTrainedTokenizerBase) -> None:
    """Make every character its own pre-token so that token boundaries can express any span.

    The change lives in the fast tokenizer's pre-tokenizer and is kept by `save_pretrained`.
    """
    assert isinstance(tokenizer, PreTrainedTokenizerFast), (
        "character splitting needs a fast tokenizer"
    )
    backend = tokenizer.backend_tokenizer
    split = pre_tokenizers.Split(Regex("."), behavior="isolated")
    current = backend.pre_tokenizer
    backend.pre_tokenizer = split if current is None else pre_tokenizers.Sequence([split, current])
