"""Inference with a saved model and metric computation."""

from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from bertgen.stages._encoding import OUTSIDE, label_maps, trimmed, unreachable_spans
from bertgen.types import (
    Example,
    LabelScore,
    Metrics,
    Prediction,
    Scored,
    Span,
    TaskKind,
    TaskSpec,
)

BATCH_SIZE = 32
MAX_INFERENCE_LENGTH = 512
MULTILABEL_THRESHOLD = 0.5


def _batches[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def decode_bio(
    text: str,
    tags: Sequence[str],
    offsets: Sequence[tuple[int, int]],
    special: Sequence[int],
) -> list[Span]:
    """Merge B-/I- token tags into character spans; an I- without a matching B- starts a span."""
    spans: list[Span] = []
    current: Span | None = None
    for tag, (start, end), is_special in zip(tags, offsets, special, strict=True):
        if is_special:
            continue
        if tag == OUTSIDE:
            current = None
            continue
        prefix, _, label = tag.partition("-")
        if current is not None and prefix == "I" and current.label == label:
            current.end = end
        else:
            current = Span(start=start, end=end, label=label)
            spans.append(current)
    return [trimmed(text, span) for span in spans]


class Predictor:
    """A saved model loaded once and applied to any number of texts."""

    def __init__(self, spec: TaskSpec, model_dir: Path) -> None:
        self.spec = spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        is_span = spec.kind is TaskKind.SPAN
        auto = AutoModelForTokenClassification if is_span else AutoModelForSequenceClassification
        self.model = auto.from_pretrained(model_dir).to(self.device).eval()
        self.id2label = label_maps(spec).id2label
        self.max_length = min(self.model.config.max_position_embeddings, MAX_INFERENCE_LENGTH)

    def __call__(self, texts: Sequence[str]) -> list[Example]:
        """Examples with labels or spans filled in, in the order of `texts`."""
        return [scored.example for scored in self.scored(texts)]

    def scored(self, texts: Sequence[str]) -> list[Scored]:
        """Like calling the predictor, with label or span probabilities attached."""
        results: list[Scored] = []
        for batch in _batches(texts, BATCH_SIZE):
            results += self._batch(batch)
        return results

    def _batch(self, batch: Sequence[str]) -> list[Scored]:
        enc = self.tokenizer(
            list(batch),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping").tolist()
        special = enc.pop("special_tokens_mask").tolist()
        mask = enc["attention_mask"].tolist()
        with torch.inference_mode():
            logits = self.model(**enc.to(self.device)).logits.float().cpu()

        id2label = self.id2label
        results: list[Scored] = []
        for i, text in enumerate(batch):
            match self.spec.kind:
                case TaskKind.SPAN:
                    keep = [j for j, m in enumerate(mask[i]) if m]
                    probs, ids = torch.softmax(logits[i], -1).max(-1)
                    tags = [id2label[k] for k in ids.tolist()]
                    token_offsets = [(offsets[i][j][0], offsets[i][j][1]) for j in keep]
                    spans = decode_bio(
                        text,
                        [tags[j] for j in keep],
                        token_offsets,
                        [special[i][j] for j in keep],
                    )
                    token_probs = [probs[j].item() for j in keep]
                    results.append(
                        Scored(
                            example=Example(text=text, spans=spans),
                            span_scores=[_span_score(s, token_offsets, token_probs) for s in spans],
                        )
                    )
                case TaskKind.MULTILABEL:
                    probs = torch.sigmoid(logits[i]).tolist()
                    names = [id2label[k] for k, p in enumerate(probs) if p > MULTILABEL_THRESHOLD]
                    results.append(
                        Scored(
                            example=Example(text=text, labels=names),
                            scores={id2label[k]: p for k, p in enumerate(probs)},
                        )
                    )
                case TaskKind.BINARY | TaskKind.MULTICLASS:
                    probs = torch.softmax(logits[i], -1).tolist()
                    best = max(range(len(probs)), key=probs.__getitem__)
                    results.append(
                        Scored(
                            example=Example(text=text, labels=[id2label[best]]),
                            scores={id2label[k]: p for k, p in enumerate(probs)},
                        )
                    )
        return results


def _span_score(span: Span, offsets: Sequence[tuple[int, int]], probs: Sequence[float]) -> float:
    inside = [
        p
        for (start, end), p in zip(offsets, probs, strict=True)
        if start < span.end and end > span.start
    ]
    return sum(inside) / len(inside) if inside else 0.0


def predict(spec: TaskSpec, model_dir: Path, texts: Sequence[str]) -> list[Example]:
    """Run the saved model over `texts`; returns Examples with labels or spans filled in."""
    return Predictor(spec, model_dir)(texts)


def _span_keys(example: Example) -> set[tuple[int, int, str]]:
    return {(s.start, s.end, s.label) for s in example.spans}


def _is_correct(kind: TaskKind, truth: Example, predicted: Example) -> bool:
    if kind is TaskKind.SPAN:
        return _span_keys(truth) == _span_keys(predicted)
    return set(truth.labels) == set(predicted.labels)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _span_metrics(
    spec: TaskSpec, test: Sequence[Example], predicted: Sequence[Example]
) -> tuple[float, list[LabelScore]]:
    tp = dict.fromkeys(spec.label_names, 0)
    fp = dict.fromkeys(spec.label_names, 0)
    fn = dict.fromkeys(spec.label_names, 0)
    for truth, pred in zip(test, predicted, strict=True):
        gold, guess = _span_keys(truth), _span_keys(pred)
        for _, _, label in gold & guess:
            tp[label] += 1
        for _, _, label in guess - gold:
            fp[label] += 1
        for _, _, label in gold - guess:
            fn[label] += 1

    per_label: list[LabelScore] = []
    for name in spec.label_names:
        precision, recall, f1 = _prf(tp[name], fp[name], fn[name])
        per_label.append(
            LabelScore(
                label=name,
                precision=precision,
                recall=recall,
                f1=f1,
                support=tp[name] + fn[name],
            )
        )
    micro_f1 = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))[2]
    return micro_f1, per_label


def _indicator(spec: TaskSpec, examples: Sequence[Example]) -> list[list[int]]:
    return [[int(name in ex.labels) for name in spec.label_names] for ex in examples]


def _classification_metrics(
    spec: TaskSpec, test: Sequence[Example], predicted: Sequence[Example]
) -> tuple[float, float, list[LabelScore]]:
    y_true, y_pred = _indicator(spec, test), _indicator(spec, predicted)
    accuracy = float(accuracy_score(y_true, y_pred))
    per_label: list[LabelScore] = []
    observed: list[float] = []
    for name, matrix in zip(
        spec.label_names, multilabel_confusion_matrix(y_true, y_pred), strict=True
    ):
        (_, fp), (fn, tp) = matrix.tolist()
        precision, recall, f1 = _prf(tp, fp, fn)
        per_label.append(
            LabelScore(label=name, precision=precision, recall=recall, f1=f1, support=tp + fn)
        )
        if tp + fp + fn:
            observed.append(f1)
    macro = sum(observed) / len(observed) if observed else 0.0
    return macro, accuracy, per_label


def evaluate(spec: TaskSpec, model_dir: Path, test: Sequence[Example]) -> Metrics:
    """Score the saved model on `test`; `errors` lists every mispredicted example."""
    unreachable = 0
    if spec.kind is TaskKind.SPAN:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        unreachable = unreachable_spans(tokenizer, test, MAX_INFERENCE_LENGTH)
    predicted = predict(spec, model_dir, [ex.text for ex in test])
    errors = [
        Prediction(example=truth, predicted=pred)
        for truth, pred in zip(test, predicted, strict=True)
        if not _is_correct(spec.kind, truth, pred)
    ]
    n_test = len(test)
    if spec.kind is TaskKind.SPAN:
        primary, per_label = _span_metrics(spec, test, predicted)
        accuracy = (n_test - len(errors)) / n_test if n_test else 0.0
    else:
        primary, accuracy, per_label = _classification_metrics(spec, test, predicted)
    return Metrics(
        primary=primary,
        accuracy=accuracy,
        per_label=per_label,
        errors=errors,
        n_test=n_test,
        unreachable_spans=unreachable,
    )
