"""Fine-tune a ModernBERT-family encoder on labeled examples."""

import logging
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from bertgen.stages._encoding import (
    Features,
    LabelMaps,
    encode,
    label_maps,
    split_characters,
    unreachable_spans,
)
from bertgen.types import Example, TaskKind, TaskSpec, TrainConfig

CHECKPOINT_DIRNAME = ".checkpoints"
MAX_UNREACHABLE_RATIO = 0.01
_SPLIT_PROBE = "ab 12 あい"

log = logging.getLogger(__name__)


class _FeatureDataset(Dataset[Features]):
    def __init__(self, features: list[Features]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Features:
        return self.features[index]


def _load_model(spec: TaskSpec, source: str, maps: LabelMaps) -> PreTrainedModel:
    labels = {
        "num_labels": len(maps.id2label),
        "id2label": maps.id2label,
        "label2id": maps.label2id,
    }
    match spec.kind:
        case TaskKind.SPAN:
            return AutoModelForTokenClassification.from_pretrained(source, **labels)
        case TaskKind.MULTILABEL:
            return AutoModelForSequenceClassification.from_pretrained(
                source, problem_type="multi_label_classification", **labels
            )
        case TaskKind.BINARY | TaskKind.MULTICLASS:
            return AutoModelForSequenceClassification.from_pretrained(source, **labels)


def _splits_characters(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Whether the fast tokenizer already makes every character its own pre-token."""
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return False
    pre = tokenizer.backend_tokenizer.pre_tokenizer
    if pre is None:
        return False
    pieces: list[tuple[str, tuple[int, int]]] = pre.pre_tokenize_str(_SPLIT_PROBE)
    return all(end - start <= 1 for _, (start, end) in pieces)


def _align_to_spans(
    tokenizer: PreTrainedTokenizerBase, examples: list[Example], max_length: int
) -> None:
    if _splits_characters(tokenizer):
        return
    total = sum(len(ex.spans) for ex in examples)
    unreachable = unreachable_spans(tokenizer, examples, max_length)
    if total and unreachable / total > MAX_UNREACHABLE_RATIO:
        log.info(
            "%d/%d training spans cut through tokens; tokenizing per character", unreachable, total
        )
        split_characters(tokenizer)


def train(
    spec: TaskSpec,
    train_set: list[Example],
    valid_set: list[Example],
    config: TrainConfig,
    out_dir: Path,
    init: Path | None = None,
) -> Path:
    """Fine-tune `config.base_model`, or continue from the model and tokenizer saved in `init`;
    writes model + tokenizer to `out_dir` with save_pretrained."""
    set_seed(config.seed)
    maps = label_maps(spec)
    source = str(init) if init is not None else config.base_model
    tokenizer = AutoTokenizer.from_pretrained(source)
    if spec.kind is TaskKind.SPAN:
        _align_to_spans(tokenizer, train_set, config.max_length)
    model = _load_model(spec, source, maps)

    def dataset(examples: list[Example]) -> _FeatureDataset:
        return _FeatureDataset(
            [encode(spec, tokenizer, ex, maps, config.max_length) for ex in examples]
        )

    if spec.kind is TaskKind.SPAN:
        collator = DataCollatorForTokenClassification(tokenizer)
    else:
        collator = DataCollatorWithPadding(tokenizer)

    checkpoint_dir = out_dir / CHECKPOINT_DIRNAME
    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_steps=config.warmup_ratio,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        disable_tqdm=True,
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset(train_set),
        eval_dataset=dataset(valid_set),
        processing_class=tokenizer,
    )
    try:
        trainer.train()
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(out_dir)
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return out_dir
