"""Domain types shared by every pipeline stage."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, Field

NONE_KEY = "__none__"  # pseudo-label for MULTILABEL/SPAN examples without any label or span


class TaskKind(StrEnum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    SPAN = "span"


class Split(StrEnum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Stage(StrEnum):
    SPEC = "spec"
    PLAN = "plan"
    GENERATE = "generate"
    TRAIN = "train"
    EVALUATE = "evaluate"
    JUDGE = "judge"


class OracleOutcome(StrEnum):
    """How an oracle's label relates to the label the generating LLM wrote."""

    AGREED = "agreed"
    DISAGREED = "disagreed"
    ABSTAINED = "abstained"


class Decision(StrEnum):
    ACCEPT = "accept"
    MORE_DATA = "more_data"
    TUNE = "tune"
    STOP = "stop"


class LabelDef(BaseModel):
    """One output class, or one entity type for SPAN tasks."""

    name: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]
    description: str


class TaskSpec(BaseModel):
    """What the classifier outputs, derived from the user's rule."""

    rule: str
    kind: TaskKind
    language: Annotated[str, Field(description="BCP-47 code of the input text, e.g. 'ja', 'en'")]
    labels: Annotated[list[LabelDef], Field(min_length=1)]
    input_description: str
    decision_criteria: str

    @property
    def label_names(self) -> list[str]:
        return [label.name for label in self.labels]


class Axis(BaseModel):
    """A dimension along which generated inputs must vary."""

    name: str
    values: Annotated[list[str], Field(min_length=1)]


class DataPlan(BaseModel):
    """How synthetic data is produced for a TaskSpec."""

    label_weights: dict[str, float]
    axes: list[Axis]
    hard_cases: list[str]
    style_notes: str
    research_notes: str = ""
    seeds: list[str] = []


class Span(BaseModel):
    start: int
    end: int
    label: str


class Example(BaseModel):
    """One labeled input.

    `labels` holds class names for BINARY/MULTICLASS (exactly one) and MULTILABEL (zero or more).
    `spans` holds character offsets for SPAN tasks.
    """

    text: str
    labels: list[str] = []
    spans: list[Span] = []


@dataclass(frozen=True)
class Draft:
    """A synthesized example before storage.

    `example` carries the label to store and `written` the label the generating LLM gave;
    `outcome` says how an oracle judged `written`, None when no oracle was consulted.
    """

    example: Example
    written: Example
    outcome: OracleOutcome | None = None

    @classmethod
    def plain(cls, example: Example) -> Self:
        return cls(example=example, written=example)


class OracleAgreement(BaseModel):
    """Counts of oracle outcomes over stored examples."""

    agreed: int = 0
    disagreed: int = 0
    abstained: int = 0

    @classmethod
    def tally(cls, outcomes: Iterable[OracleOutcome]) -> Self:
        counts = Counter(outcomes)
        return cls(
            agreed=counts[OracleOutcome.AGREED],
            disagreed=counts[OracleOutcome.DISAGREED],
            abstained=counts[OracleOutcome.ABSTAINED],
        )

    def __add__(self, other: Self) -> Self:
        return type(self)(
            agreed=self.agreed + other.agreed,
            disagreed=self.disagreed + other.disagreed,
            abstained=self.abstained + other.abstained,
        )

    @property
    def labeled(self) -> int:
        return self.agreed + self.disagreed

    @property
    def rate(self) -> float | None:
        """Share of oracle-labeled examples whose LLM label matched; None if none were labeled."""
        return self.agreed / self.labeled if self.labeled else None


class TrainConfig(BaseModel):
    base_model: str
    epochs: float = 3.0
    learning_rate: float = 5e-5
    batch_size: int = 16
    max_length: int = 256
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42


class HParamPatch(BaseModel):
    """Partial TrainConfig override proposed by the judge."""

    epochs: Annotated[float, Field(gt=0)] | None = None
    learning_rate: Annotated[float, Field(gt=0, lt=1)] | None = None
    batch_size: Annotated[int, Field(ge=1)] | None = None
    max_length: Annotated[int, Field(ge=8)] | None = None
    warmup_ratio: Annotated[float, Field(ge=0, lt=1)] | None = None
    weight_decay: Annotated[float, Field(ge=0)] | None = None

    def apply(self, config: TrainConfig) -> TrainConfig:
        return TrainConfig.model_validate(
            {**config.model_dump(), **self.model_dump(exclude_none=True)}
        )


class LabelScore(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class Prediction(BaseModel):
    example: Example
    predicted: Example


class Metrics(BaseModel):
    """Evaluation result on one split.

    `primary` is macro-F1 over labels seen in gold or predictions for classification kinds, and
    entity-level micro-F1 for SPAN. `unreachable_spans` counts gold spans whose start or end falls
    inside a token, which token-level BIO tagging cannot reproduce exactly.
    """

    primary: float
    accuracy: float
    per_label: list[LabelScore]
    errors: list[Prediction]
    n_test: int
    unreachable_spans: int = 0


class Verdict(BaseModel):
    decision: Decision
    reason: str
    focus: list[str] = []
    hparams: HParamPatch = HParamPatch()


class RoundInit(BaseModel):
    """The round whose model a round started from; None means the base model."""

    warm_from: int | None = None


class RoundResult(BaseModel):
    """A trained and evaluated round; `verdict` is None until the judge has decided."""

    round: int
    config: TrainConfig
    metrics: Metrics
    valid_metrics: Metrics | None = None
    verdict: Verdict | None = None
    warm_from: int | None = None

    @property
    def rank(self) -> tuple[float, float]:
        """Sort key for choosing the best round: test primary, then validation primary."""
        valid = self.valid_metrics.primary if self.valid_metrics else 0.0
        return (self.metrics.primary, valid)
