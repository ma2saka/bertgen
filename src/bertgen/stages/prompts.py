"""Prompt builders for the LLM-driven stages."""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from bertgen.types import NONE_KEY, DataPlan, Example, RoundResult, TaskKind, TaskSpec

EXAMPLES_MARKER = "EXAMPLES"
MAX_SEEDS = 40

_KIND_GUIDE = {
    TaskKind.BINARY: "binary classification: exactly one of two labels per text",
    TaskKind.MULTICLASS: "single-label classification: exactly one label per text",
    TaskKind.MULTILABEL: "multi-label classification: zero or more labels per text",
    TaskKind.SPAN: "span extraction: zero or more labeled substrings per text",
}

SPEC_SYSTEM = """\
You design the specification of a small text classifier or extractor from a natural-language rule.
The specification will drive synthetic data generation and human-grade evaluation, so it must be
precise and self-contained.

Choose `kind`:
- span: the rule asks to extract, locate, highlight or mark substrings. Labels are entity types.
  Spans never overlap or nest; criteria must not require overlapping spans.
- multilabel: several labels can be true for the same text at once.
- binary: the rule is a yes/no decision. Provide exactly two labels and make the negative class
  explicit with a real name such as "none" or "other" (positive label first).
- multiclass: mutually exclusive categories. Add an "other" label unless the categories are
  exhaustive.

Requirements:
- `rule` echoes the input rule.
- `language` is the BCP-47 code of the texts the classifier will read. Infer it from the rule
  (e.g. "Japanese harassment detection" -> "ja"); if the rule does not say, use the language the
  rule is written in.
- Label names are ASCII snake_case, short and stable. Each description says what belongs in the
  class and what does not.
- `input_description` says what one input looks like (length, register, source).
- `decision_criteria` is precise enough for two independent annotators to agree: concrete tests,
  thresholds, boundary cases and how ambiguity is resolved. For span tasks, state exactly which
  characters belong to a span and which do not.
- Decision criteria must follow from the rule. When the rule leaves a case open, name it as out of
  scope in one sentence instead of inventing a tie-break.
"""


def spec_user(rule: str) -> str:
    return f"Rule:\n{rule}\n\nProduce the task specification."


def describe_spec(spec: TaskSpec) -> str:
    labels = "\n".join(f"- {label.name}: {label.description}" for label in spec.labels)
    return (
        f"Rule: {spec.rule}\n"
        f"Task: {_KIND_GUIDE[spec.kind]}\n"
        f"Text language (BCP-47): {spec.language}\n"
        f"Input: {spec.input_description}\n"
        f"Labels:\n{labels}\n"
        f"Decision criteria:\n{spec.decision_criteria}"
    )


def research_question(spec: TaskSpec) -> str:
    span_hint = (
        "Also give the formal definition needed to apply the rule exactly (counting units, "
        "boundary conventions, worked examples).\n"
        if spec.kind is TaskKind.SPAN
        else ""
    )
    return (
        "I am building training data for a text classifier.\n"
        f"{describe_spec(spec)}\n\n"
        "Research how this phenomenon really appears in the language and domain above: typical "
        "real-world phrasing and registers, slang and domain vocabulary, common false-positive "
        "look-alikes, and borderline cases annotators disagree on.\n"
        f"{span_hint}"
        f"Answer in plain text with two parts. First, notes of at most 400 words with concrete "
        f"examples. Then a line containing only {EXAMPLES_MARKER}, followed by up to "
        f"{MAX_SEEDS} real texts or fragments found on the web that illustrate the rule, one "
        "per line in the form `<verbatim text> | <source URL>`. Quote verbatim, in the language "
        "above, each at most 200 characters; prefer public-domain or clearly quotable material "
        "and include look-alikes that do not qualify. Never invent examples."
    )


PLAN_SYSTEM = """\
You plan a synthetic dataset for a text classifier so that it generalizes to real data.

Produce:
- label_weights: relative share of examples targeting each label. Keep classes roughly balanced,
  with a bit more weight on classes that are easy to confuse. For multilabel and span tasks you
  may add the key "__none__" for texts with no label/span (typically 0.1 to 0.25).
- axes: 3 to 6 dimensions along which inputs must vary (topic/domain, register, length, source
  type, surface form, demographic or situational context, difficulty, ...). Each axis has 3 to 8
  concrete values. Axes must be orthogonal to the label: every value should be combinable with
  every label. Values describe a situation or property in a few words; they must not contain
  example wording, quoted phrases or literal markers that could be copied into a text.
- hard_cases: 6 to 12 near-miss or confusable situations, each one sentence, stating what makes it
  tricky and which label it should get. Include lexical traps (trigger words that do not qualify)
  and quiet positives (qualifying items with no trigger words).
- style_notes: how texts should look (length, punctuation, typos, formatting, voice) so that they
  resemble real inputs rather than textbook samples.
- seeds: when the research notes list real texts, the verbatim texts worth keeping as raw material
  for writers, without source URLs, each a self-contained text or fragment of at most 200
  characters. Drop duplicates, dead links, commentary and anything not quoted verbatim. Never
  write seeds yourself; return an empty list when the notes contain no real texts.
"""


def plan_user(spec: TaskSpec, research_notes: str) -> str:
    notes = f"\n\nResearch notes:\n{research_notes}" if research_notes else ""
    return f"{describe_spec(spec)}{notes}\n\nPlan the dataset."


def generate_system(spec: TaskSpec) -> str:
    return f"""\
You write training data for a text classifier. Quality decides whether the classifier works.

{describe_spec(spec)}

Write every text in the language with BCP-47 code '{spec.language}', with natural native phrasing.

Rules for every example:
- Realistic: it must look like something a real person or system wrote, not a textbook sample.
- Unambiguous: an independent annotator reading only the decision criteria must reach the label
  you give. Drop or rewrite anything borderline.
- Varied: vary length, register, structure, vocabulary and sentence openings. Never reuse a
  template with swapped words. Do not repeat earlier examples.
- No leakage: never mention the label names, the task or the classification in the text.
- No numbering, quotes around the text, or commentary inside `text`.
{_output_format(spec)}"""


def _output_format(spec: TaskSpec) -> str:
    if spec.kind is TaskKind.SPAN:
        return (
            "Output format: items with `text` and `spans`. Each span has `text` (an exact, "
            "character-for-character copy of the substring as it appears in the example) and "
            "`label`. List a span once per occurrence to annotate, in reading order. Spans must "
            "not overlap. A text without spans has an empty list."
        )
    if spec.kind is TaskKind.MULTILABEL:
        return (
            "Output format: items with `text` and `labels` (every label that applies; empty when "
            "none applies)."
        )
    return "Output format: items with `text` and `labels` holding exactly one label."


@dataclass(frozen=True)
class Cell:
    """What one requested example should be: a target label, whether it should instead be a
    look-alike of that label, and a value per plan axis."""

    label: str
    axes: dict[str, str]
    lookalike: bool = False


def generate_user(
    spec: TaskSpec,
    plan: DataPlan,
    *,
    cells: Sequence[Cell],
    hard_cases: Sequence[str],
    focus: Sequence[str],
    avoid: Sequence[str],
    seeds: Sequence[str] = (),
) -> str:
    lines = "\n".join(f"{i}. {_cell_line(spec, cell)}" for i, cell in enumerate(cells, 1))
    parts = [
        f"Write {len(cells)} examples, one for each numbered line below. Each line gives the "
        "example's target and situation. The situation describes circumstances only; do not "
        f"reuse its wording in the text.\n{lines}"
    ]
    if plan.style_notes:
        parts.append(f"Style: {plan.style_notes}")
    if plan.research_notes:
        parts.append(f"Background knowledge:\n{plan.research_notes}")
    if hard_cases:
        cases = "\n".join(f"- {case}" for case in hard_cases)
        parts.append(
            f"Build several examples on these hard cases, keeping each line's target:\n{cases}"
        )
    if focus:
        items = "\n".join(f"- {item}" for item in focus)
        parts.append(f"Weak spots of the current classifier; cover them generously:\n{items}")
    if seeds:
        items = "\n".join(f"- {seed}" for seed in seeds)
        parts.append(
            "Real-world material found on the web. Use it as raw material for some examples: "
            "embed a fragment in a longer text, adapt its phrasing, or deliberately break it "
            "into a look-alike where the line's target calls for one. The label always follows "
            "the decision criteria, not the material. Copy one unchanged as a whole example "
            f"only occasionally.\n{items}"
        )
    if avoid:
        items = "\n".join(f"- {snippet}" for snippet in avoid)
        parts.append(
            f"Existing examples (for diversity; do not repeat or paraphrase them):\n{items}"
        )
    return "\n\n".join(parts)


def _cell_line(spec: TaskSpec, cell: Cell) -> str:
    situation = "".join(f" | {name}: {value}" for name, value in cell.axes.items())
    return f"target: {_target(spec, cell)}{situation}"


def _target(spec: TaskSpec, cell: Cell) -> str:
    label = cell.label
    is_span = spec.kind is TaskKind.SPAN
    if label == NONE_KEY:
        if is_span:
            return "no span at all; plausible look-alikes stay unmarked"
        return "no label applies; a plausible near-miss"
    if cell.lookalike:
        if is_span:
            return f"resembles '{label}' but contains no qualifying '{label}' span"
        if spec.kind is TaskKind.MULTILABEL:
            return f"a close look-alike of '{label}' where '{label}' does not apply"
        return f"a close look-alike of '{label}' whose correct label differs"
    if is_span:
        return f"contains one or more '{label}' spans"
    if spec.kind is TaskKind.MULTILABEL:
        return f"'{label}' among its labels, plus any other label that also applies"
    return f"label '{label}'"


def verify_system(spec: TaskSpec) -> str:
    if spec.kind is TaskKind.SPAN:
        answer = (
            "For each text return `spans`: every substring that qualifies, copied exactly from "
            "the text, with its label; list a span once per occurrence. Return an empty list "
            "when nothing qualifies."
        )
    elif spec.kind is TaskKind.MULTILABEL:
        answer = "For each text return `labels`: all labels that apply (empty if none)."
    else:
        answer = "For each text return `labels` holding exactly one label."
    return f"""\
You are a careful, independent annotator. Apply the specification strictly; do not guess what the
text's author intended.

{describe_spec(spec)}

{answer} Return one item per input with the same `index`."""


def verify_user(texts: Sequence[str]) -> str:
    payload = [{"index": i, "text": text} for i, text in enumerate(texts)]
    return "Annotate these texts:\n" + json.dumps(payload, ensure_ascii=False, indent=1)


JUDGE_SYSTEM = """\
You supervise the training of a text classifier and decide what happens after each round.

Decisions:
- accept: quality is good enough for the stated purpose, or remaining errors come from ambiguous
  or mislabeled gold data rather than from the model.
- more_data: errors concentrate on patterns, labels or situations the training data covers too
  thinly. Put those weak spots in `focus`, one short sentence each, at most 8. Describe each as
  an abstract pattern; never quote, paraphrase or reuse wording from the texts shown to you.
  `hparams` may also change the training settings for the next round.
- tune: the data looks adequate but optimization is off (underfit: raise epochs or learning rate;
  overfit or unstable: lower learning rate, weight decay, fewer epochs; truncation: raise
  max_length). Put the changes in `hparams`; leave other fields null.
- stop: more rounds will not help (score plateaued across rounds, or the task is not learnable
  from the available data).

Scores come from a held-out test split; the error sample comes from the validation split.
Read the errors before choosing: distinguish model mistakes from gold labels that violate the
decision criteria. For span tasks, `unreachable_spans` counts gold spans whose boundary falls
inside a token; the model cannot reproduce them, so they cap the score and more data does not
help with them. Each earlier round lists the verdict taken after it; prefer a different action
when the previous one did not help. Give a short `reason`.
"""


def describe_example(example: Example) -> str:
    if example.spans:
        return "; ".join(
            f"{span.label}={example.text[span.start : span.end]!r}" for span in example.spans
        )
    return ", ".join(example.labels) or "(none)"


def _describe_round(result: RoundResult) -> str:
    metrics = result.metrics
    per_label = ", ".join(
        f"{s.label}: P={s.precision:.2f} R={s.recall:.2f} F1={s.f1:.2f} n={s.support}"
        for s in metrics.per_label
    )
    lines = [
        f"Round {result.round}: primary={metrics.primary:.4f} accuracy={metrics.accuracy:.4f} "
        f"n_test={metrics.n_test} unreachable_spans={metrics.unreachable_spans}"
        + ("" if result.warm_from is None else f" started_from_round={result.warm_from}"),
        f"  config: {result.config.model_dump_json(exclude={'base_model'})}",
        f"  per label: {per_label}",
    ]
    if (verdict := result.verdict) is not None:
        lines.append(f"  verdict: {verdict.decision} ({verdict.reason})")
        if verdict.focus:
            lines.append(f"  focus: {'; '.join(verdict.focus)}")
        if patch := verdict.hparams.model_dump(exclude_none=True):
            lines.append(f"  hparams: {json.dumps(patch)}")
    return "\n".join(lines)


_WARM_START = (
    "Training: every round after the first starts from the previous round's model and trains on"
    " all accumulated data; `epochs` are extra passes on top of that model, not a total. Prefer"
    " few epochs (1-3) when the training set is large. Fewer epochs or a lower learning rate"
    " only limit further overfitting; they cannot undo overfitting already in that model."
)
_COLD_START = (
    "Training: every round starts from the base model and trains on all accumulated data;"
    " `epochs` is the total number of passes."
)


def judge_user(
    spec: TaskSpec,
    plan: DataPlan,
    history: Sequence[RoundResult],
    counts: dict[str, dict[str, int]],
    errors: Sequence[tuple[str, str, str]],
    target: float,
    warm_start: bool,
) -> str:
    rounds = [_describe_round(result) for result in history]
    count_lines = "\n".join(f"- {split}: {json.dumps(c)}" for split, c in counts.items())
    error_lines = "\n".join(
        f"- text: {text[:300]!r}\n  gold: {gold}\n  predicted: {predicted}"
        for text, gold, predicted in errors
    )
    axes = "; ".join(f"{a.name} ({len(a.values)})" for a in plan.axes)
    return (
        f"{describe_spec(spec)}\n\n"
        f"Data plan: axes = {axes}; hard cases = {len(plan.hard_cases)}\n"
        f"Target primary score: {target}\n"
        f"{_WARM_START if warm_start else _COLD_START}\n\n"
        f"History (oldest first):\n" + "\n".join(rounds) + "\n\n"
        f"Examples per split and label:\n{count_lines}\n\n"
        f"Sample of validation errors from the latest round:\n{error_lines or '(none)'}\n\n"
        "Decide the next step."
    )
