# bertgen

Turn a natural-language rule into a fine-tuned ModernBERT classifier.

```sh
bgen --name harassment-ja --rule "日本語の文章がハラスメントに該当するか判定する"
```

bertgen asks LLMs to define the task, plan and synthesize a labeled dataset, cross-check it,
fine-tune a ModernBERT model, evaluate it on a held-out synthetic test set, and decide whether
another round is worth it. The result is a standard `save_pretrained` directory that
`transformers.pipeline` loads directly.

## How it works

```
rule ─► 1 spec ─► 2 plan ─► 3 generate ─► 4 train ─► 5 evaluate ─► 6 judge ─┬─► accept / stop
                                ▲                                            │
                                └──────── more_data (focus) / tune (hparams) ┘
```

1. Spec: the LLM turns the rule into a `TaskSpec`: task kind (`binary`, `multiclass`,
   `multilabel`, `span`), labels with descriptions, input language and decision criteria.
2. Plan: label mix, axes of variation (register, length, domain, ...), hard near-miss cases.
   With `--web`, a planner that can search the web (`anthropic:` or `claude-code:`) first
   collects background notes and up to 40 real example snippets ("seeds", stored in
   `plan.json`); train/valid generation prompts show a few seeds per call to adapt or turn into
   look-alikes. The test writer never sees them.
3. Generate: batches of examples are written concurrently and deduplicated. Every example in
   a request gets its own target label (or a look-alike of it) and its own point on the plan's
   axes. Test data comes from the eval LLM and train/valid data from the generator LLM; each side
   is re-annotated blind by the other, and only examples where both agree are kept. Test texts
   never appear in a generation prompt. For span tasks the LLM quotes substrings and the offsets
   are computed in code. With `--oracle`, a deterministic labeler replaces the LLM labels (see
   [Rule oracles](#rule-oracles)).
4. Train: sequence classification (or BIO token classification for spans) with the
   Hugging Face `Trainer`. For spans, if more than 1% of the training spans start or end inside
   a token, the tokenizer is switched to one pre-token per character; the change is saved in
   `tokenizer.json`, so `pipeline` reproduces it. Every round trains on all data gathered so
   far. From the second round on it starts from the previous round's model and tokenizer (warm
   start), so `epochs` are extra passes on top of that model; `--no-warm-start` retrains from
   the base model each round.
5. Evaluate: macro-F1 for classification, entity-level micro-F1 for spans, plus per-label
   scores and the misclassified examples, on both the test and the validation split. For spans,
   gold spans whose boundaries fall inside a token are counted as `unreachable_spans`: token
   tagging cannot reproduce them, so they cap the score.
6. Judge: stops when the target score is reached; otherwise the LLM reads the test metrics,
   earlier verdicts and a sample of validation errors, and chooses `more_data` (weak spots to
   generate, described as abstract patterns, optionally with a hyperparameter patch for the
   next round), `tune` (a validated hyperparameter patch alone), `accept` or `stop`. The judge
   is told whether rounds warm start, so it can ask for fewer epochs on large data. A round that would repeat the previous one unchanged is not trained.

Every stage writes its result to the run directory, so an interrupted run resumes where it
left off.

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is recommended for
training.

```sh
git clone https://github.com/ma2saka/bertgen.git && cd bertgen
uv sync                        # add --extra ja for the Japanese haiku oracle (fugashi, unidic-lite)
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY / DEEPSEEK_API_KEY, depending on --llm
```

## Usage

Binary classification in Japanese (the base model is chosen from the spec's language:
`llm-jp/llm-jp-modernbert-base` for `base`, `sbintuitions/modernbert-ja-310m` for `large`):

```sh
uv run bgen --name harassment-ja \
  --rule "日本語の職場チャットの発言がハラスメントに該当するかを判定する" \
  --examples 600 --test-examples 200 --max-rounds 4 --target 0.9
```

Span extraction, with web research through the `claude` CLI and labels from the bundled
5-7-5 oracle (`uv sync --extra ja`):

```sh
uv run bgen --name haiku-575 \
  --rule "日本語の文章から5・7・5の音数になっている部分文字列を抽出する" \
  --planner-llm claude-code:sonnet --web --oracle bertgen.oracles.haiku:HaikuOracle
```

Review the task definition before any data is generated, then continue:

```sh
uv run bgen --name harassment-ja --rule "..." --stop-after spec   # prints runs/harassment-ja/spec.json
$EDITOR runs/harassment-ja/spec.json
uv run bgen --name harassment-ja
```

Use the result, inspect progress, resume or extend a run:

```sh
uv run bgen predict --name harassment-ja "この資料、今日中にお願いできますか" "使えないやつだな"
cat texts.txt | uv run bgen predict --name harassment-ja   # one text per line on stdin
uv run bgen status --name harassment-ja
uv run bgen --name harassment-ja --max-rounds 6    # resume with only --name; flags override
```

`predict` prints one JSON object per text: `{"text": ..., "labels": [...]}`, or
`{"text": ..., "spans": [{"text", "label", "start", "end"}]}` for span tasks.

Main options (see `bgen --help`):

| option | default | meaning |
|---|---|---|
| `--model` | `base` | `base`, `large`, a preset (`en-base`, `en-large`, `llm-jp-base`, `ja-70m`, `ja-130m`, `ja-310m`, `mm-small`, `mm-base`) or any Hugging Face model id |
| `--llm` | `anthropic:claude-sonnet-5-5` | generator LLM |
| `--eval-llm` | `--llm` | writes the test set and verifies training data |
| `--planner-llm` | `--llm` | spec, plan and judge |
| `--examples` / `--test-examples` | 400 / 200 | initial train+valid and test sizes |
| `--examples-per-round` | 200 | added on a `more_data` verdict |
| `--max-rounds` / `--target` | 3 / 0.9 | loop limits |
| `--epochs --lr --batch-size --max-length --seed` | | first-round training settings; later rounds follow the judge's `tune` patches |
| `--no-verify` | | skip cross-verification |
| `--no-warm-start` | | train every round from the base model instead of the previous round's model |
| `--web` | | web research before planning (`anthropic:` or `claude-code:` planner) |
| `--oracle` | | `package.module:attr` of a rule oracle that relabels generated data; fixed once data exists |
| `--corpus` / `--corpus-max-chars` | - / 300 | shuffled `.txt` or `.jsonl` (`.gz`) file of real text for the oracle to label instead of generating data; fixed once data exists |
| `--stop-after` | | `spec`, `plan` or `generate`: stop there so the artifact can be reviewed |
| `--runs-dir` | `runs` | where runs are stored |

With the default `--eval-llm`, the same LLM writes and verifies every example, so
verification only catches its own inconsistencies (bertgen logs a warning). Use a different
provider for `--eval-llm` to get an independent cross-check.

## LLM specs

| spec | provider | credentials |
|---|---|---|
| `anthropic:<model>` | Anthropic API; supports `--web` | `ANTHROPIC_API_KEY` |
| `openai:<model>` | OpenAI API | `OPENAI_API_KEY` |
| `deepseek:<model>` | DeepSeek API | `DEEPSEEK_API_KEY` |
| `ollama:<model>[@<host>]` | a local [Ollama](https://ollama.com) server, e.g. `ollama:gemma4:12b`; the host defaults to `$OLLAMA_HOST`, then `127.0.0.1:11434` | none |
| `local:<model>@<base_url>` | any OpenAI-compatible server, e.g. `local:qwen3@http://127.0.0.1:8080/v1` for llama-server | none |
| `claude-code:<model>` | the `claude` CLI (`claude -p --safe-mode`, so your CLAUDE.md, skills, hooks and MCP servers are not loaded); no tools, or only WebSearch/WebFetch for `--web` | CLI login (`ANTHROPIC_API_KEY` is removed from its environment) |
| `codex[:<model>]` | the `codex` CLI (`codex exec`) | CLI login |

Structured output uses tool calls (Anthropic), `json_schema` / `json_object` response formats
(OpenAI-compatible) or JSON extracted from text, and malformed replies are retried with the
validation error. An OpenAI-compatible server that rejects a response format with an error
mentioning `response_format`, `json_schema` or `json_object` is retried with the next looser
format; any other client error fails the call immediately.

Generation and verification can run entirely on local models, for example with two different
Ollama models so the cross-check stays independent:

```sh
uv run bgen --name harassment-ja --rule "日本語の職場での発言がハラスメントに該当するか判定する" \
  --llm ollama:qwen3.6:35b --eval-llm ollama:gemma4:12b --planner-llm ollama:qwen3.6:35b
```

## Rule oracles

Some rules can be checked by code better than by an LLM. An oracle is any callable
`(TaskSpec) -> Oracle` whose `label(text)` returns the gold `Example`, or `None` to abstain:

```python
class Oracle(Protocol):
    def label(self, text: str) -> Example | None: ...
```

With `--oracle package.module:attr`, every generated example (train, valid and test) is
relabeled by the oracle and skips LLM verification; examples it abstains on go through the
usual cross-check. The rate at which the writing LLM's labels matched the oracle is logged
and stored per split in `run.json` (`oracle_agreement`), in each `report.md` and in
`bgen status`; a low rate means the LLM misunderstands the rule (or the oracle has blind
spots). Relabeling can shift the label balance away from the plan. Every relabeled text is
logged, and `data.db` keeps the writer's labels next to the stored ones for auditing:

```sql
SELECT split, text, labels, spans, llm_labels, llm_spans
FROM examples WHERE oracle = 'disagreed';
```

`bertgen.oracles.haiku:HaikuOracle` finds 5-7-5 mora runs in Japanese with fugashi and
unidic-lite (`uv sync --extra ja`) for a span task with one label. It abstains on texts with
words that have no known reading. Readings come from the dictionary and are trusted: a
misread word (兵 as ヘー in 夏草や兵どもが夢の跡) turns a real haiku into a confident "no span".
Cuts fall on unidic short-unit boundaries and each part starts on a unit that is not a particle,
auxiliary verb, suffix or symbol, so a haiku whose count needs a cut inside one unit
(さんじのお/やつは…) is not found either. Spans never overlap; the leftmost match wins.

### Labeling real text

With an oracle, labels need no LLM, so `--corpus PATH` takes examples from real text instead of
generating them. The file holds one document per line (`.txt`) or one `{"text": ...}` object per
line (`.jsonl`), optionally gzipped, and should be shuffled: it is read in order. Documents are
split into passages of at most `--corpus-max-chars` characters, the oracle labels them, and
passages are kept until each label reaches its share of the plan's `label_weights`; abstentions
are dropped. For span and multi-label tasks a plan without a `__none__` weight gets a 25% share
of passages with no span or label, so the data always has negatives. A test set with an empty
label share is an error. Sampling stops after 1,000,000 passages in a row add nothing. A hash of
each document's text sends all of its passages to either the test pool (about 20%) or the
train/valid pool, so the two never share a document. Spec, plan, training and the judge work as
before; the judge's `focus` is ignored. Rows are stored with source `corpus:<file name>` and no
oracle outcome. [examples/haiku575](examples/haiku575) builds a Japanese corpus from Aozora
Bunko and Wikipedia with DuckDB for the 5-7-5 oracle.

## Run directory

```
runs/<name>/
  run.json             resolved options, current round, oracle agreement
  spec.json            task specification
  plan.json            data plan
  data.db              SQLite: examples with split, round, source LLM, oracle outcome and
                       the writer LLM's own labels
  rounds/<n>/
    train_config.json
    model/             save_pretrained output (model + tokenizer, id2label set)
    metrics.json       test split
    valid_metrics.json validation split; its errors are what the judge reads
    verdict.json
    report.md          metrics table, verdict and sample errors
  model -> rounds/<best>/model
```

Delete a file to redo that stage, for example `rounds/2/verdict.json` to ask the judge again.

## Loading the model

```python
from transformers import pipeline

classify = pipeline("text-classification", model="runs/harassment-ja/model")
classify("使えないやつだな")

extract = pipeline(
    "token-classification", model="runs/haiku-575/model", aggregation_strategy="simple"
)
extract("古池や蛙飛び込む水の音、と彼は言った")
```

Multi-label models are saved with `problem_type="multi_label_classification"`; pass
`top_k=None, function_to_apply="sigmoid"` to the pipeline to get every label's score.

## Development

```sh
uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest
```

Tests use scripted fake LLMs and a tiny randomly initialized ModernBERT, so they need no
network or API keys.

## License

MIT
