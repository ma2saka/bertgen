# 5-7-5 extraction from real Japanese text

`bertgen.oracles.haiku:HaikuOracle` labels 5-7-5 mora runs without an LLM, so the training and
test data can be real paragraphs labeled by the oracle (`--corpus`) instead of LLM-written text.
Positives are then the same kind in train and test: 5-7-5 runs that occur by accident in prose.

## Build the corpus

Aozora Bunko (novels and essays) and one shard of Japanese Wikipedia, split into paragraphs of
at least 20 characters, deduplicated on their kanji and kana (so edition variants that differ
only in punctuation or digits keep one row) and shuffled with a fixed hash:

```sh
hf download globis-university/aozorabunko-clean aozorabunko-dedupe-clean.jsonl.gz \
  --repo-type dataset --local-dir data/aozora
hf download wikimedia/wikipedia 20231101.ja/train-00005-of-00015.parquet \
  --repo-type dataset --local-dir data/wikipedia

mkdir -p runs/corpus
duckdb -cmd "SET VARIABLE aozora = 'data/aozora/aozorabunko-dedupe-clean.jsonl.gz';
             SET VARIABLE wikipedia = 'data/wikipedia/20231101.ja/train-00005-of-00015.parquet'" \
  -f examples/haiku575/prepare_corpus.sql
```

This writes `runs/corpus/ja575.jsonl` (about 3.6 million paragraphs, 1.1 GB) with
`{"text": ..., "source": "aozora" | "wikipedia"}` per line. bertgen reads the file in order, so
it must stay shuffled.

## Run

```sh
uv sync --extra ja
uv run bgen --name haiku575 \
  --rule "日本語の文章の中から、5・7・5音（モーラ）のリズムになっている部分文字列を抽出する。5・7・5の各句は文節の切れ目で始まり、スパンは文節の切れ目で終わる。数字や英字を含む区間は対象外。文をまたがず、スパンは重ならない。" \
  --planner-llm claude-code:sonnet \
  --oracle bertgen.oracles.haiku:HaikuOracle \
  --corpus runs/corpus/ja575.jsonl \
  --examples 40000 --test-examples 2000 --examples-per-round 20000 \
  --max-rounds 6 --epochs 3
```

The LLM still defines the spec, plans the label mix and judges each round; the data comes from
the corpus. About a fifth of the paragraphs, chosen by a hash of their text, can only become test
data, so train and test never share a paragraph.

## What the oracle marks

Readings and morae come from unidic-lite. A span and each of its three parts begin at a bunsetsu
start, and the span ends at a bunsetsu end (end of text, a symbol, or the next bunsetsu start).
A short unit begins a bunsetsu unless it is a particle, auxiliary verb, suffix or symbol, or it
continues the previous unit:

- after a prefix: お|茶
- a noun after a noun: 一|件, 東京|都立|大学
- a 非自立可能 verb or adjective after a content word, an auxiliary verb or a conjunctive
  particle: 通っ|て|い|ます, 勉強|し, 雨|で|ある (but 家で|見た begins at 見)

A symbol or whitespace between two units always separates them. So a span cannot end at
「通ってい」 inside 「通っています」, and a part cannot begin at 件 inside 「一件ごとに」.

Units without a reading, and units with digits or latin letters of either width (1, lint, JSON,
full-width 50 which unidic reads フィフティー), never appear inside a span but do not stop the
oracle from labeling the rest of the passage. It abstains only on passages with no readable
unit at all, which are mostly English titles in Wikipedia. A span never crosses 。！？.

On the first 3000 passages (300 characters at most), the oracle abstains on 1.5% and finds at
least one 5-7-5 in 13.5% (471 spans). Of the 1021 passages that contain digits or latin letters,
67 have a span elsewhere in the text. The oracle takes about 0.2 ms per passage, so ten thousand
passages take about two seconds.

## Licensing

The aozorabunko-clean dataset keeps only works that Aozora Bunko lists as public domain, and is
itself released under CC BY 4.0. Wikipedia text is CC BY-SA 4.0.
The prepared corpus and the run directory stay local; `runs/` and `data/` are not committed.
