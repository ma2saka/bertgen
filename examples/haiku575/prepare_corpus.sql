-- Build a shuffled Japanese paragraph corpus for `bgen --corpus`.
--
-- Variables (set with -cmd): aozora = aozorabunko-dedupe-clean.jsonl.gz,
-- wikipedia = one 20231101.ja parquet shard. Writes runs/corpus/ja575.jsonl, one
-- {"text": ..., "source": ...} object per line, relative to the working directory.
--
-- Paragraphs are deduplicated on their kanji and kana alone, so variants that differ only in
-- punctuation, digits or Latin letters keep one row. Long paragraphs are kept whole; bgen
-- splits them into passages.

SET VARIABLE min_chars = 20;

CREATE TEMP TABLE documents AS
SELECT 'aozora' AS source, text
FROM read_json(getvariable('aozora'), columns = {text: 'VARCHAR'}, format = 'newline_delimited',
               maximum_object_size = 67108864)
UNION ALL
SELECT 'wikipedia' AS source, text
FROM read_parquet(getvariable('wikipedia'));

CREATE TEMP TABLE lines AS
SELECT source, text, regexp_replace(text, '[^\p{Han}\p{Hiragana}\p{Katakana}ー]', '', 'g') AS kana
FROM (
    SELECT source, regexp_replace(unnest(string_split(text, chr(10))), '^[\s　]+|[\s　]+$', '', 'g')
        AS text
    FROM documents
)
WHERE length(text) >= getvariable('min_chars');

CREATE TEMP TABLE paragraphs AS
SELECT min(text) AS text, arg_min(source, text) AS source
FROM lines
GROUP BY CASE WHEN kana = '' THEN text ELSE kana END;

SELECT source, count(*) AS paragraphs, sum(length(text)) AS chars
FROM paragraphs GROUP BY source ORDER BY source;

COPY (
    SELECT text, source FROM paragraphs ORDER BY hash(text, 575)
) TO 'runs/corpus/ja575.jsonl' (FORMAT json);
