"""SQLite store for synthetic examples, deduplicated by normalized text."""

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import TypeAdapter

from bertgen.text import normalize
from bertgen.types import NONE_KEY, Draft, Example, OracleAgreement, OracleOutcome, Span, Split

_SCHEMA = """
CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    norm TEXT NOT NULL UNIQUE,
    labels TEXT NOT NULL,
    spans TEXT NOT NULL,
    split TEXT NOT NULL,
    round INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    oracle TEXT,
    llm_labels TEXT,
    llm_spans TEXT
)
"""

_ADDED_COLUMNS = ("oracle", "llm_labels", "llm_spans")

_AGREEMENT_SQL = """
SELECT split, oracle, COUNT(*) FROM examples WHERE oracle IS NOT NULL GROUP BY split, oracle
"""

_COUNTS_SQL = """
SELECT split, label, COUNT(*) FROM (
    SELECT e.split AS split, j.value AS label
    FROM examples e, json_each(e.labels) j
    UNION ALL
    SELECT e.split, json_extract(j.value, '$.label')
    FROM examples e, json_each(e.spans) j
    UNION ALL
    SELECT split, ? FROM examples WHERE labels = '[]' AND spans = '[]'
)
GROUP BY split, label
"""

_labels_adapter = TypeAdapter(list[str])
_spans_adapter = TypeAdapter(list[Span])


class ExampleStore:
    """Persistent example collection; use as a context manager or call `close`.

    Besides the stored label, each row keeps the label its writer LLM gave (`llm_labels`,
    `llm_spans`) and the oracle outcome (`oracle`, NULL when no oracle judged it).
    """

    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(path)
        with self._db:
            self._db.execute(_SCHEMA)
            present = {row[1] for row in self._db.execute("PRAGMA table_info(examples)")}
            for column in _ADDED_COLUMNS:
                if column not in present:
                    self._db.execute(f"ALTER TABLE examples ADD COLUMN {column} TEXT")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def add(self, rows: Iterable[tuple[Draft, Split]], round_: int, source: str) -> int:
        """Insert, in one transaction, drafts whose normalized text is new; return the count."""
        now = datetime.now(UTC).isoformat()
        inserted = 0
        with self._db:
            for draft, split in rows:
                ex, written = draft.example, draft.written
                cursor = self._db.execute(
                    "INSERT OR IGNORE INTO examples (text, norm, labels, spans, split, round,"
                    " source, created_at, oracle, llm_labels, llm_spans)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ex.text,
                        normalize(ex.text),
                        _labels_adapter.dump_json(ex.labels).decode(),
                        _spans_adapter.dump_json(ex.spans).decode(),
                        split.value,
                        round_,
                        source,
                        now,
                        draft.outcome and draft.outcome.value,
                        _labels_adapter.dump_json(written.labels).decode(),
                        _spans_adapter.dump_json(written.spans).decode(),
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def load(self, split: Split, round_: int | None = None) -> list[Example]:
        """Return examples of `split` in insertion order, optionally only from one round."""
        sql = "SELECT text, labels, spans FROM examples WHERE split = ?"
        params: tuple[str, ...] | tuple[str, int] = (split.value,)
        if round_ is not None:
            sql += " AND round = ?"
            params = (split.value, round_)
        rows = self._db.execute(sql + " ORDER BY id", params).fetchall()
        return [
            Example(
                text=text,
                labels=_labels_adapter.validate_json(labels),
                spans=_spans_adapter.validate_json(spans),
            )
            for text, labels, spans in rows
        ]

    def counts(self) -> dict[Split, dict[str, int]]:
        """Return per-label counts for every split; span labels count once per span, and examples
        without any label or span count under NONE_KEY."""
        result: dict[Split, dict[str, int]] = {split: {} for split in Split}
        for split, label, n in self._db.execute(_COUNTS_SQL, (NONE_KEY,)):
            result[Split(split)][label] = n
        return result

    def agreement(self) -> dict[Split, OracleAgreement]:
        """Oracle outcomes of the stored examples per split; splits without any are omitted."""
        counts: dict[Split, dict[str, int]] = {}
        for split, outcome, n in self._db.execute(_AGREEMENT_SQL):
            counts.setdefault(Split(split), {})[OracleOutcome(outcome).value] = n
        return {split: OracleAgreement.model_validate(c) for split, c in counts.items()}

    def texts(self, *splits: Split) -> list[str]:
        """Return raw texts in insertion order, restricted to `splits` when any are given."""
        chosen = [split.value for split in splits or Split]
        marks = ", ".join("?" * len(chosen))
        sql = f"SELECT text FROM examples WHERE split IN ({marks}) ORDER BY id"
        return [text for (text,) in self._db.execute(sql, chosen)]
