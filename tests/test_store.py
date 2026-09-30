import sqlite3
from pathlib import Path

from bertgen.store import ExampleStore
from bertgen.text import normalize
from bertgen.types import NONE_KEY, Draft, Example, OracleAgreement, OracleOutcome, Span, Split


def test_normalize() -> None:
    assert normalize("  \uff21\uff22\uff23 \n def\t") == "abc def"


def test_add_dedups_and_load(tmp_path: Path) -> None:
    with ExampleStore(tmp_path / "d.db") as store:
        first = [Example(text="Hello  World", labels=["pos"]), Example(text="x", labels=["neg"])]
        assert store.add(((Draft.plain(e), Split.TRAIN) for e in first), 0, "gen") == 2
        dup = [
            (Example(text="hello world", labels=["neg"]), Split.TRAIN),
            (Example(text="y", labels=["neg"]), Split.TRAIN),
            (Example(text="z", labels=["neg"]), Split.TEST),
        ]
        dup = [(Draft.plain(e), split) for e, split in dup]
        assert store.add(dup, 1, "gen") == 2
        assert [e.text for e in store.load(Split.TRAIN)] == ["Hello  World", "x", "y"]
        assert [e.text for e in store.load(Split.TRAIN, round_=1)] == ["y"]
        assert store.load(Split.VALID) == []
        assert store.texts() == ["Hello  World", "x", "y", "z"]
        assert store.texts(Split.TRAIN, Split.VALID) == ["Hello  World", "x", "y"]


def test_counts_labels_and_spans(tmp_path: Path) -> None:
    with ExampleStore(tmp_path / "d.db") as store:
        train = [
            Example(text="a", labels=["x", "y"]),
            Example(text="b", labels=["x"]),
            Example(text="c"),
        ]
        store.add(((Draft.plain(e), Split.TRAIN) for e in train), 0, "gen")
        span = Span(start=0, end=1, label="HAIKU")
        test = [Example(text="d e", spans=[span, span]), Example(text="f")]
        store.add(((Draft.plain(e), Split.TEST) for e in test), 0, "eval")
        counts = store.counts()
        assert counts[Split.TRAIN] == {"x": 2, "y": 1, NONE_KEY: 1}
        assert counts[Split.TEST] == {"HAIKU": 2, NONE_KEY: 1}
        assert counts[Split.VALID] == {}


def test_roundtrip_spans_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "d.db"
    ex = Example(text="abc def", spans=[Span(start=4, end=7, label="W")])
    with ExampleStore(path) as store:
        store.add([(Draft.plain(ex), Split.VALID)], 2, "gen")
    with ExampleStore(path) as store:
        assert store.load(Split.VALID) == [ex]


def test_oracle_outcomes_and_llm_labels_are_kept(tmp_path: Path) -> None:
    path = tmp_path / "d.db"
    gold = Example(text="古池や", spans=[])
    written = Example(text="古池や", spans=[Span(start=0, end=3, label="haiku")])
    rows = [
        (Draft(gold, written, OracleOutcome.DISAGREED), Split.TEST),
        (Draft.plain(Example(text="a")), Split.TEST),
        (Draft(Example(text="b"), Example(text="b"), OracleOutcome.AGREED), Split.TRAIN),
        (Draft(Example(text="c"), Example(text="c"), OracleOutcome.ABSTAINED), Split.TRAIN),
    ]
    with ExampleStore(path) as store:
        store.add(rows, 0, "gen")
        assert store.load(Split.TEST)[0] == gold
        assert store.agreement() == {
            Split.TEST: OracleAgreement(disagreed=1),
            Split.TRAIN: OracleAgreement(agreed=1, abstained=1),
        }
    with sqlite3.connect(path) as db:
        audit = db.execute("SELECT spans, llm_spans FROM examples WHERE oracle = 'disagreed'")
        assert audit.fetchall() == [("[]", '[{"start":0,"end":3,"label":"haiku"}]')]


def test_opens_a_database_without_audit_columns(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE examples (id INTEGER PRIMARY KEY, text TEXT NOT NULL,"
            " norm TEXT NOT NULL UNIQUE, labels TEXT NOT NULL, spans TEXT NOT NULL,"
            " split TEXT NOT NULL, round INTEGER NOT NULL, source TEXT NOT NULL,"
            " created_at TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO examples VALUES (1, 'old', 'old', '[\"x\"]', '[]', 'train', 0, 'g', 't')"
        )
    with ExampleStore(path) as store:
        store.add([(Draft.plain(Example(text="new", labels=["y"])), Split.TRAIN)], 1, "g")
        assert [e.text for e in store.load(Split.TRAIN)] == ["old", "new"]
        assert store.agreement() == {}
