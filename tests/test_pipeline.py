import io
import json
import re
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import pytest
from fakes import ScriptedLLM
from pydantic import BaseModel, ValidationError
from tiny_model import save_tiny_model
from transformers import pipeline

import bertgen.pipeline
from bertgen.cli import main
from bertgen.corpus import CorpusError, Pool, pool_of
from bertgen.llm import LLM
from bertgen.oracles import OracleError
from bertgen.pipeline import LLMFactory, RunConfig, load_state, run
from bertgen.stages.generate import (
    ClassBatch,
    ClassItem,
    ClassVerdict,
    ClassVerdictBatch,
    SpanBatch,
    SpanItem,
    SpanText,
    SpanVerdict,
    SpanVerdictBatch,
)
from bertgen.stages.plan import PlanDraft
from bertgen.store import ExampleStore
from bertgen.types import (
    NONE_KEY,
    Axis,
    DataPlan,
    Decision,
    Example,
    HParamPatch,
    LabelDef,
    OracleAgreement,
    Split,
    Stage,
    TaskKind,
    TaskSpec,
    TrainConfig,
    Verdict,
)

WORDS = ["apple", "river", "stone", "cloud", "lemon", "tiger", "piano", "maple"]


@pytest.fixture(scope="module")
def base_model(tmp_path_factory: pytest.TempPathFactory) -> str:
    return save_tiny_model(tmp_path_factory.mktemp("base"))


def make_config(tmp_path: Path, base_model: str, **overrides: object) -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "demo",
            "rule": "rule",
            "runs_dir": tmp_path,
            "model": base_model,
            "examples": 16,
            "test_examples": 8,
            "examples_per_round": 8,
            "max_rounds": 1,
            "target": 1.5,
            "workers": 2,
            "hparams": HParamPatch(epochs=1, batch_size=4, max_length=64, learning_rate=1e-3),
            **overrides,
        }
    )


def binary_label(text: str) -> str:
    return "pos" if "good" in text else "neg"


def number_spans(text: str) -> list[SpanText]:
    return [SpanText(text=m.group(), label="NUM") for m in re.finditer(r"\d+", text)]


def annotated_texts(user: str) -> list[str]:
    payload = json.loads(user.split("\n", 1)[1])
    return [item["text"] for item in payload]


def fake_llm(kind: TaskKind, verdicts: list[Decision | Verdict]) -> ScriptedLLM:
    """A deterministic oracle for every schema the pipeline asks for; a Verdict in `verdicts` is
    returned as is."""
    counter = iter(range(10**6))
    names = ["NUM"] if kind is TaskKind.SPAN else ["pos", "neg"]

    def text() -> str:
        i = next(counter)
        word = WORDS[i % len(WORDS)]
        if kind is TaskKind.SPAN:
            return f"{word} {i * 7} x" if i % 3 else f"{word} {word[::-1]} {i % 5 * 'z'}"
        return f"{word} {i} is {'good' if i % 2 else 'bad'}"

    def respond(system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        if schema is TaskSpec:
            return TaskSpec(
                rule="ignored",
                kind=kind,
                language="en",
                labels=[LabelDef(name=n, description=n) for n in names],
                input_description="short text",
                decision_criteria="oracle",
            )
        if schema is PlanDraft:
            return PlanDraft(
                label_weights={n: 1.0 for n in names},
                axes=[Axis(name="tone", values=["a", "b", "c"])],
                hard_cases=["near miss"],
                style_notes="plain",
            )
        if schema is ClassBatch:
            texts = [text() for _ in range(5)]
            return ClassBatch(items=[ClassItem(text=t, labels=[binary_label(t)]) for t in texts])
        if schema is SpanBatch:
            texts = [text() for _ in range(5)]
            return SpanBatch(items=[SpanItem(text=t, spans=number_spans(t)) for t in texts])
        if schema is ClassVerdictBatch:
            return ClassVerdictBatch(
                items=[
                    ClassVerdict(index=i, labels=[binary_label(t)])
                    for i, t in enumerate(annotated_texts(user))
                ]
            )
        if schema is SpanVerdictBatch:
            return SpanVerdictBatch(
                items=[
                    SpanVerdict(index=i, spans=number_spans(t))
                    for i, t in enumerate(annotated_texts(user))
                ]
            )
        if schema is Verdict:
            decision = verdicts.pop(0)
            if isinstance(decision, Verdict):
                return decision
            focus = ["longer texts"] if decision is Decision.MORE_DATA else []
            return Verdict(decision=decision, reason="scripted", focus=focus)
        raise AssertionError(f"unexpected schema {schema.__name__}")

    return ScriptedLLM(responder=respond)


def providers(llm: ScriptedLLM) -> LLMFactory:
    return lambda _: llm


def unavailable(spec: str) -> LLM:
    raise AssertionError(f"no LLM should be built, got {spec}")


def asked(llm: ScriptedLLM, schema: type[BaseModel]) -> int:
    return sum(1 for _, _, s in llm.calls if s is schema)


def mtime(model_dir: Path) -> int:
    return (model_dir / "config.json").stat().st_mtime_ns


def test_more_data_then_accept_resumes(tmp_path: Path, base_model: str) -> None:
    first = fake_llm(TaskKind.BINARY, [Decision.MORE_DATA])
    model = run(make_config(tmp_path, base_model), providers(first))
    run_dir = tmp_path / "demo"
    assert model == run_dir / "model"
    assert model.resolve() == (run_dir / "rounds/0/model").resolve()
    assert not (run_dir / "rounds/1").exists()
    stamp = mtime(run_dir / "rounds/0/model")

    second = fake_llm(TaskKind.BINARY, [Decision.ACCEPT])
    run(make_config(tmp_path, base_model, max_rounds=3), providers(second))
    assert asked(second, TaskSpec) == asked(second, PlanDraft) == 0
    assert asked(second, ClassBatch) > 0
    assert mtime(run_dir / "rounds/0/model") == stamp
    assert (run_dir / "rounds/1/model/config.json").exists()
    assert not (run_dir / "rounds/2").exists()
    verdict = Verdict.model_validate_json((run_dir / "rounds/1/verdict.json").read_text())
    assert verdict.decision is Decision.ACCEPT
    assert "## Verdict: accept" in (run_dir / "rounds/1/report.md").read_text()
    assert load_state(run_dir).round == 1
    with ExampleStore(run_dir / "data.db") as store:
        assert store.load(Split.TRAIN, round_=1)
        assert len(store.load(Split.TEST)) == 8

    run(make_config(tmp_path, base_model, max_rounds=3), unavailable)
    assert (run_dir / "rounds/1/valid_metrics.json").exists()

    run(make_config(tmp_path, base_model, max_rounds=1), unavailable)

    def rank(n: int) -> tuple[float, float]:
        return tuple(
            json.loads((run_dir / f"rounds/{n}/{name}.json").read_text())["primary"]
            for name in ("metrics", "valid_metrics")
        )

    best = max((0, 1), key=rank)
    assert model.resolve() == (run_dir / f"rounds/{best}/model").resolve()

    with ExampleStore(run_dir / "data.db") as store:
        test_texts = store.texts(Split.TEST)
    prompts = [
        user for llm in (first, second) for _, user, s in llm.calls if s is not ClassVerdictBatch
    ]
    assert not any(text in user for text in test_texts for user in prompts)

    classifier = pipeline("text-classification", model=str(run_dir / "model"))
    result = classifier("apple 3 is good")
    assert isinstance(result, list)
    assert result[0]["label"] in {"pos", "neg"}


def test_span_run_and_cli(
    tmp_path: Path,
    base_model: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = fake_llm(TaskKind.SPAN, [Decision.STOP])
    run(make_config(tmp_path, base_model, verify=True), providers(llm))
    assert asked(llm, SpanVerdictBatch) > 0
    run_dir = tmp_path / "demo"
    assert (run_dir / "rounds/0/report.md").exists()

    tagger = pipeline("token-classification", model=str(run_dir / "model"))
    assert isinstance(tagger("river 42 x"), list)

    main(["status", "--name", "demo", "--runs-dir", str(tmp_path)])
    status = capsys.readouterr().out
    assert "round 0: primary" in status
    assert "-> stop *" in status

    main(["predict", "--name", "demo", "--runs-dir", str(tmp_path), "river 42 x", "no digits"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["text"] for line in lines] == ["river 42 x", "no digits"]
    assert all("spans" in line or "labels" in line for line in lines)
    assert all(
        all("score" in span for span in line.get("spans", [])) or "scores" in line for line in lines
    )

    monkeypatch.setattr("sys.stdin", io.StringIO("river 42 x\n\nno digits\r\n"))
    main(["predict", "--name", "demo", "--runs-dir", str(tmp_path)])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["text"] for line in lines] == ["river 42 x", "no digits"]


def test_stop_after_spec_then_resume(tmp_path: Path, base_model: str) -> None:
    llm = fake_llm(TaskKind.BINARY, [])
    spec_path = run(make_config(tmp_path, base_model), providers(llm), stop_after=Stage.SPEC)
    assert spec_path == tmp_path / "demo/spec.json"
    assert asked(llm, TaskSpec) == 1
    assert asked(llm, PlanDraft) == 0
    db = run(make_config(tmp_path, base_model), providers(llm), stop_after=Stage.GENERATE)
    assert db.exists()
    assert asked(llm, TaskSpec) == 1
    assert not (tmp_path / "demo/rounds").exists()


@pytest.mark.parametrize("warm_start", [True, False])
def test_rounds_warm_start_from_the_previous_model(
    tmp_path: Path, base_model: str, monkeypatch: pytest.MonkeyPatch, warm_start: bool
) -> None:
    inits: list[Path | None] = []
    real_train = bertgen.pipeline.train

    def recording(
        spec: TaskSpec,
        train_set: list[Example],
        valid_set: list[Example],
        config: TrainConfig,
        out_dir: Path,
        init: Path | None = None,
    ) -> Path:
        inits.append(init)
        return real_train(spec, train_set, valid_set, config, out_dir, init)

    monkeypatch.setattr(bertgen.pipeline, "train", recording)
    llm = fake_llm(TaskKind.BINARY, [Decision.MORE_DATA, Decision.STOP])
    config = make_config(tmp_path, base_model, max_rounds=2, warm_start=warm_start)
    run(config, providers(llm))
    run_dir = tmp_path / "demo"
    previous = run_dir / "rounds/0/model" if warm_start else None
    assert inits == [None, previous]
    judge_prompts = [user for _, user, schema in llm.calls if schema is Verdict]
    assert all(("extra passes" in user) is warm_start for user in judge_prompts)
    report = (run_dir / "rounds/1/report.md").read_text()
    assert ("initialized from round 0" in report) is warm_start

    shutil.rmtree(run_dir / "rounds/1/model")
    flipped = make_config(tmp_path, base_model, max_rounds=2, warm_start=not warm_start)
    run(flipped, unavailable)
    assert inits == [None, previous, previous]


def test_more_data_applies_the_verdict_hparams(tmp_path: Path, base_model: str) -> None:
    more = Verdict(
        decision=Decision.MORE_DATA, reason="r", focus=["x"], hparams=HParamPatch(epochs=2)
    )
    llm = fake_llm(TaskKind.BINARY, [more, Decision.STOP])
    run(make_config(tmp_path, base_model, max_rounds=2), providers(llm))
    saved = (tmp_path / "demo/rounds/1/train_config.json").read_text()
    assert TrainConfig.model_validate_json(saved).epochs == 2


def test_unchanged_tune_does_not_retrain(tmp_path: Path, base_model: str) -> None:
    llm = fake_llm(TaskKind.BINARY, [Decision.TUNE])
    run(make_config(tmp_path, base_model, max_rounds=3), providers(llm))
    assert (tmp_path / "demo/rounds/0/verdict.json").exists()
    assert not (tmp_path / "demo/rounds/1").exists()


def test_oracle_relabels_and_records_agreement(tmp_path: Path, base_model: str) -> None:
    llm = fake_llm(TaskKind.BINARY, [Decision.STOP])
    run(make_config(tmp_path, base_model, oracle="fakes:FakeOracle"), providers(llm))
    run_dir = tmp_path / "demo"

    verified = [t for _, u, s in llm.calls if s is ClassVerdictBatch for t in annotated_texts(u)]
    assert verified
    assert all("apple" in text for text in verified)

    agreement = load_state(run_dir).oracle_agreement
    with ExampleStore(run_dir / "data.db") as store:
        for split in Split:
            examples = store.load(split)
            stats = agreement[split]
            assert stats.labeled + stats.abstained == len(examples)
            for ex in examples:
                if "river" in ex.text:
                    assert ex.labels == ["neg"]
                elif "apple" not in ex.text:
                    assert ex.labels == [binary_label(ex.text)]
    total = sum((agreement[split] for split in Split), start=OracleAgreement())
    assert total.agreed > 0
    assert total.disagreed > 0
    assert total.abstained > 0
    report = (run_dir / "rounds/0/report.md").read_text()
    assert "## Oracle agreement with LLM labels" in report

    run(make_config(tmp_path, base_model, oracle="fakes:FakeOracle"), unavailable)
    assert load_state(run_dir).oracle_agreement == agreement


def test_total_oracle_skips_verification(tmp_path: Path, base_model: str) -> None:
    llm = fake_llm(TaskKind.BINARY, [])
    config = make_config(tmp_path, base_model, oracle="fakes:TotalOracle")
    run(config, providers(llm), stop_after=Stage.GENERATE)
    assert asked(llm, ClassVerdictBatch) == 0
    agreement = load_state(tmp_path / "demo").oracle_agreement
    assert all(stats.abstained == 0 for stats in agreement.values())
    assert agreement[Split.TEST].labeled == 8


def test_span_oracle_stores_its_rejections_with_the_llm_spans(
    tmp_path: Path, base_model: str
) -> None:
    llm = fake_llm(TaskKind.SPAN, [])
    config = make_config(tmp_path, base_model, oracle="fakes:FakeSpanOracle")
    db = run(config, providers(llm), stop_after=Stage.GENERATE)

    with ExampleStore(db) as store:
        rivers = [e for split in Split for e in store.load(split) if "river" in e.text]
        agreement = store.agreement()
    assert rivers
    assert all(e.spans == [] for e in rivers)
    assert load_state(tmp_path / "demo").oracle_agreement == agreement
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT text, spans, llm_spans FROM examples WHERE oracle = 'disagreed'"
        ).fetchall()
        none_rows = conn.execute(
            "SELECT COUNT(*) FROM examples WHERE spans = '[]' AND labels = '[]'"
        ).fetchone()
    assert rows
    assert sum(a.disagreed for a in agreement.values()) == len(rows)
    for text, spans, llm_spans in rows:
        assert "river" in text and spans == "[]" and llm_spans != "[]"
    with ExampleStore(db) as store:
        counts = store.counts()
    assert sum(c.get(NONE_KEY, 0) for c in counts.values()) == none_rows[0]


def test_test_writer_never_sees_plan_seeds(tmp_path: Path, base_model: str) -> None:
    shared = fake_llm(TaskKind.BINARY, [])
    writers = {
        name: ScriptedLLM(responder=shared.ask, name=name)
        for name in ("anthropic:gen", "anthropic:eval")
    }
    config = make_config(
        tmp_path, base_model, llm="anthropic:gen", eval_llm="anthropic:eval", test_examples=60
    )
    plan_path = run(config, writers.__getitem__, stop_after=Stage.PLAN)
    seeds = [f"seed text number {i}" for i in range(6)]
    plan = DataPlan.model_validate_json(plan_path.read_text())
    plan_path.write_text(plan.model_copy(update={"seeds": seeds}).model_dump_json())

    run(config, writers.__getitem__, stop_after=Stage.GENERATE)

    def prompts(name: str) -> list[str]:
        return [user for _, user, s in writers[name].calls if s is ClassBatch]

    assert prompts("anthropic:eval")
    assert not any(seed in user for seed in seeds for user in prompts("anthropic:eval"))
    assert any(seed in user for seed in seeds for user in prompts("anthropic:gen"))


def test_oracle_cannot_change_once_data_exists(tmp_path: Path, base_model: str) -> None:
    llm = fake_llm(TaskKind.BINARY, [])
    run(make_config(tmp_path, base_model), providers(llm), stop_after=Stage.GENERATE)
    argv = ["--name", "demo", "--runs-dir", str(tmp_path), "--oracle", "fakes:FakeOracle"]
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2


def test_unloadable_oracle_fails_before_planning(tmp_path: Path, base_model: str) -> None:
    llm = fake_llm(TaskKind.BINARY, [])
    with pytest.raises(OracleError, match="no callable"):
        run(make_config(tmp_path, base_model, oracle="fakes:Missing"), providers(llm))
    assert asked(llm, PlanDraft) == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["--name", "fresh"],
        ["--name", "fresh", "--rule", "r", "--model", "ja-130"],
        ["--name", "fresh", "--rule", "r", "--llm", "nonsense"],
        ["--name", "fresh", "--rule", "r", "--oracle", "nonsense"],
        ["--name", "fresh", "--rule", "r", "--corpus", "c.txt"],
        ["--name", "fresh", "--rule", "r", "--oracle", "fakes:FakeOracle", "--corpus", "c.csv"],
        ["--name", "../x", "--rule", "r"],
        ["status", "--name", "missing"],
        ["predict", "--name", "missing", "text"],
    ],
)
def test_cli_rejects_bad_input_without_traceback(tmp_path: Path, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([*argv, "--runs-dir", str(tmp_path)])
    assert exit_info.value.code == 2
    assert not (tmp_path / "fresh").exists()


def write_corpus(path: Path, lines: int = 400) -> Path:
    path.write_text(
        "".join(
            f"{WORDS[i % len(WORDS)]} {i} is {'good' if i % 2 else 'bad'}\n" for i in range(lines)
        )
    )
    return path


def test_corpus_run_labels_real_text_with_the_oracle(tmp_path: Path, base_model: str) -> None:
    corpus = write_corpus(tmp_path / "corpus.txt")
    config = make_config(
        tmp_path, base_model, oracle="fakes:TotalOracle", corpus=corpus, max_rounds=2
    )
    llm = fake_llm(TaskKind.BINARY, [Decision.MORE_DATA, Decision.STOP])
    run(config, providers(llm))

    assert asked(llm, ClassBatch) == asked(llm, ClassVerdictBatch) == 0
    run_dir = tmp_path / "demo"
    assert (run_dir / "rounds/1/verdict.json").exists()
    assert load_state(run_dir).oracle_agreement == {}
    with sqlite3.connect(run_dir / "data.db") as conn:
        rows = conn.execute(
            "SELECT text, split, round, source, oracle, labels, llm_labels FROM examples"
        ).fetchall()
    assert Counter((split == "test", round_) for _, split, round_, *_ in rows) == {
        (True, 0): 8,
        (False, 0): 16,
        (False, 1): 8,
    }
    for text, split, _, source, oracle, labels, llm_labels in rows:
        assert source == "corpus:corpus.txt"
        assert (pool_of(text) is Pool.TEST) == (split == "test")
        assert oracle is None
        assert labels == llm_labels
        gold = "pos" if "good" in text and "river" not in text else "neg"
        assert json.loads(labels) == [gold]
    test_labels = Counter(
        json.loads(labels)[0] for _, split, *_, labels, _ in rows if split == "test"
    )
    assert test_labels == {"pos": 4, "neg": 4}

    run(config, unavailable)
    argv = ["--name", "demo", "--runs-dir", str(tmp_path)]
    for extra in (["--corpus", str(tmp_path / "x.txt")], ["--corpus-max-chars", "50"]):
        with pytest.raises(SystemExit) as exit_info:
            main([*argv, *extra])
        assert exit_info.value.code == 2


def test_corpus_resume_accepts_another_spelling_of_the_same_file(
    tmp_path: Path, base_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = write_corpus(tmp_path / "corpus.txt")
    run(
        make_config(tmp_path, base_model, oracle="fakes:TotalOracle", corpus=corpus),
        providers(fake_llm(TaskKind.BINARY, [Decision.STOP])),
    )
    resumed: list[RunConfig] = []

    def fake_run(config: RunConfig, stop_after: Stage | None) -> None:
        resumed.append(config)

    monkeypatch.setattr("bertgen.cli.run", fake_run)
    same = tmp_path / ".." / tmp_path.name / "corpus.txt"
    assert same != corpus
    main(["--name", "demo", "--runs-dir", str(tmp_path), "--corpus", str(same)])
    assert [c.corpus for c in resumed] == [same]
    main(["--name", "demo", "--runs-dir", str(tmp_path), "--no-warm-start"])
    assert [c.warm_start for c in resumed] == [True, False]


def test_corpus_span_run_samples_negatives_without_a_none_weight(
    tmp_path: Path, base_model: str
) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "".join(f"{WORDS[i % len(WORDS)]} {i} {'river' if i % 3 else 'x'}\n" for i in range(400))
    )
    config = make_config(tmp_path, base_model, oracle="fakes:FakeSpanOracle", corpus=corpus)
    llm = fake_llm(TaskKind.SPAN, [Decision.STOP])
    run(config, providers(llm))

    plan = json.loads((tmp_path / "demo/plan.json").read_text())
    assert NONE_KEY not in plan["label_weights"]
    with sqlite3.connect(tmp_path / "demo/data.db") as conn:
        rows = conn.execute("SELECT split, spans FROM examples").fetchall()
    negatives = Counter(split == "test" for split, spans in rows if not json.loads(spans))
    assert negatives == {True: 2, False: 4}


def test_corpus_test_set_with_an_empty_bucket_is_an_error(tmp_path: Path, base_model: str) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(f"{WORDS[i % len(WORDS)]} {i} is bad\n" for i in range(400)))
    config = make_config(tmp_path, base_model, oracle="fakes:TotalOracle", corpus=corpus)
    with pytest.raises(CorpusError, match="empty bucket: pos 0/4, neg 4/4"):
        run(config, providers(fake_llm(TaskKind.BINARY, [])))


def test_corpus_requires_an_oracle_and_an_existing_file(tmp_path: Path, base_model: str) -> None:
    corpus = write_corpus(tmp_path / "corpus.txt")
    with pytest.raises(ValidationError, match="requires an oracle"):
        make_config(tmp_path, base_model, corpus=corpus)
    llm = fake_llm(TaskKind.BINARY, [])
    missing = make_config(
        tmp_path, base_model, oracle="fakes:TotalOracle", corpus=tmp_path / "missing.jsonl"
    )
    with pytest.raises(CorpusError, match="not found"):
        run(missing, providers(llm))
    assert asked(llm, TaskSpec) == asked(llm, PlanDraft) == 0
