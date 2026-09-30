import pytest

from bertgen.oracles import OracleError, load_oracle
from bertgen.oracles.haiku import HaikuOracle, Unit, count_morae, starts_bunsetsu
from bertgen.types import LabelDef, Span, TaskKind, TaskSpec

pytest.importorskip("fugashi")


def make_spec(kind: TaskKind = TaskKind.SPAN, labels: tuple[str, ...] = ("haiku",)) -> TaskSpec:
    return TaskSpec(
        rule="find 5-7-5",
        kind=kind,
        language="ja",
        labels=[LabelDef(name=name, description=name) for name in labels],
        input_description="text",
        decision_criteria="mora count",
    )


@pytest.fixture(scope="module")
def oracle() -> HaikuOracle:
    return HaikuOracle(make_spec())


@pytest.mark.parametrize(
    ("pron", "morae"),
    [("キャ", 1), ("キャット", 3), ("ガッコー", 4), ("ホンヤ", 3), ("ニホン", 3), ("シュミ", 2)],
)
def test_count_morae(pron: str, morae: int):
    assert count_morae(pron) == morae


def test_whole_text_is_one_span(oracle: HaikuOracle):
    text = "古池や蛙飛び込む水の音"
    ex = oracle.label(text)
    assert ex is not None
    assert ex.spans == [Span(start=0, end=len(text), label="haiku")]


def test_embedded_span_offsets(oracle: HaikuOracle):
    haiku = "柿くへば鐘が鳴るなり法隆寺"
    text = f"子規は「{haiku}」と詠んだ。"
    ex = oracle.label(text)
    assert ex is not None
    start = text.index(haiku)
    assert ex.spans == [Span(start=start, end=start + len(haiku), label="haiku")]


def test_wrong_count_yields_no_span(oracle: HaikuOracle):
    ex = oracle.label("古池や蛙飛び込む水の音楽")
    assert ex is not None
    assert ex.spans == []


def test_segment_starting_with_particle_rejected(oracle: HaikuOracle):
    ex = oracle.label("古池や蛙飛び込む水の音")
    assert ex is not None and len(ex.spans) == 1
    ex = oracle.label("や古池蛙飛び込む水の音")
    assert ex is not None
    assert ex.spans == []


@pytest.mark.parametrize("text", ["白黒の映画なのです。その色の", "古池や！蛙飛び込む水の音"])
def test_span_does_not_cross_a_sentence_end(oracle: HaikuOracle, text: str):
    ex = oracle.label(text)
    assert ex is not None
    assert ex.spans == []
    joined = oracle.label(text.replace("。", "、").replace("！", "、"))
    assert joined is not None and len(joined.spans) == 1


def spans_of(oracle: HaikuOracle, text: str) -> list[str]:
    ex = oracle.label(text)
    assert ex is not None
    return [text[s.start : s.end] for s in ex.spans]


@pytest.mark.parametrize(
    ("text", "bunsetsu"),
    [
        ("家で見た本", ["家で", "見た", "本"]),
        ("通っています", ["通っています"]),
        ("勉強している", ["勉強している"]),
        ("高すぎる", ["高すぎる"]),
        ("雨である", ["雨である"]),
        ("一件ごとに", ["一件ごとに"]),
        ("お茶を飲んだ", ["お茶を", "飲んだ"]),
        ("東京都立大学へ", ["東京都立大学へ"]),
        ("足だけがやけに重くて、駅まで", ["足だけが", "やけに", "重くて、", "駅まで"]),
        ("夢みたいな話", ["夢みたいな", "話"]),
        ("少年のような目", ["少年のような", "目"]),
    ],
)
def test_starts_bunsetsu(oracle: HaikuOracle, text: str, bunsetsu: list[str]):
    starts = [m.start for m in oracle._analyze(text) if m.starts_bunsetsu]  # pyright: ignore[reportPrivateUsage]
    ends = [*starts[1:], len(text)]
    assert [text[a:b] for a, b in zip(starts, ends, strict=True)] == bunsetsu


def test_starts_bunsetsu_needs_a_content_word():
    noun = Unit(pos1="名詞", pos2="普通名詞", is_symbol=False)
    particle = Unit(pos1="助詞", pos2="格助詞", is_symbol=False)
    assert starts_bunsetsu(None, noun, separated=False)
    assert not starts_bunsetsu(None, particle, separated=False)
    assert not starts_bunsetsu(noun, noun, separated=False)
    assert starts_bunsetsu(noun, noun, separated=True)


@pytest.mark.parametrize(
    "text",
    [
        "型チェック・テストはすべて通っています",
        "空行は読み飛ばし、一件ごとに処理",
        "全部夢みたいな話信じない",
        "今日のこの静かな部屋で本を読む",
    ],
)
def test_span_edges_stay_on_bunsetsu_boundaries(oracle: HaikuOracle, text: str):
    assert spans_of(oracle, text) == []


@pytest.mark.parametrize(
    "text",
    ["雨なので家でのんびり過ごします", "足だけがやけに重くて、駅までの"],
)
def test_prose_positives(oracle: HaikuOracle, text: str):
    assert spans_of(oracle, text) == [text]


def test_spans_skip_digits_and_latin(oracle: HaikuOracle):
    haiku = "古池や蛙飛び込む水の音"
    text = f"1 件目の lint で{haiku}、２０２４年"
    assert spans_of(oracle, text) == [haiku]
    assert spans_of(oracle, "古池や蛙飛び込む水の音 x") == [haiku]
    assert spans_of(oracle, "古池や蛙飛び込む水の音ｘ") == []


@pytest.mark.parametrize("text", ["lint JSON", "123 abc", "", "、。"])
def test_abstains_without_any_reading(oracle: HaikuOracle, text: str):
    assert oracle.label(text) is None


def test_requires_span_kind():
    with pytest.raises(ValueError, match="SPAN"):
        HaikuOracle(make_spec(kind=TaskKind.BINARY))


def test_requires_single_label():
    with pytest.raises(ValueError, match="one label"):
        HaikuOracle(make_spec(labels=("a", "b")))


def test_broken_dictionary_is_an_oracle_error(monkeypatch: pytest.MonkeyPatch):
    import fugashi  # pyright: ignore[reportMissingTypeStubs]

    def broken() -> None:
        raise RuntimeError("Failed initializing MeCab")

    monkeypatch.setattr(fugashi, "Tagger", broken)
    with pytest.raises(OracleError, match="uv sync --extra ja"):
        load_oracle("bertgen.oracles.haiku:HaikuOracle", make_spec())


def test_load_oracle():
    loaded = load_oracle("bertgen.oracles.haiku:HaikuOracle", make_spec())
    assert isinstance(loaded, HaikuOracle)


@pytest.mark.parametrize(
    "ref",
    [
        "nocolon",
        ":attr",
        "mod:",
        "no.such.module:X",
        "bertgen.oracles.haiku:missing",
        "bertgen.oracles.haiku:PATTERN",
    ],
)
def test_load_oracle_bad_ref(ref: str):
    with pytest.raises(ValueError):
        load_oracle(ref, make_spec())
