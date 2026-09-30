"""Oracle extracting Japanese 5-7-5 mora spans."""

import unicodedata
from dataclasses import dataclass

from bertgen.types import Example, Span, TaskKind, TaskSpec

SMALL_KANA = frozenset("ァィゥェォャュョヮ")
SYMBOL_POS = frozenset({"補助記号", "記号", "空白"})
DEPENDENT_POS = frozenset({"助詞", "助動詞", "接尾辞"})
AUXILIARY_HOSTS = frozenset({"名詞", "動詞", "形容詞", "形状詞", "助動詞", "接尾辞"})
PATTERN = (5, 7, 5)
SENTENCE_TERMINATORS = frozenset("。．！？!?")


def count_morae(pron: str) -> int:
    """Count morae in a katakana reading; small ャュョ-type kana attach to the previous mora."""
    return sum(ch not in SMALL_KANA for ch in pron)


@dataclass(frozen=True)
class Unit:
    """One unidic short unit, reduced to the fields the bunsetsu rule needs. A symbol is a
    punctuation or whitespace unit without a reading; unidic gives some kana (く) symbol POS with
    a reading, and those count as words."""

    pos1: str
    pos2: str
    is_symbol: bool


def starts_bunsetsu(prev: Unit | None, cur: Unit, separated: bool) -> bool:
    """Whether `cur` begins a bunsetsu, given the unit before it.

    `separated` means a symbol or whitespace lies between the two units. Particles, auxiliary
    verbs, auxiliary stems (よう, そう, みたい), suffixes and symbols never begin one. Otherwise
    `cur` continues the previous bunsetsu when it follows a prefix (お|茶), when both units are
    nouns (一|件, 東京|都立), or when it is a 非自立可能 verb or adjective (いる, する, ない)
    directly after a content word, an auxiliary verb or a conjunctive particle (通っ|て|い,
    勉強|し, 高|すぎる, 雨|で|ある). After any other particle it begins one (家で|見た).

    POS alone cannot tell an adverbial noun from a compound head, nor a dropped particle from a
    サ変 noun, so 今日|学校で and 学校|行く stay one bunsetsu and no part ends inside them.
    """
    if cur.is_symbol or cur.pos1 in DEPENDENT_POS or cur.pos2 == "助動詞語幹":
        return False
    if prev is None or separated:
        return True
    if prev.pos1 == "接頭辞":
        return False
    if prev.pos1 == "名詞" and cur.pos1 == "名詞":
        return False
    after_host = prev.pos1 in AUXILIARY_HOSTS or prev.pos2 == "接続助詞"
    return not (cur.pos2 == "非自立可能" and cur.pos1 in {"動詞", "形容詞"} and after_host)


@dataclass(frozen=True)
class Morpheme:
    start: int
    end: int
    morae: int | None
    is_symbol: bool
    is_terminator: bool
    is_prenominal: bool
    starts_bunsetsu: bool


def _reading(surface: str, pron: str | None) -> str | None:
    """The katakana reading, or None when there is none or the surface has digits or latin
    letters in any width, whose dictionary readings (full-width 50 read フィフティー) are not
    trusted."""
    if not pron or pron == "*":
        return None
    if any(c.isascii() and c.isalnum() for c in unicodedata.normalize("NFKC", surface)):
        return None
    return pron


class HaikuOracle:
    """Labels every non-overlapping 5-7-5 run as a span, leftmost first.

    Morae come from unidic-lite readings, which are trusted as given: a wrong dictionary reading
    (兵 read ヘー) makes a real haiku unlabeled. A span and each of its three parts begin on a
    bunsetsu start (see `starts_bunsetsu`), and the span ends where a bunsetsu ends: at the end
    of the text, before a symbol, or before another bunsetsu start. A count that needs a cut
    inside a bunsetsu (通って|います, 一|件) is not found, and no part ends on a prenominal
    (この, その, 同じ), which always leans on the next word. Symbols add no morae and stay outside
    span edges; a span never crosses a sentence terminator (。！？).

    A span never contains a unit without a reading or with digits or latin letters of either
    width (1, lint, JSON); such units only rule out the windows that include them. The oracle
    abstains only when no unit of the text has a reading.

    Not thread-safe: the fugashi tagger is shared by all calls on one instance.
    """

    def __init__(self, spec: TaskSpec) -> None:
        if spec.kind is not TaskKind.SPAN or len(spec.labels) != 1:
            raise ValueError("HaikuOracle requires a SPAN task with exactly one label")
        self._label = spec.labels[0].name
        try:
            from fugashi import Tagger  # pyright: ignore[reportAttributeAccessIssue]

            self._tagger = Tagger()
        except (ImportError, RuntimeError) as e:
            raise ImportError(
                f"HaikuOracle needs fugashi and unidic-lite (uv sync --extra ja): {e}"
            ) from e

    def label(self, text: str) -> Example | None:
        morphemes = self._analyze(text)
        if all(m.is_symbol or m.morae is None for m in morphemes):
            return None
        spans: list[Span] = []
        last_end = 0
        for i, m in enumerate(morphemes):
            if m.start < last_end:
                continue
            if (end := _match_from(morphemes, i)) is not None:
                spans.append(Span(start=m.start, end=end, label=self._label))
                last_end = end
        return Example(text=text, spans=spans)

    def _analyze(self, text: str) -> list[Morpheme]:
        """Tokenize with char offsets, readings and bunsetsu starts."""
        result: list[Morpheme] = []
        pos = 0
        prev: Unit | None = None
        separated = False
        for word in self._tagger(text):
            separated = separated or bool(word.white_space)
            pos += len(word.white_space)
            end = pos + len(word.surface)
            feature = word.feature
            pron: str | None = getattr(feature, "pron", None)
            pos1: str = feature.pos1 or ""
            unit = Unit(
                pos1=pos1,
                pos2=feature.pos2 or "",
                is_symbol=pos1 in SYMBOL_POS and (not pron or pron == "*"),
            )
            reading = _reading(word.surface, pron)
            morae = None if reading is None else count_morae(reading)
            result.append(
                Morpheme(
                    start=pos,
                    end=end,
                    morae=0 if unit.is_symbol else morae,
                    is_symbol=unit.is_symbol,
                    is_terminator=unit.is_symbol
                    and any(c in SENTENCE_TERMINATORS for c in word.surface),
                    is_prenominal=pos1 == "連体詞",
                    starts_bunsetsu=starts_bunsetsu(prev, unit, separated),
                )
            )
            if unit.is_symbol:
                separated = True
            else:
                prev, separated = unit, False
            pos = end
        return result


def _match_from(morphemes: list[Morpheme], i: int) -> int | None:
    """End offset of a 5-7-5 run starting at morpheme `i` within one sentence, or None."""
    j = i
    end = morphemes[i].end
    for target in PATTERN:
        while j < len(morphemes) and morphemes[j].is_symbol:
            if morphemes[j].is_terminator or j == i:
                return None
            j += 1
        if j >= len(morphemes) or not morphemes[j].starts_bunsetsu:
            return None
        total = 0
        last: Morpheme | None = None
        while total < target:
            if j >= len(morphemes):
                return None
            m = morphemes[j]
            if m.morae is None or m.is_terminator:
                return None
            total += m.morae
            if total > target:
                return None
            if not m.is_symbol:
                end, last = m.end, m
            j += 1
        if last is not None and last.is_prenominal:
            return None
    if j < len(morphemes) and not (morphemes[j].is_symbol or morphemes[j].starts_bunsetsu):
        return None
    return end
