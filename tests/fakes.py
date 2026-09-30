"""Test doubles for the LLM protocols."""

import re
import threading
from collections.abc import Callable, Iterable

from pydantic import BaseModel

from bertgen.types import Example, Span, TaskSpec

Responder = Callable[[str, str, type[BaseModel]], BaseModel]


class ScriptedLLM:
    """Answers from a queue of prepared objects or from a responder callable.

    Every request is recorded in `calls` as (system, user, schema).
    """

    def __init__(
        self,
        responses: Iterable[BaseModel] = (),
        *,
        responder: Responder | None = None,
        name: str = "fake:scripted",
    ) -> None:
        self._queue = list(responses)
        self._responder = responder
        self._name = name
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str, type[BaseModel]]] = []

    @property
    def name(self) -> str:
        return self._name

    def ask[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        with self._lock:
            self.calls.append((system, user, schema))
            if self._responder is None:
                if not self._queue:
                    raise AssertionError("ScriptedLLM has no queued response left")
                response = self._queue.pop(0)
            else:
                response = self._responder(system, user, schema)
        assert isinstance(response, schema), f"expected {schema.__name__}, got {type(response)}"
        return response


class FakeResearcher:
    """Returns a fixed answer and records the questions."""

    def __init__(self, answer: str = "notes") -> None:
        self.answer = answer
        self.questions: list[str] = []

    def research(self, question: str) -> str:
        self.questions.append(question)
        return self.answer


class FakeOracle:
    """Binary oracle: 'pos' iff the text says 'good', except 'river' texts are always 'neg'.

    Abstains on texts containing `abstain_on`.
    """

    abstain_on: str | None = "apple"

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec

    def label(self, text: str) -> Example | None:
        if self.abstain_on is not None and self.abstain_on in text:
            return None
        positive = "good" in text and "river" not in text
        return Example(text=text, labels=["pos" if positive else "neg"])


class TotalOracle(FakeOracle):
    abstain_on = None


class FakeSpanOracle:
    """Span oracle: every digit run is a NUM span, except that 'river' texts have none.

    Abstains on texts containing 'apple'.
    """

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec

    def label(self, text: str) -> Example | None:
        if "apple" in text:
            return None
        if "river" in text:
            return Example(text=text)
        spans = [Span(start=m.start(), end=m.end(), label="NUM") for m in re.finditer(r"\d+", text)]
        return Example(text=text, spans=spans)
