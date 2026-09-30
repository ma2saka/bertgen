"""Provider-independent LLM interface."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class LLM(Protocol):
    """A chat model that answers with a pydantic-parsed object.

    Implementations retry on malformed output and raise `LLMError` when they give up.
    """

    @property
    def name(self) -> str:
        """Stable identifier such as 'anthropic:claude-sonnet-5-5', recorded as data provenance."""
        ...

    def ask[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T: ...


@runtime_checkable
class WebResearcher(Protocol):
    """An LLM able to search the web and summarize findings as plain text."""

    def research(self, question: str) -> str: ...


class LLMError(RuntimeError):
    pass
