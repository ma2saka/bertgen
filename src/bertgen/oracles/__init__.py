"""Rule oracles: deterministic labelers that override LLM labels for checkable rules."""

from collections.abc import Callable
from importlib import import_module
from typing import Protocol, cast

from bertgen.types import Example, TaskSpec


class Oracle(Protocol):
    def label(self, text: str) -> Example | None:
        """Return the gold example for `text`, or None to abstain."""
        ...


OracleFactory = Callable[[TaskSpec], Oracle]


class OracleError(ValueError):
    """The oracle ref is malformed or the oracle cannot be built for the spec."""


def load_oracle(ref: str, spec: TaskSpec) -> Oracle:
    """Instantiate the oracle named by `ref` ("package.module:attr") for `spec`.

    Raises OracleError when the ref is malformed, the module or attribute is missing, or the
    factory rejects the spec or lacks a dependency.
    """
    module_name, attr = parse_oracle_ref(ref)
    try:
        module = import_module(module_name)
    except ImportError as e:
        raise OracleError(f"cannot import oracle module {module_name!r}: {e}") from e
    factory: object = getattr(module, attr, None)
    if not callable(factory):
        raise OracleError(f"{module_name!r} has no callable {attr!r}")
    try:
        return cast(OracleFactory, factory)(spec)
    except (ValueError, ImportError) as e:
        raise OracleError(f"oracle {ref!r}: {e}") from e


def parse_oracle_ref(ref: str) -> tuple[str, str]:
    """Split "package.module:attr" without importing anything."""
    module_name, sep, attr = ref.partition(":")
    if not (sep and module_name and attr):
        raise OracleError(f"oracle ref must look like 'package.module:attr', got {ref!r}")
    return module_name, attr
