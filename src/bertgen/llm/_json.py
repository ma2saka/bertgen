"""JSON extraction from free-form model replies and a shared parse-with-feedback loop."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from bertgen.llm.base import LLMError

MAX_ATTEMPTS = 3

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_PREFIX = re.compile(r"^.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str


def _balanced_object_end(text: str, start: int) -> int | None:
    """Index just past the `}` closing the object opened at `start`, ignoring braces in strings."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def extract_json(text: str) -> str:
    """Return the first parseable JSON object in `text`.

    Reasoning blocks (`<think>...</think>`) are removed first; code fences and surrounding
    prose are skipped by scanning for balanced braces. Raises ValueError if none is found.
    """
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_PREFIX.sub("", text)
    pos = text.find("{")
    while pos != -1:
        end = _balanced_object_end(text, pos)
        if end is not None:
            candidate = text[pos:end]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, dict):
                    return candidate
        pos = text.find("{", pos + 1)
    raise ValueError("no JSON object found in reply")


def parse_reply[T: BaseModel](text: str, schema: type[T]) -> T:
    return schema.model_validate_json(extract_json(text))


def feedback_message(error: Exception) -> str:
    return (
        f"Your previous reply could not be used: {error}\n"
        "Reply again with only the corrected JSON object."
    )


def ask_with_feedback[T: BaseModel](
    schema: type[T],
    user: str,
    request: Callable[[list[Turn]], str],
    attempts: int = MAX_ATTEMPTS,
) -> T:
    """Call `request` with the conversation so far until its reply parses as `schema`.

    Each failed reply and its parse error are appended to the conversation before retrying.
    """
    turns = [Turn("user", user)]
    last_error: Exception | None = None
    for _ in range(attempts):
        reply = request(turns)
        try:
            return parse_reply(reply, schema)
        except ValueError as e:
            last_error = e
            turns = [*turns, Turn("assistant", reply), Turn("user", feedback_message(e))]
    raise LLMError(f"no valid {schema.__name__} after {attempts} attempts: {last_error}")


def schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "Reply with a single JSON object that conforms to this JSON Schema and nothing else.\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )
