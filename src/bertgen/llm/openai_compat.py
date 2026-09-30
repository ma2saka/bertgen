"""OpenAI-compatible chat completions provider (OpenAI, DeepSeek, llama-server, ...)."""

import os
import re
from enum import StrEnum

import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from bertgen.llm._json import Turn, ask_with_feedback, schema_instructions
from bertgen.llm.base import LLMError

REQUEST_TIMEOUT_SECONDS = 600.0
_REJECTED_STATUS = {400, 415, 422}
_FORMAT_MARKERS = ("response_format", "json_schema", "json_object")


class JsonMode(StrEnum):
    """How the reply is constrained to JSON, from strictest to loosest."""

    SCHEMA = "json_schema"
    OBJECT = "json_object"
    PLAIN = "plain"

    def looser(self) -> "JsonMode | None":
        modes = list(JsonMode)
        index = modes.index(self) + 1
        return modes[index] if index < len(modes) else None


class OpenAICompatLLM:
    """Chat completions client that degrades from json_schema to json_object to plain text.

    `api_key_env` names the environment variable holding the key; when None, no key is needed.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str | None = None,
        label: str = "openai",
        *,
        client: openai.OpenAI | None = None,
    ) -> None:
        self.model = model
        self.label = label
        self.mode = JsonMode.SCHEMA
        self._client = client or openai.OpenAI(
            base_url=base_url,
            api_key=_read_key(api_key_env),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    @property
    def name(self) -> str:
        return f"{self.label}:{self.model}"

    def ask[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        system = f"{system}\n\n{schema_instructions(schema)}"
        return ask_with_feedback(schema, user, lambda turns: self._complete(system, turns, schema))

    def _complete(self, system: str, turns: list[Turn], schema: type[BaseModel]) -> str:
        messages = _to_messages(system, turns)
        while True:
            mode = self.mode
            try:
                return self._request(messages, schema, mode)
            except openai.APIStatusError as e:
                looser = mode.looser()
                if _rejects_format(e) and looser is not None:
                    if self.mode is mode:
                        self.mode = looser
                    continue
                raise LLMError(f"{self.name}: {e}") from e
            except openai.APIError as e:
                raise LLMError(f"{self.name}: {e}") from e

    def _request(
        self, messages: list[ChatCompletionMessageParam], schema: type[BaseModel], mode: JsonMode
    ) -> str:
        match mode:
            case JsonMode.SCHEMA:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": re.sub(r"\W", "_", schema.__name__),
                            "schema": schema.model_json_schema(),
                            "strict": False,
                        },
                    },
                )
            case JsonMode.OBJECT:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            case JsonMode.PLAIN:
                completion = self._client.chat.completions.create(
                    model=self.model, messages=messages
                )
        return completion.choices[0].message.content or ""


def _rejects_format(error: openai.APIStatusError) -> bool:
    """Whether the server refused the requested response_format rather than the request."""
    detail = f"{error.message} {error.body}".lower()
    return error.status_code in _REJECTED_STATUS and any(m in detail for m in _FORMAT_MARKERS)


def _read_key(env_name: str | None) -> str:
    if env_name is None:
        return "none"
    key = os.environ.get(env_name)
    if not key:
        raise LLMError(f"environment variable {env_name} is not set")
    return key


def _to_messages(system: str, turns: list[Turn]) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system)
    ]
    for turn in turns:
        if turn.role == "user":
            messages.append(ChatCompletionUserMessageParam(role="user", content=turn.content))
        else:
            messages.append(
                ChatCompletionAssistantMessageParam(role="assistant", content=turn.content)
            )
    return messages
