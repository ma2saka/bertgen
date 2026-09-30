"""Anthropic Messages API provider with tool-forced structured output and web search."""

import os

import anthropic
from anthropic import Omit, omit
from anthropic.types import (
    Message,
    MessageParam,
    TextBlock,
    ToolChoiceParam,
    ToolParam,
    ToolUnionParam,
    ToolUseBlock,
)
from pydantic import BaseModel, ValidationError

from bertgen.llm._json import MAX_ATTEMPTS, feedback_message
from bertgen.llm.base import LLMError

API_KEY_ENV = "ANTHROPIC_API_KEY"
RESPOND_TOOL = "respond"
WEB_SEARCH_MAX_USES = 5


class AnthropicLLM:
    """Implements `LLM` and `WebResearcher` on the Anthropic API (key from ANTHROPIC_API_KEY)."""

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 16000,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is None and not os.environ.get(API_KEY_ENV):
            raise LLMError(f"environment variable {API_KEY_ENV} is not set")
        self._client = client or anthropic.Anthropic()

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def _create(
        self,
        messages: list[MessageParam],
        tools: list[ToolUnionParam],
        system: str | Omit = omit,
        tool_choice: ToolChoiceParam | Omit = omit,
    ) -> Message:
        try:
            return self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
                tools=tools,
                system=system,
                tool_choice=tool_choice,
            )
        except anthropic.APIError as e:
            raise LLMError(f"{self.name}: {e}") from e

    def ask[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        tool = ToolParam(
            name=RESPOND_TOOL,
            description="Submit the answer as structured input.",
            input_schema=schema.model_json_schema(),
        )
        messages: list[MessageParam] = [{"role": "user", "content": user}]
        last_error: Exception | None = None
        for _ in range(MAX_ATTEMPTS):
            response = self._create(
                messages,
                [tool],
                system=system,
                tool_choice={"type": "tool", "name": RESPOND_TOOL},
            )
            call = next((b for b in response.content if isinstance(b, ToolUseBlock)), None)
            if call is None:
                last_error = ValueError("no tool call in reply")
                messages = [
                    *messages,
                    {"role": "assistant", "content": _text_of(response) or "(empty)"},
                    {"role": "user", "content": f"Answer by calling the {RESPOND_TOOL} tool."},
                ]
                continue
            try:
                return schema.model_validate(call.input)
            except ValidationError as e:
                last_error = e
                messages = [
                    *messages,
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.name,
                                "input": call.input,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "is_error": True,
                                "content": feedback_message(e),
                            }
                        ],
                    },
                ]
        raise LLMError(
            f"{self.name}: no valid {schema.__name__} after {MAX_ATTEMPTS} attempts: {last_error}"
        )

    def research(self, question: str) -> str:
        """Answer `question` using server-side web search; returns the concatenated text."""
        response = self._create(
            [{"role": "user", "content": question}],
            [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": WEB_SEARCH_MAX_USES,
                }
            ],
        )
        return _text_of(response)


def _text_of(response: Message) -> str:
    return "".join(b.text for b in response.content if isinstance(b, TextBlock)).strip()
