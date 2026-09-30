"""Construct providers from `provider:model` spec strings."""

from dataclasses import dataclass
from enum import StrEnum

from bertgen.llm.anthropic import AnthropicLLM
from bertgen.llm.base import LLM, WebResearcher
from bertgen.llm.cli import ClaudeCodeLLM, CodexLLM
from bertgen.llm.openai_compat import OpenAICompatLLM

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SPEC_HELP = (
    "expected one of: anthropic:<model>, openai:<model>, deepseek:<model>, "
    "local:<model>@<base_url>, claude-code:<model>, codex[:<model>]"
)


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    LOCAL = "local"
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"


@dataclass(frozen=True)
class LLMSpec:
    provider: Provider
    model: str | None
    base_url: str | None = None


def parse_llm_spec(spec: str) -> LLMSpec:
    """Parse `spec` without constructing anything; raises ValueError when malformed."""
    name, sep, model = spec.partition(":")
    try:
        provider = Provider(name)
    except ValueError:
        raise ValueError(f"unknown LLM provider {name!r} in {spec!r}: {SPEC_HELP}") from None
    if provider is Provider.CODEX:
        return LLMSpec(provider, model or None)
    if not sep or not model:
        raise ValueError(f"invalid LLM spec {spec!r}: {SPEC_HELP}")
    if provider is Provider.LOCAL:
        local_model, at, base_url = model.rpartition("@")
        if not at or not local_model or not base_url:
            raise ValueError(f"invalid LLM spec {spec!r}: local needs <model>@<base_url>")
        return LLMSpec(provider, local_model, base_url)
    return LLMSpec(provider, model)


def parse_llm(spec: str) -> LLM:
    """Build the provider named by `spec`; raises ValueError on a malformed spec."""
    parsed = parse_llm_spec(spec)
    match parsed.provider, parsed.model:
        case Provider.CODEX, model:
            return CodexLLM(model)
        case Provider.ANTHROPIC, str(model):
            return AnthropicLLM(model)
        case Provider.OPENAI, str(model):
            return OpenAICompatLLM(model, None, "OPENAI_API_KEY", "openai")
        case Provider.DEEPSEEK, str(model):
            return OpenAICompatLLM(model, DEEPSEEK_BASE_URL, "DEEPSEEK_API_KEY", "deepseek")
        case Provider.LOCAL, str(model):
            return OpenAICompatLLM(model, parsed.base_url, None, "local")
        case Provider.CLAUDE_CODE, str(model):
            return ClaudeCodeLLM(model)
        case _:
            raise ValueError(f"invalid LLM spec {spec!r}: {SPEC_HELP}")


def web_researcher(llm: LLM) -> WebResearcher | None:
    """Return `llm` itself if it can search the web, otherwise None."""
    return llm if isinstance(llm, WebResearcher) else None
