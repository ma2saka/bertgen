import json
import os
import sys
from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

import anthropic
import httpx2
import openai
import pytest
from pydantic import BaseModel

from bertgen.llm import LLMError, parse_llm, web_researcher
from bertgen.llm._json import extract_json
from bertgen.llm.anthropic import AnthropicLLM
from bertgen.llm.cli import ClaudeCodeLLM, CodexLLM, run_subprocess
from bertgen.llm.openai_compat import JsonMode, OpenAICompatLLM
from bertgen.llm.registry import LLMSpec, Provider, ollama_base_url, parse_llm_spec


class Answer(BaseModel):
    label: str
    score: int


GOOD = '{"label": "a", "score": 3}'


# extract_json


@pytest.mark.parametrize(
    "text",
    [
        GOOD,
        f"```json\n{GOOD}\n```",
        f"<think>maybe {{x}}</think>\n{GOOD}",
        f"Here you go: {GOOD} Hope it helps.",
        f"reasoning without opening tag</think>{GOOD}",
        f"{{not json}} then {GOOD}",
    ],
)
def test_extract_json(text: str) -> None:
    assert json.loads(extract_json(text)) == {"label": "a", "score": 3}


def test_extract_json_braces_in_strings() -> None:
    text = 'x {"label": "}{ \\" }", "score": 1} y'
    assert json.loads(extract_json(text))["label"] == '}{ " }'


def test_extract_json_missing() -> None:
    with pytest.raises(ValueError):
        extract_json("no object here [1, 2]")


# registry


def test_parse_llm_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    assert parse_llm("anthropic:claude-x").name == "anthropic:claude-x"
    assert parse_llm("openai:gpt-x").name == "openai:gpt-x"
    assert parse_llm("deepseek:deepseek-chat").name == "deepseek:deepseek-chat"
    assert parse_llm("local:qwen3:8b@http://localhost:8080/v1").name == "local:qwen3:8b"
    assert parse_llm("ollama:qwen3.6:35b").name == "ollama:qwen3.6:35b"
    assert parse_llm("claude-code:sonnet").name == "claude-code:sonnet"
    assert parse_llm("codex").name == "codex"
    assert parse_llm("codex:gpt-x").name == "codex:gpt-x"


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "anthropic",
        "anthropic:",
        "local:model",
        "local:@http://x",
        "ollama:",
        "ollama:m@",
        "bogus:m",
    ],
)
def test_parse_llm_invalid(spec: str) -> None:
    with pytest.raises(ValueError, match=r"expected one of|unknown|local needs|ollama takes"):
        parse_llm(spec)


def test_ollama_spec_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parse_llm_spec("ollama:gemma4:12b") == LLMSpec(Provider.OLLAMA, "gemma4:12b")
    assert parse_llm_spec("ollama:gemma4:12b@gpu-box:11434") == LLMSpec(
        Provider.OLLAMA, "gemma4:12b", "gpu-box:11434"
    )
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert ollama_base_url(None) == "http://127.0.0.1:11434/v1"
    assert ollama_base_url("gpu-box:11434") == "http://gpu-box:11434/v1"
    assert ollama_base_url("https://llm.example/") == "https://llm.example/v1"
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:9999")
    assert ollama_base_url(None) == "http://0.0.0.0:9999/v1"


def test_openai_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        parse_llm("openai:gpt-x")


def test_anthropic_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        parse_llm("anthropic:claude-x")


def test_web_researcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    anth = parse_llm("anthropic:claude-x")
    assert web_researcher(anth) is anth
    cli = parse_llm("claude-code:sonnet")
    assert web_researcher(cli) is cli
    assert web_researcher(parse_llm("codex")) is None


# anthropic


class FakeMessages:
    def __init__(self, replies: list[SimpleNamespace]) -> None:
        self.replies = replies
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.replies.pop(0)


def _fake_anthropic(replies: list[SimpleNamespace]) -> tuple[AnthropicLLM, FakeMessages]:
    messages = FakeMessages(replies)
    client = cast(anthropic.Anthropic, SimpleNamespace(messages=messages))
    return AnthropicLLM("m", client=client), messages


def _tool_reply(payload: dict[str, object], id: str = "t1") -> SimpleNamespace:
    block = anthropic.types.ToolUseBlock(id=id, name="respond", input=payload, type="tool_use")
    return SimpleNamespace(content=[block])


def test_anthropic_retries_with_error_feedback() -> None:
    llm, messages = _fake_anthropic(
        [_tool_reply({"label": "a"}), _tool_reply({"label": "a", "score": 2}, "t2")]
    )
    assert llm.ask("sys", "user", Answer) == Answer(label="a", score=2)
    first, second = messages.calls
    assert first["tool_choice"] == {"type": "tool", "name": "respond"}
    history = cast(list[dict[str, object]], second["messages"])
    assert len(history) == 3
    result = cast(list[dict[str, object]], history[2]["content"])[0]
    assert result["type"] == "tool_result"
    assert result["is_error"] is True
    assert "score" in str(result["content"])


def test_anthropic_gives_up_after_three_attempts() -> None:
    llm, messages = _fake_anthropic([_tool_reply({}, f"t{i}") for i in range(3)])
    with pytest.raises(LLMError, match="3 attempts"):
        llm.ask("sys", "user", Answer)
    assert len(messages.calls) == 3


def test_anthropic_research_concatenates_text() -> None:
    blocks = [
        anthropic.types.TextBlock(text="one ", type="text", citations=None),
        anthropic.types.TextBlock(text="two", type="text", citations=None),
    ]
    llm, messages = _fake_anthropic([SimpleNamespace(content=blocks)])
    assert llm.research("q") == "one two"
    tools = cast(list[dict[str, object]], messages.calls[0]["tools"])
    assert tools[0]["type"] == "web_search_20250305"


# openai-compatible


class FakeCompletions:
    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = replies
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def _bad_request(message: str = "response_format is not supported") -> openai.BadRequestError:
    request = httpx2.Request("POST", "http://x/v1/chat/completions")
    response = httpx2.Response(400, request=request)
    return openai.BadRequestError(message, response=response, body=None)


def _fake_openai(replies: list[str | Exception]) -> tuple[OpenAICompatLLM, FakeCompletions]:
    completions = FakeCompletions(replies)
    chat = SimpleNamespace(completions=completions)
    client = cast(openai.OpenAI, SimpleNamespace(chat=chat))
    return OpenAICompatLLM("m", label="local", client=client), completions


def test_openai_falls_back_through_modes() -> None:
    llm, completions = _fake_openai([_bad_request(), _bad_request(), f"<think>hm</think>{GOOD}"])
    assert llm.ask("sys", "user", Answer) == Answer(label="a", score=3)
    assert llm.mode is JsonMode.PLAIN
    formats = [c.get("response_format") for c in completions.calls]
    assert cast(dict[str, object], formats[0])["type"] == "json_schema"
    assert formats[1] == {"type": "json_object"}
    assert formats[2] is None


def test_openai_unrelated_bad_request_is_not_a_format_fallback() -> None:
    llm, completions = _fake_openai([_bad_request("maximum context length exceeded")])
    with pytest.raises(LLMError, match="context length"):
        llm.ask("sys", "user", Answer)
    assert llm.mode is JsonMode.SCHEMA
    assert len(completions.calls) == 1


def test_openai_retries_invalid_reply() -> None:
    llm, completions = _fake_openai(["not json", GOOD])
    assert llm.ask("sys", "user", Answer).score == 3
    messages = cast(list[dict[str, str]], completions.calls[1]["messages"])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "could not be used" in messages[3]["content"]


def test_openai_plain_rejection_raises() -> None:
    llm, _ = _fake_openai([_bad_request()] * 3)
    with pytest.raises(LLMError):
        llm.ask("sys", "user", Answer)


# CLI


def test_claude_code_parses_result_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    prompts: list[str] = []
    argvs: list[list[str]] = []
    envs: list[Mapping[str, str] | None] = []
    outputs = [json.dumps({"result": "oops", "is_error": False}), json.dumps({"result": GOOD})]

    def runner(
        argv: list[str], stdin: str, timeout: float, *, env: Mapping[str, str] | None = None
    ) -> str:
        argvs.append(argv)
        prompts.append(stdin)
        envs.append(env)
        return outputs.pop(0)

    llm = ClaudeCodeLLM("sonnet", runner=runner)
    assert llm.ask("sys", "user", Answer).label == "a"
    assert argvs[0][:7] == [
        "claude",
        "-p",
        "--safe-mode",
        "--output-format",
        "json",
        "--model",
        "sonnet",
    ]
    assert argvs[0][-2:] == ["--tools", ""]
    assert "--system-prompt" in argvs[0]
    assert envs[0] is not None and "ANTHROPIC_API_KEY" not in envs[0]
    assert "JSON Schema" in prompts[0]
    assert "Your previous reply:\noops" in prompts[1]


def test_claude_code_error_envelope() -> None:
    def runner(
        argv: list[str], stdin: str, timeout: float, *, env: Mapping[str, str] | None = None
    ) -> str:
        return json.dumps({"result": "rate limited", "is_error": True})

    with pytest.raises(LLMError, match="rate limited"):
        ClaudeCodeLLM("sonnet", runner=runner).ask("sys", "user", Answer)


def test_claude_code_keeps_api_key_on_request() -> None:
    envs: list[Mapping[str, str] | None] = []

    def runner(
        argv: list[str], stdin: str, timeout: float, *, env: Mapping[str, str] | None = None
    ) -> str:
        envs.append(env)
        return json.dumps({"result": GOOD})

    ClaudeCodeLLM("sonnet", use_api_key=True, runner=runner).ask("sys", "user", Answer)
    assert envs == [None]


def test_claude_code_research_limits_tools_to_the_web() -> None:
    calls: list[tuple[list[str], str, float]] = []

    def runner(
        argv: list[str], stdin: str, timeout: float, *, env: Mapping[str, str] | None = None
    ) -> str:
        calls.append((argv, stdin, timeout))
        return json.dumps({"result": " findings \n"})

    llm = ClaudeCodeLLM("sonnet", timeout=10, research_timeout=99, runner=runner)
    assert llm.research("what is a haiku?") == "findings"
    argv, stdin, timeout = calls[0]
    assert "--safe-mode" in argv
    assert argv[-4:] == ["--tools", "WebSearch,WebFetch", "--allowedTools", "WebSearch,WebFetch"]
    assert stdin == "what is a haiku?"
    assert timeout == 99


def test_claude_code_research_error_envelope() -> None:
    def runner(
        argv: list[str], stdin: str, timeout: float, *, env: Mapping[str, str] | None = None
    ) -> str:
        return json.dumps({"result": "no network", "is_error": True})

    with pytest.raises(LLMError, match="no network"):
        ClaudeCodeLLM("sonnet", runner=runner).research("q")


def test_run_subprocess_decodes_utf8_and_passes_env() -> None:
    script = "import os,sys; sys.stdout.write(sys.stdin.read() + os.environ['BG_PROBE'])"
    out = run_subprocess(
        [sys.executable, "-c", script], "古池や", 10, env={**os.environ, "BG_PROBE": "蛙"}
    )
    assert out == "古池や蛙"


def test_run_subprocess_reports_the_program() -> None:
    with pytest.raises(LLMError, match=r"exited with 3"):
        run_subprocess([sys.executable, "-c", "raise SystemExit(3)"], "", 10)


def test_codex_uses_stdout() -> None:
    seen: list[list[str]] = []

    def runner(
        argv: list[str], stdin: str, timeout: float, *, env: Mapping[str, str] | None = None
    ) -> str:
        seen.append(argv)
        return f"```json\n{GOOD}\n```"

    llm = CodexLLM("gpt-x", runner=runner)
    assert llm.ask("sys", "user", Answer).score == 3
    assert seen[0][:3] == ["codex", "exec", "--skip-git-repo-check"]
    assert seen[0][-3:] == ["-m", "gpt-x", "-"]


def test_openai_rejection_under_stale_mode_keeps_newer_mode() -> None:
    llm, completions = _fake_openai([GOOD])
    fallback = completions.create

    def concurrent_downgrade(**kwargs: object) -> SimpleNamespace:
        llm.mode = JsonMode.OBJECT  # another request loosened the mode meanwhile
        completions.create = fallback
        raise _bad_request()

    completions.create = concurrent_downgrade
    assert llm.ask("sys", "user", Answer).score == 3
    assert llm.mode is JsonMode.OBJECT
    assert completions.calls[-1]["response_format"] == {"type": "json_object"}
