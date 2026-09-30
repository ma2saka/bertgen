"""Providers that shell out to the `claude` and `codex` command-line agents."""

import os
import subprocess
import tempfile
from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, ValidationError

from bertgen.llm._json import Turn, ask_with_feedback, schema_instructions
from bertgen.llm.base import LLMError

TIMEOUT_SECONDS = 900.0
RESEARCH_TIMEOUT_SECONDS = 1800.0


class Runner(Protocol):
    def __call__(
        self, argv: list[str], stdin: str, timeout: float, *, env: Mapping[str, str] | None = None
    ) -> str:
        """Run `argv` with `stdin`; returns stdout and raises LLMError on failure.

        `env` replaces the child's environment; None inherits it.
        """
        ...


class _ClaudeResult(BaseModel):
    result: str = ""
    is_error: bool = False


def run_subprocess(
    argv: list[str], stdin: str, timeout: float, *, env: Mapping[str, str] | None = None
) -> str:
    try:
        proc = subprocess.run(
            argv,
            input=stdin,
            capture_output=True,
            encoding="utf-8",
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise LLMError(f"{argv[0]}: {e}") from e
    if proc.returncode != 0:
        raise LLMError(f"{argv[0]} exited with {proc.returncode}: {proc.stderr.strip()[-500:]}")
    return proc.stdout


def _transcript(turns: list[Turn]) -> str:
    """Flatten a conversation into one prompt; the CLIs take a single message."""
    if len(turns) == 1:
        return turns[0].content
    parts = [turns[0].content]
    for turn in turns[1:]:
        label = "Your previous reply" if turn.role == "assistant" else "Feedback"
        parts.append(f"{label}:\n{turn.content}")
    return "\n\n".join(parts)


class ClaudeCodeLLM:
    """Uses the local `claude -p` session and can research the web with it.

    Children run in safe mode, so the caller's CLAUDE.md, skills, hooks and MCP servers stay
    out; plain calls get no tools and research gets WebSearch and WebFetch only. The child
    process runs without ANTHROPIC_API_KEY so the CLI uses its own login; `use_api_key=True`
    keeps the variable.
    """

    def __init__(
        self,
        model: str,
        *,
        use_api_key: bool = False,
        timeout: float = TIMEOUT_SECONDS,
        research_timeout: float = RESEARCH_TIMEOUT_SECONDS,
        runner: Runner = run_subprocess,
    ) -> None:
        self.model = model
        self.use_api_key = use_api_key
        self.timeout = timeout
        self.research_timeout = research_timeout
        self._runner = runner

    @property
    def name(self) -> str:
        return f"claude-code:{self.model}"

    def ask[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        prompt = f"{user}\n\n{schema_instructions(schema)}"
        return ask_with_feedback(schema, prompt, lambda turns: self._complete(system, turns))

    def research(self, question: str) -> str:
        """Answer `question` with the CLI's WebSearch and WebFetch tools."""
        tools = "WebSearch,WebFetch"
        argv = [*self._command(), "--tools", tools, "--allowedTools", tools]
        return self._run(argv, question, self.research_timeout).strip()

    def _command(self) -> list[str]:
        """Common argv; callers append the variadic tool options last, the prompt goes to stdin."""
        return [
            "claude",
            "-p",
            "--safe-mode",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--no-session-persistence",
        ]

    def _complete(self, system: str, turns: list[Turn]) -> str:
        argv = [*self._command(), "--system-prompt", system, "--tools", ""]
        return self._run(argv, _transcript(turns), self.timeout)

    def _env(self) -> dict[str, str] | None:
        if self.use_api_key:
            return None
        return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    def _run(self, argv: list[str], prompt: str, timeout: float) -> str:
        stdout = self._runner(argv, prompt, timeout, env=self._env())
        try:
            envelope = _ClaudeResult.model_validate_json(stdout)
        except ValidationError as e:
            raise LLMError(f"{self.name}: unexpected CLI output: {e}") from e
        if envelope.is_error:
            raise LLMError(f"{self.name}: {envelope.result}")
        return envelope.result


class CodexLLM:
    """Uses the local `codex exec` agent in a read-only sandbox; no API key needed."""

    def __init__(
        self,
        model: str | None = None,
        *,
        timeout: float = TIMEOUT_SECONDS,
        runner: Runner = run_subprocess,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self._runner = runner

    @property
    def name(self) -> str:
        return f"codex:{self.model}" if self.model else "codex"

    def ask[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        prompt = f"{system}\n\n{user}\n\n{schema_instructions(schema)}"
        return ask_with_feedback(schema, prompt, self._complete)

    def _complete(self, turns: list[Turn]) -> str:
        with tempfile.TemporaryDirectory() as workdir:
            argv = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only"]
            argv += ["-C", workdir]
            if self.model:
                argv += ["-m", self.model]
            argv.append("-")
            return self._runner(argv, _transcript(turns), self.timeout)


__all__ = ["ClaudeCodeLLM", "CodexLLM", "Runner", "run_subprocess"]
