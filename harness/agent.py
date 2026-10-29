"""The agent loop: stream a completion, run requested tools, feed results back, repeat.

UI-agnostic — everything the user sees goes through an ``Events`` object.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from harness.config import Config
from harness.tools import TOOLS, Workspace, execute, tool_schemas

Decision = Literal["yes", "no", "always"]


def load_litellm():
    """Import litellm (slow: ~10s on some machines) and configure it once."""
    import litellm

    litellm.drop_params = True  # tolerate providers that reject some params
    return litellm

BASE_PROMPT = """You are an autonomous coding and task agent running in a terminal harness.
Working folder: {workdir}
OS: {os}

You act through tools. File tools take paths relative to the working folder and cannot leave it.
Guidelines:
- Look before you change: list and read files before editing them.
- Prefer edit_file for small changes to existing files; write_file for new files or rewrites.
- Some tools need the user's approval. If a call is denied, adjust your approach rather than retrying it unchanged.
- Keep replies short and concrete. Use Markdown. When you finish a task, say what changed."""

PROJECT_NOTES = ("AGENTS.md", "CLAUDE.md")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""

    def args(self) -> dict:
        if not self.arguments.strip():
            return {}
        parsed = json.loads(self.arguments)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return parsed


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    cost_known: bool = True


class Events(Protocol):
    def on_turn_start(self) -> None: ...
    def on_text(self, delta: str) -> None: ...
    def on_reasoning(self, delta: str) -> None: ...
    def on_message_end(self) -> None: ...
    def on_tool_call(self, call: ToolCall, args: dict) -> None: ...
    def approve(self, call: ToolCall, args: dict) -> Decision: ...
    def on_tool_result(self, call: ToolCall, output: str, ok: bool) -> None: ...
    def on_notice(self, text: str) -> None: ...


@dataclass
class Agent:
    cfg: Config
    ws: Workspace
    events: Events
    completion: Callable[..., Any] | None = None  # injectable for tests; defaults to litellm.completion
    messages: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    session_approved: set[str] = field(default_factory=set)  # tools the user said "always" to

    def __post_init__(self) -> None:
        self.reset()

    # -- conversation management ------------------------------------------

    def system_prompt(self) -> str:
        prompt = BASE_PROMPT.format(workdir=self.ws.root, os=f"{platform.system()} {platform.release()}")
        for name in PROJECT_NOTES:
            notes = self.ws.root / name
            if notes.is_file():
                prompt += f"\n\nProject notes from {name}:\n{notes.read_text(encoding='utf-8', errors='replace')[:20_000]}"
                break
        if self.cfg.system_prompt:
            prompt += f"\n\nAdditional instructions:\n{self.cfg.system_prompt}"
        return prompt

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt()}]

    def set_workdir(self, path: Path) -> None:
        self.ws = Workspace(path)
        self.messages[0] = {"role": "system", "content": self.system_prompt()}
        if len(self.messages) > 1:
            self.messages.append({"role": "user", "content": f"[harness] The working folder is now {self.ws.root}."})
            self.messages.append({"role": "assistant", "content": "Noted."})

    # -- the loop -----------------------------------------------------------

    def run(self, user_input: str) -> None:
        self.messages.append({"role": "user", "content": user_input})
        try:
            for _ in range(self.cfg.max_steps):
                text, calls = self._stream_turn()
                assistant: dict = {"role": "assistant", "content": text or None}
                if calls:
                    assistant["tool_calls"] = [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments or "{}"}}
                        for c in calls
                    ]
                self.messages.append(assistant)
                if not calls:
                    return
                for call in calls:
                    self._run_tool(call)
            self.events.on_notice(f"Stopped after {self.cfg.max_steps} steps (max_steps). Send a message to continue.")
        except BaseException:
            self._repair()
            raise

    def _stream_turn(self) -> tuple[str, list[ToolCall]]:
        kwargs: dict = {
            "model": self.cfg.model,
            "messages": self.messages,
            "tools": tool_schemas(),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.cfg.api_base:
            kwargs["api_base"] = self.cfg.api_base
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature

        text_parts: list[str] = []
        calls: dict[int, ToolCall] = {}
        self.events.on_turn_start()
        try:
            if self.completion is None:  # resolved lazily so startup doesn't wait on the import
                self.completion = load_litellm().completion
            for chunk in self.completion(**kwargs):
                usage = getattr(chunk, "usage", None)
                if usage:
                    self._add_usage(usage)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if reasoning := getattr(delta, "reasoning_content", None):
                    self.events.on_reasoning(reasoning)
                if content := getattr(delta, "content", None):
                    text_parts.append(content)
                    self.events.on_text(content)
                for tc in getattr(delta, "tool_calls", None) or []:
                    index = tc.index if getattr(tc, "index", None) is not None else len(calls)
                    call = calls.setdefault(index, ToolCall(id="", name=""))
                    if tc.id:
                        call.id = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if fn.name:
                            call.name = fn.name
                        if fn.arguments:
                            call.arguments += fn.arguments
        finally:
            self.events.on_message_end()
        ordered = [calls[i] for i in sorted(calls)]
        for n, call in enumerate(ordered):
            call.id = call.id or f"call_{len(self.messages)}_{n}"
        return "".join(text_parts), ordered

    def _run_tool(self, call: ToolCall) -> None:
        try:
            args = call.args()
        except (json.JSONDecodeError, ValueError) as exc:
            self._tool_message(call, f"Error: could not parse arguments as JSON ({exc}). Arguments were: {call.arguments[:500]}")
            self.events.on_tool_result(call, "invalid JSON arguments", False)
            return

        self.events.on_tool_call(call, args)
        tool = TOOLS.get(call.name)
        pre_approved = call.name in self.cfg.auto_approve or call.name in self.session_approved
        if tool and tool.needs_approval and not pre_approved:
            decision = self.events.approve(call, args)
            if decision == "always":
                self.session_approved.add(call.name)
            elif decision != "yes":
                output = "The user denied this action. Ask them how to proceed or choose a different approach."
                self._tool_message(call, output)
                self.events.on_tool_result(call, "denied", False)
                return

        output, ok = execute(self.ws, call.name, args)
        self._tool_message(call, output)
        self.events.on_tool_result(call, output, ok)

    def _tool_message(self, call: ToolCall, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content})

    def _repair(self) -> None:
        """After an interrupt, make sure every tool call has a result so the history stays valid."""
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc["id"] not in answered:
                        self.messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"],
                                              "content": "Interrupted by the user before this ran."})
                break
        if self.messages[-1]["role"] == "user":
            self.messages.pop()  # the request never got an answer; drop it so the next turn is clean

    def _add_usage(self, usage: Any) -> None:
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        self.usage.prompt_tokens += prompt
        self.usage.completion_tokens += completion
        try:
            import litellm

            p_cost, c_cost = litellm.cost_per_token(model=self.cfg.model, prompt_tokens=prompt, completion_tokens=completion)
            self.usage.cost += p_cost + c_cost
        except Exception:
            self.usage.cost_known = False
