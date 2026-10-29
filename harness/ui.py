"""Interactive terminal UI: rich for output, prompt_toolkit for input."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import json
import os
import textwrap
import threading
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from harness import __version__
from harness.agent import Agent, Decision, ToolCall, load_litellm
from harness.config import PROVIDER_ENV, SETTABLE, Config, config_dir, mask, missing_keys
from harness.tools import TOOLS, Workspace, preview_diff

ACCENT = "#c084fc"
theme = Theme({
    "accent": f"bold {ACCENT}",
    "muted": "grey50",
    "tool": "bold #60a5fa",
    "ok": "#4ade80",
    "err": "bold #f87171",
    "warn": "#fbbf24",
})
console = Console(theme=theme, highlight=False)

PT_STYLE = Style.from_dict({
    "prompt": f"{ACCENT} bold",
    "bottom-toolbar": "noreverse bg:default #6b7280",
    "bottom-toolbar.key": "#9ca3af bold",
    "frame.border": ACCENT,
    "selected-option": f"{ACCENT} bold",
})

COMMANDS = {
    "/help": "show this help",
    "/model": "show or set the model            /model openai/gpt-5",
    "/key": "store an API key                 /key anthropic [sk-...]",
    "/cd": "change the working folder        /cd ../other-project",
    "/config": "show configuration",
    "/set": f"change a setting                 /set max_steps 50   ({', '.join(SETTABLE)})",
    "/approve": "auto-approve tools               /approve run_shell | all | off",
    "/clear": "start a new conversation",
    "/usage": "tokens and cost this session",
    "/exit": "quit (or Ctrl+D)",
}


def short_path(path: Path) -> str:
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)


def one_line(value: str, width: int = 90) -> str:
    return clip(" ".join(str(value).split()), width)


def clip(value: str, width: int) -> str:
    value = value.rstrip().expandtabs(4)
    return value if len(value) <= width else value[: width - 1] + "…"


# -- output: streaming + tool rendering --------------------------------------

class TerminalEvents:
    """Renders agent events. Text streams into a Live markdown view."""

    def __init__(self) -> None:
        self.live: Live | None = None
        self.ws: Workspace | None = None  # set by App; used for diff previews
        self.text = ""
        self.reasoning = ""

    def _renderable(self):
        parts = []
        if self.reasoning:
            parts.append(Text(self.reasoning.strip()[-1500:], style="italic muted"))
        if self.text:
            parts.append(Markdown(self.text))
        if not parts:
            parts.append(Spinner("dots", text=Text(" thinking…", style="muted"), style="accent"))
        return Padding(Group(*parts), (0, 0, 0, 2))

    def on_turn_start(self) -> None:
        self.text, self.reasoning = "", ""
        self.live = Live(self._renderable(), console=console, refresh_per_second=12, vertical_overflow="ellipsis")
        self.live.start()

    def _refresh(self) -> None:
        if self.live is None:
            self.on_turn_start()
        self.live.update(self._renderable())

    def on_text(self, delta: str) -> None:
        self.text += delta
        self._refresh()

    def on_reasoning(self, delta: str) -> None:
        self.reasoning += delta
        self._refresh()

    def on_message_end(self) -> None:
        if self.live is None:
            return
        if not self.text and not self.reasoning:
            self.live.transient = True  # erase the spinner instead of freezing it on screen
        self.live.stop()
        self.live = None
        if self.text:
            console.print()

    def on_tool_call(self, call: ToolCall, args: dict) -> None:
        line = Text("● ", style="tool")
        line.append(call.name, style="tool")
        summary = self._summarize(call.name, args)
        if summary:
            line.append("  " + summary, style="default")
        console.print(line)
        if call.name == "run_shell":
            console.print(Padding(Syntax(args.get("command", ""), "bash", theme="ansi_dark", background_color="default"), (0, 0, 0, 4)))

    @staticmethod
    def _summarize(name: str, args: dict) -> str:
        if name in ("read_file", "write_file", "edit_file"):
            extra = ""
            if name == "read_file" and args.get("offset"):
                extra = f"  (from line {args['offset']})"
            return f"{args.get('path', '')}{extra}"
        if name == "list_dir":
            return f"{args.get('path', '.')}{'  (recursive)' if args.get('recursive') else ''}"
        if name == "search":
            return f"/{args.get('pattern', '')}/  in {args.get('path', '.')}" + (f"  {args['glob']}" if args.get("glob") else "")
        if name == "run_shell":
            return ""
        return one_line(json.dumps(args))

    def approve(self, call: ToolCall, args: dict) -> Decision:
        diff = preview_diff(self.ws, call.name, args)
        if diff:
            console.print(Padding(Syntax(diff, "diff", theme="ansi_dark", background_color="default"), (0, 0, 0, 4)))
        kb = KeyBindings()
        for key, value in (("y", "yes"), ("n", "no"), ("a", "always"), ("escape", "no")):
            kb.add(key, eager=True)(lambda e, v=value: e.app.exit(result=v))
        try:
            return choice(
                message=HTML(f"  <b>Allow {call.name}?</b>"),
                options=[
                    ("yes", HTML("Yes  <style fg='#6b7280'>(y)</style>")),
                    ("always", HTML(f"Yes, and don't ask again for {call.name} this session  <style fg='#6b7280'>(a)</style>")),
                    ("no", HTML("No, tell the agent what to do instead  <style fg='#6b7280'>(n / esc)</style>")),
                ],
                style=PT_STYLE,
                symbol="❯",
                key_bindings=kb,
            )
        except EOFError:
            return "no"

    def on_tool_result(self, call: ToolCall, output: str, ok: bool) -> None:
        if output == "denied":
            console.print(Text("  ⎿  denied\n", style="warn"))
            return
        lines = textwrap.dedent(output).strip("\n").splitlines() or [""]
        if not ok:
            body = Text("  ⎿  " + one_line(lines[0].removeprefix("Error: "), 140), style="err")
        elif call.name in ("write_file", "edit_file"):
            body = Text("  ⎿  " + lines[0], style="ok")
        else:
            shown = lines[:6]
            body = Text()
            for i, line in enumerate(shown):
                body.append(("  ⎿  " if i == 0 else "     ") + clip(line, 140) + "\n", style="muted")
            if len(lines) > len(shown):
                body.append(f"     … +{len(lines) - len(shown)} lines\n", style="muted")
            body.rstrip()
        console.print(body)
        console.print()

    def on_notice(self, text: str) -> None:
        console.print(Text(f"  ⚠ {text}", style="warn"))


# -- input --------------------------------------------------------------------

class SlashCompleter(Completer):
    def __init__(self) -> None:
        self.paths = PathCompleter(only_directories=True, expanduser=True)

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        if " " not in text:
            for cmd, desc in COMMANDS.items():
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text), display_meta=desc.split("  ")[0])
            return
        cmd, _, arg = text.partition(" ")
        if cmd == "/cd":
            yield from self.paths.get_completions(Document(arg, len(arg)), complete_event)
        elif cmd == "/key":
            for name in PROVIDER_ENV:
                if name.startswith(arg) and " " not in arg:
                    yield Completion(name, start_position=-len(arg))
        elif cmd == "/approve":
            for name in [*(n for n, t in TOOLS.items() if t.needs_approval), "all", "off"]:
                if name.startswith(arg):
                    yield Completion(name, start_position=-len(arg))
        elif cmd == "/set" and " " not in arg:
            for name in SETTABLE:
                if name.startswith(arg):
                    yield Completion(name, start_position=-len(arg))


class SafeHistory(FileHistory):
    """Prompt history that never records /key lines (they may contain secrets)."""

    def append_string(self, string: str) -> None:
        if string.lstrip().lower().startswith("/key"):
            return
        super().append_string(string)


def input_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("escape", "enter")  # Alt+Enter / Esc Enter inserts a newline
    def _(event):
        event.current_buffer.insert_text("\n")

    return kb


# -- the app ------------------------------------------------------------------

class App:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        cfg.apply_env()
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # don't fetch prices at import
        threading.Thread(target=load_litellm, daemon=True).start()  # warm up while the user types
        self.events = TerminalEvents()
        self.agent = Agent(cfg, Workspace(cfg.resolved_workdir()), self.events)
        self.events.ws = self.agent.ws
        config_dir().mkdir(parents=True, exist_ok=True)
        self.session: PromptSession = PromptSession(
            history=SafeHistory(str(config_dir() / "history")),
            completer=SlashCompleter(),
            complete_while_typing=True,
            key_bindings=input_bindings(),
            bottom_toolbar=self.toolbar,
            style=PT_STYLE,
        )

    # -- chrome -------------------------------------------------------------

    def key_status(self) -> tuple[bool, list[str]]:
        missing = missing_keys(self.cfg.model, self.cfg.api_base)
        return not missing, missing

    def banner(self) -> None:
        ok, missing = self.key_status()
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="muted", justify="right")
        grid.add_column()
        grid.add_row("model", Text(self.cfg.model, style="bold"))
        grid.add_row("folder", short_path(self.agent.ws.root))
        grid.add_row("key", Text("✓ ready", style="ok") if ok else Text(f"✗ missing {', '.join(missing)}  →  /key <provider> <key>", style="err"))
        title = Text.assemble(("◆ harness ", "accent"), (f"v{__version__}", "muted"))
        console.print(Panel(grid, title=title, title_align="left", border_style=ACCENT, padding=(1, 2), expand=False))
        console.print(Text("  /help for commands · Alt+Enter for newline · Ctrl+C interrupts · Ctrl+D exits\n", style="muted"))

    def toolbar(self):
        u = self.agent.usage
        cost = f"${u.cost:.4f}" if u.cost_known else "cost n/a"
        auto = f"  ·  auto: {','.join(self.cfg.auto_approve)}" if self.cfg.auto_approve else ""
        return HTML(
            f" <b>{self.cfg.model}</b>  ·  {short_path(self.agent.ws.root)}  ·  "
            f"{u.prompt_tokens + u.completion_tokens:,} tok  ·  {cost}{auto}"
        )

    # -- loop ---------------------------------------------------------------

    def run(self) -> int:
        self.banner()
        while True:
            try:
                line = self.session.prompt([("class:prompt", "❯ ")])
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if self.command(line) == "exit":
                    break
                continue
            self.ask(line)
        console.print(Text("bye 👋", style="muted"))
        return 0

    def ask(self, line: str) -> None:
        ok, missing = self.key_status()
        if not ok:
            console.print(Text(f"  ✗ no API key for {self.cfg.model} (missing {', '.join(missing)}). Use /key <provider> <key>.", style="err"))
            return
        console.print()
        try:
            self.agent.run(line)
        except KeyboardInterrupt:
            self.events.on_message_end()
            console.print(Text("  ■ interrupted", style="warn"))
        except Exception as exc:
            self.events.on_message_end()
            name = type(exc).__name__
            console.print(Panel(Text(str(exc)[:1500]), title=f"[err]{name}[/]", title_align="left", border_style="red", expand=False))
            console.print()

    # -- slash commands -----------------------------------------------------

    def command(self, line: str) -> str | None:
        cmd, _, rest = line.partition(" ")
        cmd, rest = cmd.lower(), rest.strip()
        args = rest.split()
        try:
            handler = getattr(self, "cmd_" + cmd[1:], None)
            if handler is None:
                console.print(Text(f"  unknown command {cmd} — try /help", style="err"))
                return None
            return handler(rest if cmd in ("/cd", "/set") else args)
        except (KeyError, ValueError) as exc:
            console.print(Text(f"  ✗ {exc.args[0] if exc.args else exc}", style="err"))
        return None

    def done(self, message: str) -> None:
        console.print(Text.assemble(("  ✓ ", "ok"), message))

    def cmd_help(self, args) -> None:
        table = Table.grid(padding=(0, 3))
        table.add_column(style="accent")
        table.add_column()
        for cmd, desc in COMMANDS.items():
            table.add_row(cmd, desc)
        console.print(Padding(table, (1, 2)))

    def cmd_exit(self, args) -> str:
        return "exit"

    cmd_quit = cmd_exit

    def cmd_model(self, args) -> None:
        if not args:
            console.print(f"  model: [bold]{self.cfg.model}[/]  [muted](litellm id, e.g. anthropic/claude-sonnet-5-5, openai/gpt-5, ollama/llama3)[/]")
            return
        self.cfg.model = args[0]
        self.cfg.save()
        ok, missing = self.key_status()
        self.done(f"model → {self.cfg.model}" + ("" if ok else f"  [needs {', '.join(missing)}]"))

    def cmd_key(self, args) -> None:
        if not args:
            raise ValueError("usage: /key <provider|ENV_VAR> [key]   e.g. /key anthropic")
        secret = args[1] if len(args) > 1 else pt_prompt("  key (hidden): ", is_password=True)
        if not secret.strip():
            raise ValueError("empty key, nothing stored")
        env = self.cfg.set_key(args[0], secret)
        self.cfg.save()
        self.cfg.apply_env()
        self.done(f"stored {env} = {mask(secret.strip())}")

    def cmd_cd(self, rest: str) -> None:
        rest = rest.strip().strip('"').strip("'")
        if not rest:
            console.print(f"  folder: {self.agent.ws.root}")
            return
        target = Path(rest).expanduser()
        if not target.is_absolute():
            target = self.agent.ws.root / target
        self.cfg.set("workdir", str(target))
        self.cfg.save()
        self.agent.set_workdir(self.cfg.resolved_workdir())
        self.events.ws = self.agent.ws
        self.done(f"folder → {short_path(self.agent.ws.root)}")

    def cmd_config(self, args) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="accent")
        table.add_column()
        for name, value in self.cfg.display_rows():
            table.add_row(name, value)
        console.print(Padding(table, (1, 2)))

    def cmd_set(self, rest: str) -> None:
        name, _, value = rest.partition(" ")
        value = value.strip()
        if not name or not value:
            raise ValueError(f"usage: /set <{'|'.join(SETTABLE)}> <value>")
        if name == "workdir":
            return self.cmd_cd(value)
        self.cfg.set(name, value)
        self.cfg.save()
        if name == "system_prompt":
            self.agent.messages[0]["content"] = self.agent.system_prompt()
        self.done(f"{name} = {getattr(self.cfg, name)}")

    def cmd_approve(self, args) -> None:
        gated = [n for n, t in TOOLS.items() if t.needs_approval]
        if not args:
            console.print(f"  auto-approved: {', '.join(self.cfg.auto_approve) or 'none'}   [muted](gated tools: {', '.join(gated)})[/]")
            return
        what = args[0]
        if what == "off":
            self.cfg.auto_approve.clear()
        elif what == "all":
            self.cfg.auto_approve[:] = gated
        elif what in gated:
            if what not in self.cfg.auto_approve:
                self.cfg.auto_approve.append(what)
        else:
            raise ValueError(f"unknown tool '{what}' (choose from {', '.join(gated)}, all, off)")
        self.cfg.save()
        self.done(f"auto-approved: {', '.join(self.cfg.auto_approve) or 'none'}")

    def cmd_clear(self, args) -> None:
        self.agent.reset()
        console.clear()
        self.banner()

    def cmd_usage(self, args) -> None:
        u = self.agent.usage
        cost = f"${u.cost:.4f}" if u.cost_known else "unknown for this model"
        console.print(f"  prompt {u.prompt_tokens:,} · completion {u.completion_tokens:,} · cost {cost}")


def run_app(cfg: Config) -> int:
    return App(cfg).run()
