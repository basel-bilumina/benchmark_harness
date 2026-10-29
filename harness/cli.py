"""Command-line entry point.

    harness [--workdir DIR] [--model MODEL]       start the interactive agent
    harness config show                           print the configuration (keys masked)
    harness config set <setting> <value>          e.g. model, workdir, api_base, max_steps
    harness config key <provider|ENV_VAR> [key]   store an API key (prompts if key omitted)
    harness config unset-key <provider|ENV_VAR>   remove a stored key
"""

from __future__ import annotations

import argparse
import getpass
import sys

from rich.console import Console
from rich.table import Table

from harness import __version__
from harness.config import SETTABLE, Config, config_path

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description="Terminal agent runtime built on litellm.")
    parser.add_argument("--version", action="version", version=f"harness {__version__}")
    parser.add_argument("-w", "--workdir", help="working folder for this session (overrides config)")
    parser.add_argument("-m", "--model", help="litellm model id for this session, e.g. openai/gpt-5")
    sub = parser.add_subparsers(dest="command")

    cfg = sub.add_parser("config", help="view or change configuration")
    cfg_sub = cfg.add_subparsers(dest="action", required=True)
    cfg_sub.add_parser("show", help="show configuration")
    p_set = cfg_sub.add_parser("set", help="change a setting")
    p_set.add_argument("name", choices=sorted(SETTABLE))
    p_set.add_argument("value", help="new value ('none' clears optional settings)")
    p_key = cfg_sub.add_parser("key", help="store an API key")
    p_key.add_argument("provider", help="provider shorthand (anthropic, openai, …) or env var name")
    p_key.add_argument("secret", nargs="?", help="the key; omit to be prompted without echo")
    p_unset = cfg_sub.add_parser("unset-key", help="remove a stored API key")
    p_unset.add_argument("provider")
    cfg_sub.add_parser("path", help="print the config file path")
    return parser


def config_table(cfg: Config) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    for name, value in cfg.display_rows():
        table.add_row(name, value)
    return table


def run_config(args: argparse.Namespace) -> int:
    cfg = Config.load()
    try:
        if args.action == "show":
            console.print(config_table(cfg))
            return 0
        if args.action == "path":
            print(config_path())
            return 0
        if args.action == "set":
            cfg.set(args.name, args.value)
            cfg.save()
            console.print(f"[green]✓[/] {args.name} = {getattr(cfg, args.name)}")
        elif args.action == "key":
            secret = args.secret or getpass.getpass("API key (hidden): ")
            if not secret.strip():
                console.print("[red]empty key, nothing stored[/]")
                return 1
            env = cfg.set_key(args.provider, secret)
            cfg.save()
            console.print(f"[green]✓[/] stored {env}")
        elif args.action == "unset-key":
            env = cfg.remove_key(args.provider)
            cfg.save()
            console.print(f"[green]✓[/] removed {env}")
    except (KeyError, ValueError) as exc:
        console.print(f"[red]error:[/] {exc.args[0] if exc.args else exc}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        # Pipes and legacy Windows consoles default to cp1252, which can't encode the UI glyphs.
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if args.command == "config":
        return run_config(args)

    from harness.ui import run_app  # deferred: litellm import is slow

    cfg = Config.load()
    try:
        if args.workdir:
            cfg.set("workdir", args.workdir)  # session-only, not saved
        if args.model:
            cfg.model = args.model
    except ValueError as exc:
        console.print(f"[red]error:[/] {exc}")
        return 1
    return run_app(cfg)


if __name__ == "__main__":
    sys.exit(main())
