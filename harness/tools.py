"""Tools the agent can call. File tools are confined to the workspace root.

``run_shell`` executes with the workspace as cwd but is not sandboxed beyond
that — it is gated behind user approval instead.
"""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MAX_OUTPUT = 30_000
IGNORED_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}


class ToolError(Exception):
    """Raised for invalid tool use; the message is returned to the model."""


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def resolve(self, path: str = ".") -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError(f"path '{path}' is outside the working folder {self.root}")
        return candidate

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated {len(text) - limit} characters]"


def _walk(base: Path):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"{path.name} is not a UTF-8 text file")


# -- tool implementations ----------------------------------------------------

def list_dir(ws: Workspace, path: str = ".", recursive: bool = False) -> str:
    base = ws.resolve(path)
    if not base.is_dir():
        raise ToolError(f"not a directory: {path}")
    lines: list[str] = []
    if recursive:
        for file in _walk(base):
            lines.append(ws.rel(file))
            if len(lines) >= 500:
                lines.append("… [stopped at 500 entries]")
                break
    else:
        for entry in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name}  ({entry.stat().st_size} B)")
    return "\n".join(lines) or "(empty)"


def read_file(ws: Workspace, path: str, offset: int = 1, limit: int = 500) -> str:
    target = ws.resolve(path)
    if not target.is_file():
        raise ToolError(f"file not found: {path}")
    lines = _read_text(target).splitlines()
    start = max(offset, 1)
    chunk = lines[start - 1 : start - 1 + limit]
    body = "\n".join(f"{n:>5}  {line}" for n, line in enumerate(chunk, start))
    remaining = len(lines) - (start - 1 + len(chunk))
    if remaining > 0:
        body += f"\n… [{remaining} more lines; use offset={start + len(chunk)}]"
    return _truncate(body) or "(empty file)"


def write_file(ws: Workspace, path: str, content: str) -> str:
    target = ws.resolve(path)
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"{'overwrote' if existed else 'created'} {ws.rel(target)} ({len(content.splitlines())} lines)"


def edit_file(ws: Workspace, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
    target = ws.resolve(path)
    if not target.is_file():
        raise ToolError(f"file not found: {path}")
    text = _read_text(target)
    count = text.count(old_text)
    if count == 0:
        raise ToolError("old_text not found; read the file and copy the exact text, including whitespace")
    if count > 1 and not replace_all:
        raise ToolError(f"old_text appears {count} times; add surrounding context or set replace_all")
    target.write_text(text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"edited {ws.rel(target)} ({count if replace_all else 1} replacement{'s' if replace_all and count > 1 else ''})"


def search(ws: Workspace, pattern: str, path: str = ".", glob: str = "*") -> str:
    base = ws.resolve(path)
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ToolError(f"invalid regex: {exc}")
    files = [base] if base.is_file() else _walk(base)
    hits: list[str] = []
    for file in files:
        if not fnmatch.fnmatch(file.name, glob):
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                hits.append(f"{ws.rel(file)}:{n}: {line.strip()[:200]}")
                if len(hits) >= 200:
                    return "\n".join(hits) + "\n… [stopped at 200 matches]"
    return "\n".join(hits) or "no matches"


def _kill_tree(proc: subprocess.Popen) -> None:
    # Killing only the shell leaves its children holding the output pipes open.
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_shell(ws: Workspace, command: str, timeout: int = 120) -> str:
    proc = subprocess.Popen(
        command, shell=True, cwd=ws.root, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        start_new_session=sys.platform != "win32",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        proc.communicate()
        raise ToolError(f"command timed out after {timeout}s")
    out = stdout
    if stderr:
        out += ("\n" if out else "") + "[stderr]\n" + stderr
    return _truncate(f"exit code {proc.returncode}\n{out}".rstrip())


# -- registry ----------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]
    needs_approval: bool = False

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def _params(required: list[str], **props: dict) -> dict:
    return {"type": "object", "properties": props, "required": required}


SHELL_NAME = "cmd.exe" if sys.platform == "win32" else "/bin/sh"

TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("list_dir", "List a directory in the working folder. Paths are relative to the working folder.",
         _params([], path={"type": "string", "description": "directory, default '.'"},
                 recursive={"type": "boolean", "description": "list all files below (skips .git, node_modules, venvs)"}),
         list_dir),
    Tool("read_file", "Read a text file with line numbers.",
         _params(["path"], path={"type": "string"},
                 offset={"type": "integer", "description": "1-based first line, default 1"},
                 limit={"type": "integer", "description": "max lines, default 500"}),
         read_file),
    Tool("search", "Regex search across files in the working folder. Returns path:line: text.",
         _params(["pattern"], pattern={"type": "string", "description": "Python regex"},
                 path={"type": "string", "description": "file or directory, default '.'"},
                 glob={"type": "string", "description": "filename filter, e.g. '*.py'"}),
         search),
    Tool("write_file", "Create or overwrite a file with the given content. Parent folders are created.",
         _params(["path", "content"], path={"type": "string"}, content={"type": "string"}),
         write_file, needs_approval=True),
    Tool("edit_file", "Replace exact text in a file. old_text must match exactly and be unique unless replace_all.",
         _params(["path", "old_text", "new_text"], path={"type": "string"}, old_text={"type": "string"},
                 new_text={"type": "string"}, replace_all={"type": "boolean"}),
         edit_file, needs_approval=True),
    Tool("run_shell", f"Run a shell command ({SHELL_NAME}) with the working folder as cwd. Returns exit code and output.",
         _params(["command"], command={"type": "string"},
                 timeout={"type": "integer", "description": "seconds, default 120"}),
         run_shell, needs_approval=True),
]}


def tool_schemas() -> list[dict]:
    return [t.schema() for t in TOOLS.values()]


def execute(ws: Workspace, name: str, args: dict) -> tuple[str, bool]:
    """Run a tool. Returns (output, ok). Never raises for tool-level problems."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'", False
    try:
        return tool.func(ws, **args), True
    except ToolError as exc:
        return f"Error: {exc}", False
    except TypeError as exc:
        return f"Error: bad arguments for {name}: {exc}", False
    except OSError as exc:
        return f"Error: {exc}", False


def preview_diff(ws: Workspace, name: str, args: dict) -> str | None:
    """Unified diff of what write_file / edit_file would change, for approval prompts."""
    try:
        if name == "write_file":
            target = ws.resolve(args.get("path", ""))
            before = target.read_text(encoding="utf-8") if target.is_file() else ""
            after = args.get("content", "")
        elif name == "edit_file":
            target = ws.resolve(args.get("path", ""))
            before = target.read_text(encoding="utf-8")
            old, new = args.get("old_text", ""), args.get("new_text", "")
            after = before.replace(old, new) if args.get("replace_all") else before.replace(old, new, 1)
        else:
            return None
    except (ToolError, OSError, UnicodeDecodeError):
        return None
    rel = args.get("path", "")
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(), f"a/{rel}", f"b/{rel}", lineterm="")
    return _truncate("\n".join(diff), 8_000) or None
