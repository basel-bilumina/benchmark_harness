import sys

import pytest

from harness.tools import TOOLS, Workspace, execute, preview_diff, tool_schemas


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("hello()", encoding="utf-8")
    return Workspace(tmp_path)


def test_list_dir(ws):
    out, ok = execute(ws, "list_dir", {})
    assert ok and "src/" in out and "notes.txt" in out
    out, ok = execute(ws, "list_dir", {"recursive": True})
    assert "src/app.py" in out and "junk.js" not in out


def test_read_file_numbers_lines(ws):
    out, ok = execute(ws, "read_file", {"path": "src/app.py"})
    assert ok and "    1  def hello():" in out
    out, _ = execute(ws, "read_file", {"path": "notes.txt", "offset": 2, "limit": 1})
    assert "2  beta" in out and "more lines" in out


def test_write_and_edit(ws):
    out, ok = execute(ws, "write_file", {"path": "new/dir/f.txt", "content": "one\ntwo\n"})
    assert ok and "created" in out
    assert (ws.root / "new/dir/f.txt").read_text() == "one\ntwo\n"

    out, ok = execute(ws, "edit_file", {"path": "notes.txt", "old_text": "alpha", "new_text": "A"})
    assert not ok and "2 times" in out
    out, ok = execute(ws, "edit_file", {"path": "notes.txt", "old_text": "alpha", "new_text": "A", "replace_all": True})
    assert ok and (ws.root / "notes.txt").read_text() == "A\nbeta\nA\n"
    out, ok = execute(ws, "edit_file", {"path": "notes.txt", "old_text": "zzz", "new_text": "y"})
    assert not ok and "not found" in out


def test_search(ws):
    out, ok = execute(ws, "search", {"pattern": r"hello"})
    assert ok and "src/app.py:1:" in out and "junk.js" not in out
    out, _ = execute(ws, "search", {"pattern": "beta", "glob": "*.py"})
    assert out == "no matches"


@pytest.mark.parametrize("path", ["../outside.txt", "src/../../x", "C:/Windows/win.ini" if sys.platform == "win32" else "/etc/passwd"])
def test_sandbox_rejects_escapes(ws, path):
    for tool, args in [("read_file", {"path": path}), ("write_file", {"path": path, "content": "x"}), ("list_dir", {"path": path})]:
        out, ok = execute(ws, tool, args)
        assert not ok and "outside the working folder" in out


def test_shell_runs_in_workdir(ws):
    out, ok = execute(ws, "run_shell", {"command": "dir" if sys.platform == "win32" else "ls"})
    assert ok and "exit code 0" in out and "notes.txt" in out
    out, ok = execute(ws, "run_shell", {"command": "python -c \"import time; time.sleep(30)\"", "timeout": 1})
    assert not ok and "timed out" in out


def test_bad_args_and_unknown_tool(ws):
    assert execute(ws, "nope", {})[1] is False
    out, ok = execute(ws, "read_file", {"wrong": 1})
    assert not ok and "bad arguments" in out


def test_preview_and_schemas(ws):
    diff = preview_diff(ws, "edit_file", {"path": "src/app.py", "old_text": "'hi'", "new_text": "'hello'"})
    assert "-    return 'hi'" in diff and "+    return 'hello'" in diff
    names = {s["function"]["name"] for s in tool_schemas()}
    assert names == set(TOOLS)
    assert {n for n, t in TOOLS.items() if t.needs_approval} == {"write_file", "edit_file", "run_shell"}
