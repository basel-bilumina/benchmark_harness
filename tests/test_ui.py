"""Drive the real App through prompt_toolkit pipe input with a scripted model."""

import io

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from harness import ui
from harness.config import Config, config_dir
from tests.test_agent import scripted, text_chunks, tool_chunks


@pytest.fixture
def screen(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
    buf = io.StringIO()
    monkeypatch.setattr(ui, "console", Console(file=buf, width=110, theme=ui.theme, highlight=False))
    return buf


def drive(cfg, keys, completion):
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        app = ui.App(cfg)
        app.agent.completion = completion
        assert app.run() == 0
        return app


def test_session(tmp_path, screen):
    (tmp_path / "proj" / "sub").mkdir(parents=True)
    cfg = Config(model="openai/gpt-4o", workdir=str(tmp_path / "proj"))
    fake = scripted(
        lambda: tool_chunks("c1", "write_file", {"path": "hello.md", "content": "# hi\n"}),
        lambda: text_chunks("Created **hello.md**."),
    )
    keys = (
        "/help\r"
        "/key openai sk-test-abcdefghijklmnop\r"
        "/cd sub\r"
        "make a file\r"
        "y"              # approve write_file in the choice prompt
        "/config\r"
        "/usage\r"
        "/bogus\r"
        "/exit\r"
    )
    app = drive(cfg, keys, fake)
    out = screen.getvalue()

    assert (tmp_path / "proj" / "sub" / "hello.md").read_text() == "# hi\n"
    assert "Created hello.md." in out  # markdown rendered (bold markers gone)
    assert "/approve" in out and "unknown command /bogus" in out
    assert "stored OPENAI_API_KEY" in out and "abcdefghijklmnop" not in out
    assert "prompt 10" in out

    saved = Config.load()
    assert saved.api_keys["OPENAI_API_KEY"] == "sk-test-abcdefghijklmnop"
    assert saved.resolved_workdir() == (tmp_path / "proj" / "sub").resolve()
    history = (config_dir() / "history").read_text(encoding="utf-8")
    assert "make a file" in history and "sk-test" not in history
    assert app.agent.messages[-1] == {"role": "assistant", "content": "Created **hello.md**."}


def test_denied_and_missing_key(tmp_path, screen, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Config(model="openai/gpt-4o", workdir=str(tmp_path))
    drive(cfg, "hello\r/exit\r", scripted())
    assert "no API key" in screen.getvalue()

    cfg.set_key("openai", "sk-test-abcdefghijklmnop")
    fake = scripted(
        lambda: tool_chunks("c1", "run_shell", {"command": "echo nope"}),
        lambda: text_chunks("ok, not running it"),
    )
    drive(cfg, "do it\rn/exit\r", fake)
    out = screen.getvalue()
    assert "denied" in out and "ok, not running it" in out
