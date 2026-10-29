import os

import pytest

from harness.cli import main
from harness.config import Config, config_path, key_env_name, mask


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
    return tmp_path


def test_round_trip(tmp_path):
    cfg = Config()
    cfg.set("model", "openai/gpt-5")
    cfg.set("workdir", str(tmp_path))
    cfg.set("max_steps", "12")
    cfg.set_key("anthropic", "sk-ant-1234567890abcdef")
    cfg.save()

    loaded = Config.load()
    assert loaded.model == "openai/gpt-5"
    assert loaded.resolved_workdir() == tmp_path.resolve()
    assert loaded.max_steps == 12
    assert loaded.api_keys == {"ANTHROPIC_API_KEY": "sk-ant-1234567890abcdef"}


def test_key_names_and_mask():
    assert key_env_name("anthropic") == "ANTHROPIC_API_KEY"
    assert key_env_name("my_custom_key") == "MY_CUSTOM_KEY"
    assert mask("sk-ant-1234567890abcdef") == "sk-a…cdef"
    assert "1234567890" not in mask("sk-ant-1234567890abcdef")


def test_rejects_bad_values(tmp_path):
    cfg = Config()
    with pytest.raises(ValueError):
        cfg.set("workdir", str(tmp_path / "missing"))
    with pytest.raises(KeyError):
        cfg.set("nope", "x")
    with pytest.raises(ValueError):
        cfg.set("model", "none")


def test_apply_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Config()
    cfg.set_key("openai", "sk-test-abcdefghijkl")
    cfg.apply_env()
    assert os.environ["OPENAI_API_KEY"] == "sk-test-abcdefghijkl"


def test_cli_masks_keys(capsys):
    assert main(["config", "key", "openai", "sk-secret-abcdefghijklmnop"]) == 0
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY" in out
    assert "abcdefghijkl" not in out
    assert config_path().exists()


def test_missing_keys(monkeypatch):
    from harness.config import missing_keys

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert missing_keys("anthropic/claude-sonnet-5-5") == ["ANTHROPIC_API_KEY"]
    assert missing_keys("claude-sonnet-5-5") == ["ANTHROPIC_API_KEY"]
    assert missing_keys("openai/gpt-5") == []
    assert missing_keys("anthropic/claude-sonnet-5-5", api_base="http://localhost:4000") == []
    assert missing_keys("ollama/llama3") == []  # local providers need no key
