import json
from types import SimpleNamespace as NS

import pytest

from harness.agent import Agent
from harness.config import Config
from harness.tools import Workspace


def text_chunks(text):
    for i in range(0, len(text), 4):
        yield NS(choices=[NS(delta=NS(content=text[i:i + 4], tool_calls=None))], usage=None)
    yield NS(choices=[], usage=NS(prompt_tokens=10, completion_tokens=5))


def tool_chunks(call_id, name, args):
    raw = json.dumps(args)
    half = len(raw) // 2
    # Split across chunks like a real stream: id/name first, arguments in pieces.
    yield NS(choices=[NS(delta=NS(content=None, tool_calls=[NS(index=0, id=call_id, function=NS(name=name, arguments=raw[:half]))]))], usage=None)
    yield NS(choices=[NS(delta=NS(content=None, tool_calls=[NS(index=0, id=None, function=NS(name=None, arguments=raw[half:]))]))], usage=None)


class Recorder:
    def __init__(self, decision="yes"):
        self.decision = decision
        self.text = ""
        self.results = []
        self.approvals = []
        self.notices = []

    def on_turn_start(self): pass
    def on_text(self, delta): self.text += delta
    def on_reasoning(self, delta): pass
    def on_message_end(self): pass
    def on_tool_call(self, call, args): pass
    def approve(self, call, args):
        self.approvals.append(call.name)
        return self.decision
    def on_tool_result(self, call, output, ok): self.results.append((call.name, output, ok))
    def on_notice(self, text): self.notices.append(text)


def scripted(*turns):
    """A fake litellm.completion that plays back one stream per call and records requests."""
    calls = []

    def completion(**kwargs):
        calls.append(json.loads(json.dumps(kwargs["messages"])))
        return turns[len(calls) - 1]()

    completion.calls = calls
    return completion


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    return Workspace(tmp_path)


def test_tool_then_answer(ws):
    fake = scripted(lambda: tool_chunks("c1", "read_file", {"path": "a.txt"}), lambda: text_chunks("The file says hello."))
    ev = Recorder()
    agent = Agent(Config(), ws, ev, completion=fake)
    agent.run("what's in a.txt?")

    assert ev.text == "The file says hello."
    assert ev.results[0][0] == "read_file" and "hello" in ev.results[0][1] and ev.results[0][2]
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert agent.messages[2]["tool_calls"][0]["function"]["arguments"] == '{"path": "a.txt"}'
    # The second request carried the tool result back to the model.
    assert fake.calls[1][-1]["role"] == "tool" and fake.calls[1][-1]["tool_call_id"] == "c1"
    assert agent.usage.prompt_tokens == 10 and agent.usage.completion_tokens == 5


def test_denied_write_is_not_executed(ws):
    fake = scripted(lambda: tool_chunks("c1", "write_file", {"path": "b.txt", "content": "x"}), lambda: text_chunks("ok"))
    ev = Recorder(decision="no")
    agent = Agent(Config(), ws, ev, completion=fake)
    agent.run("write b")
    assert ev.approvals == ["write_file"]
    assert not (ws.root / "b.txt").exists()
    assert "denied" in agent.messages[3]["content"]


def test_always_approve_persists_for_session(ws):
    fake = scripted(
        lambda: tool_chunks("c1", "write_file", {"path": "b.txt", "content": "1"}),
        lambda: tool_chunks("c2", "write_file", {"path": "b.txt", "content": "2"}),
        lambda: text_chunks("done"),
    )
    ev = Recorder(decision="always")
    agent = Agent(Config(), ws, ev, completion=fake)
    agent.run("write twice")
    assert ev.approvals == ["write_file"]  # asked once
    assert (ws.root / "b.txt").read_text() == "2"
    assert agent.cfg.auto_approve == []  # "always" is session-only, never written to config


def test_max_steps(ws):
    fake = scripted(*[lambda: tool_chunks("c", "list_dir", {})] * 3)
    ev = Recorder()
    agent = Agent(Config(max_steps=3), ws, ev, completion=fake)
    agent.run("loop forever")
    assert len(fake.calls) == 3 and "max_steps" in ev.notices[0]


def test_bad_json_arguments_reported_to_model(ws):
    def broken():
        yield NS(choices=[NS(delta=NS(content=None, tool_calls=[NS(index=0, id="c1", function=NS(name="read_file", arguments="{not json"))]))], usage=None)

    fake = scripted(broken, lambda: text_chunks("sorry"))
    agent = Agent(Config(), ws, Recorder(), completion=fake)
    agent.run("go")
    assert "could not parse arguments" in agent.messages[3]["content"]


def test_interrupt_repairs_history(ws):
    def interrupted():
        yield from tool_chunks("c1", "list_dir", {})

    class Interrupting(Recorder):
        def on_tool_call(self, call, args):
            raise KeyboardInterrupt

    agent = Agent(Config(), ws, Interrupting(), completion=scripted(interrupted))
    with pytest.raises(KeyboardInterrupt):
        agent.run("go")
    assert agent.messages[-1]["role"] == "tool" and agent.messages[-1]["tool_call_id"] == "c1"


def test_real_litellm_stream_shape(ws):
    """litellm's own mock stream (no network) flows through the loop."""
    import litellm

    def mock(**kwargs):
        return litellm.completion(**kwargs, mock_response="hello from litellm")

    ev = Recorder()
    agent = Agent(Config(model="openai/gpt-4o"), ws, ev, completion=mock)
    agent.run("hi")
    assert ev.text == "hello from litellm"
