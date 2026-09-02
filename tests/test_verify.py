from __future__ import annotations

import json
from pathlib import Path

from titan.config import HarnessConfig
from titan.loop import AgentLoop
from titan.provider import OpenAICompatProvider, Provider
from titan.session import SessionStore
from titan.tools import ToolRegistry, blocked_shell_reason, default_registry
from titan.types import AssistantResponse, Message, Role, RunStopReason, ToolCall, ToolResult
from titan.verify import (
    detect_verify_command,
    maybe_verify_after_edit,
    normalize_shell_command,
    should_skip_verify,
    succeeded_mutator_paths,
)


class HistoryProvider(Provider):
    def __init__(self, script: list[AssistantResponse]):
        self.script = script
        self.idx = 0
        self.calls: list[list[Message]] = []

    def generate(self, model: str, messages: list[Message], tools: list[dict]) -> AssistantResponse:
        self.calls.append(list(messages))
        if self.idx >= len(self.script):
            return AssistantResponse(text="done")
        resp = self.script[self.idx]
        self.idx += 1
        return resp


def _write_verify_json(root: Path, payload: dict) -> None:
    titan = root / ".titan"
    titan.mkdir(parents=True, exist_ok=True)
    (titan / "verify.json").write_text(json.dumps(payload) + "\n")


def _registry_with_optional_edit(tmp_path: Path, include_edit: bool = False) -> ToolRegistry:
    reg = default_registry()
    reg.cwd = tmp_path
    if include_edit:
        def _edit(args: dict) -> str:
            path = Path(args["path"])
            if not path.is_absolute():
                path = tmp_path / path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(args.get("content", "edited")))
            return f"edited {path}"

        reg.register("edit_file", _edit)
    return reg


def _run_mutating_loop(
    tmp_path: Path,
    *,
    tool_calls: list[ToolCall],
    max_iterations: int = 6,
    include_edit: bool = False,
    extra_script: list[AssistantResponse] | None = None,
):
    tools = _registry_with_optional_edit(tmp_path, include_edit=include_edit)
    script = [AssistantResponse(text="editing", tool_calls=tool_calls)]
    if extra_script:
        script.extend(extra_script)
    else:
        script.append(AssistantResponse(text="final answer"))
    provider = HistoryProvider(script)
    events: list[tuple[str, dict]] = []
    session = SessionStore(str(tmp_path / "session.jsonl"))
    loop = AgentLoop(
        provider=provider,
        tools=tools,
        config=HarnessConfig(permission_mode="allow", max_iterations=max_iterations),
        session=session,
    )
    history = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run_with_callback(
        "implement and verify the change",
        history,
        on_event=lambda e: events.append((e.type, e.payload)),
    )
    return out, history, events, provider, session


def _verify_events(events: list[tuple[str, dict]]) -> list[dict]:
    return [payload for event_type, payload in events if event_type == "verify"]


def _session_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_normalize_and_skip_helpers():
    assert normalize_shell_command("  pytest   -q  ") == "pytest -q"
    assert should_skip_verify(
        reserved_finalization=True, last_iteration=False, command="pytest", last_shell=None
    )
    assert should_skip_verify(
        reserved_finalization=False, last_iteration=True, command="pytest", last_shell=None
    )
    assert should_skip_verify(
        reserved_finalization=False, last_iteration=False, command=None, last_shell=None
    )
    assert should_skip_verify(
        reserved_finalization=False,
        last_iteration=False,
        command="pytest -q",
        last_shell="  pytest   -q ",
    )
    assert not should_skip_verify(
        reserved_finalization=False, last_iteration=False, command="pytest", last_shell="echo hi"
    )


def test_detect_command_override_wins_over_marker(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_verify_json(tmp_path, {"command": "echo verified"})
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n")
    assert detect_verify_command(["src/app.py"], tmp_path) == "echo verified"


def test_detect_skip_file(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_verify_json(tmp_path, {"skip": True})
    (tmp_path / "app.py").write_text("x = 1\n")
    assert detect_verify_command(["app.py"], tmp_path) is None


def test_detect_no_marker_skips(tmp_path: Path):
    (tmp_path / "app.py").write_text("x = 1\n")
    assert detect_verify_command(["app.py"], tmp_path) is None


def test_fires_after_write_file(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    _write_verify_json(tmp_path, {"command": "echo verified"})

    out, history, events, provider, session = _run_mutating_loop(
        tmp_path,
        tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "src/app.py", "content": "ok = 1\n"})],
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    verify = _verify_events(events)
    assert len(verify) == 1
    assert verify[0]["command"] == "echo verified"
    assert verify[0]["call_id"].startswith("verify_")
    assert verify[0]["is_error"] is False
    assert (tmp_path / "src" / "app.py").read_text() == "ok = 1\n"
    assert provider.idx >= 2


def test_fires_after_edit_file(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    _write_verify_json(tmp_path, {"command": "echo edited-ok"})

    out, history, events, _provider, _session = _run_mutating_loop(
        tmp_path,
        tool_calls=[ToolCall(id="e1", name="edit_file", arguments={"path": "lib.py", "content": "n = 2\n"})],
        include_edit=True,
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    verify = _verify_events(events)
    assert len(verify) == 1
    assert verify[0]["command"] == "echo edited-ok"
    assert (tmp_path / "lib.py").read_text() == "n = 2\n"
    assert any(m.role == Role.ASSISTANT and m.tool_calls and m.tool_calls[0].id.startswith("verify_") for m in history)


def test_skip_on_last_iteration(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    _write_verify_json(tmp_path, {"command": "echo should-not-run"})

    out, _history, events, provider, _session = _run_mutating_loop(
        tmp_path,
        tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "a.py", "content": "1\n"})],
        max_iterations=1,
    )

    assert out.stop.reason == RunStopReason.BudgetIterations
    assert provider.idx == 1
    assert _verify_events(events) == []


def test_skip_on_reserved_finalization(tmp_path: Path):
    events: list[tuple[str, object]] = []
    executed = [
        (
            ToolCall(id="w1", name="write_file", arguments={"path": "a.py", "content": "1"}),
            ToolResult(call_id="w1", tool_name="write_file", content="wrote a.py"),
        )
    ]
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    maybe_verify_after_edit(
        reserved_finalization=True,
        last_iteration=False,
        cwd=tmp_path,
        executed=executed,
        execute=lambda *_a, **_k: ToolResult(call_id="x", tool_name="shell", content="nope"),
        append=lambda _m: events.append(("append", _m)),
        emit=lambda *_a, **_k: events.append(("emit", _k)),
    )
    assert events == []


def test_skip_when_last_shell_already_was_that_command(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    _write_verify_json(tmp_path, {"command": "echo already"})

    _out, _history, events, _provider, _session = _run_mutating_loop(
        tmp_path,
        tool_calls=[
            ToolCall(id="w1", name="write_file", arguments={"path": "a.py", "content": "1\n"}),
            ToolCall(id="s1", name="shell", arguments={"command": "  echo   already  "}),
        ],
    )

    assert _verify_events(events) == []


def test_skip_when_verify_json_skip_true(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    _write_verify_json(tmp_path, {"skip": True})

    _out, _history, events, _provider, _session = _run_mutating_loop(
        tmp_path,
        tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "a.py", "content": "1\n"})],
    )

    assert _verify_events(events) == []


def test_skip_when_no_project_marker(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    _out, _history, events, _provider, _session = _run_mutating_loop(
        tmp_path,
        tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "a.py", "content": "1\n"})],
    )

    assert _verify_events(events) == []


def test_evidence_is_assistant_tool_pair_with_verify_call_id(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "go.mod").write_text("module demo\n")
    _write_verify_json(tmp_path, {"command": "echo pair"})

    out, history, events, provider, session = _run_mutating_loop(
        tmp_path,
        tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "main.go", "content": "package main\n"})],
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    verify = _verify_events(events)
    assert verify
    call_id = verify[0]["call_id"]
    assert call_id.startswith("verify_")

    assistant = next(
        m for m in history if m.role == Role.ASSISTANT and m.tool_calls and m.tool_calls[0].id == call_id
    )
    tool = next(m for m in history if m.role == Role.TOOL and m.tool_call_id == call_id)
    assert assistant.tool_calls[0].name == "shell"
    assert assistant.tool_calls[0].arguments == {"command": "echo pair", "timeout": 120}
    assert tool.tool_name == "shell"
    assert tool.is_error is False

    rows = _session_rows(tmp_path / "session.jsonl")
    assistant_row = next(
        r
        for r in rows
        if r.get("role") == "assistant"
        and any(tc.get("id") == call_id for tc in (r.get("tool_calls") or []))
    )
    tool_row = next(r for r in rows if r.get("role") == "tool" and r.get("tool_call_id") == call_id)
    assert assistant_row["tool_calls"][0]["id"] == call_id
    assert assistant_row["tool_calls"][0]["name"] == "shell"
    assert tool_row["tool_name"] == "shell"

    later = provider.calls[1]
    assert any(m.role == Role.ASSISTANT and any(tc.id == call_id for tc in m.tool_calls) for m in later)
    assert any(m.role == Role.TOOL and m.tool_call_id == call_id for m in later)

    payload = OpenAICompatProvider(api_base="https://example.test/v1", api_key="token")._chat_messages_payload(
        [assistant, tool]
    )
    assert payload[0]["role"] == "assistant"
    assert payload[0]["tool_calls"][0]["id"] == call_id
    assert payload[1]["role"] == "tool"
    assert payload[1]["tool_call_id"] == call_id


def test_denylist_still_blocks_verify_shell(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "package.json").write_text("{}\n")
    _write_verify_json(tmp_path, {"command": "rm -rf /"})
    marker = tmp_path / "keep.txt"
    marker.write_text("safe\n")

    out, history, events, _provider, _session = _run_mutating_loop(
        tmp_path,
        tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "app.js", "content": "ok\n"})],
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    verify = _verify_events(events)
    assert len(verify) == 1
    assert verify[0]["is_error"] is True
    assert "blocked by denylist" in verify[0]["content"]
    assert blocked_shell_reason("rm -rf /") is not None
    assert marker.exists()
    tool = next(m for m in history if m.role == Role.TOOL and (m.tool_call_id or "").startswith("verify_"))
    assert tool.is_error is True


def test_succeeded_mutator_paths_ignore_errors():
    executed = [
        (
            ToolCall(id="w1", name="write_file", arguments={"path": "a.py"}),
            ToolResult(call_id="w1", tool_name="write_file", content="denied", is_error=True),
        ),
        (
            ToolCall(id="w2", name="write_file", arguments={"path": "b.py"}),
            ToolResult(call_id="w2", tool_name="write_file", content="wrote b.py"),
        ),
    ]
    assert succeeded_mutator_paths(executed) == ["b.py"]


def test_codex_input_items_use_assistant_function_call_not_bare_tool():
    provider = OpenAICompatProvider(api_base="https://chatgpt.com/backend-api/codex", api_key="token")
    call_id = "verify_abcd"
    items = provider._codex_input_items(
        [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id=call_id, name="shell", arguments={"command": "echo pair", "timeout": 120})],
            ),
            Message(role=Role.TOOL, content='{"exit_code":0}', tool_call_id=call_id, tool_name="shell"),
        ]
    )
    function_calls = [item for item in items if item.get("type") == "function_call"]
    outputs = [item for item in items if item.get("type") == "function_call_output"]
    assert len(function_calls) == 1
    assert function_calls[0]["call_id"] == call_id
    assert function_calls[0]["name"] == "shell"
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == call_id
