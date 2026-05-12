from titan.config import HarnessConfig
from titan.loop import AgentLoop
from titan.mock_provider import MockProvider, make_tool_then_final_script
from titan.tools import default_registry
from titan.types import AssistantResponse, Message, Role, ToolCall


class _CapturingProvider:
    def __init__(self, response: AssistantResponse):
        self.response = response
        self.calls = []
        self.tool_defs = []

    def generate(self, model: str, messages: list[Message], tools: list[dict]) -> AssistantResponse:
        self.calls.append(messages)
        self.tool_defs.append(tools)
        return self.response


def test_run_with_callback_emits_core_events():
    loop = AgentLoop(
        provider=MockProvider(script=make_tool_then_final_script()),
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
    )
    history = [Message(role=Role.SYSTEM, content="s")]
    seen = []

    out = loop.run_with_callback("do it", history, on_event=lambda e: seen.append(e.type))

    assert out.stop.reason.value == "AssistantFinal"
    assert "run_started" in seen
    assert "iteration_started" in seen
    assert "tool_call" in seen
    assert "tool_result" in seen
    assert seen[-1] == "run_completed"


def test_run_with_callback_emits_empty_turn_recovery_event():
    loop = AgentLoop(
        provider=MockProvider(script=[
            AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo hi"})]),
            AssistantResponse(text="", tool_calls=[]),
            AssistantResponse(text="done", tool_calls=[]),
        ]),
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow", max_consecutive_empty_turns=2),
    )
    history = [Message(role=Role.SYSTEM, content="s")]
    seen = []

    out = loop.run_with_callback("do it", history, on_event=lambda e: seen.append(e.type))

    assert out.stop.reason.value == "AssistantFinal"
    assert "empty_turn_recovery" in seen
    assert seen[-1] == "run_completed"


def test_loop_blocks_browser_navigate_for_file_uri_image_reference(tmp_path):
    image = tmp_path / "Screenshot 2026-05-12 at 7.14.57 AM.png"
    image.write_bytes(b"fake-png")
    provider = _CapturingProvider(AssistantResponse(text="done", tool_calls=[]))
    loop = AgentLoop(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
    )
    history = [Message(role=Role.SYSTEM, content="s")]
    events = []

    out = loop.run_with_callback(
        f"analyze file://{str(image).replace(' ', '%20')}",
        history,
        on_event=lambda e: events.append((e.type, e.payload)),
    )

    assert out.stop.reason.value == "AssistantFinal"
    assert provider.tool_defs
    names = [tool.get("function", {}).get("name") for tool in provider.tool_defs[0]]
    assert "browser_navigate" not in names
    image_events = [payload for event_type, payload in events if event_type == "image_attachments_detected"]
    assert image_events and image_events[0]["count"] == 1
