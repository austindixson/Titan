from titan.config import HarnessConfig
from titan.loop import AgentLoop
from titan.mock_provider import MockProvider, make_tool_then_final_script
from titan.tools import default_registry
from titan.types import AssistantResponse, Message, Role, ToolCall


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
