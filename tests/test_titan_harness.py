from pathlib import Path

from titan.config import HarnessConfig
from titan.titan import TitanHarness, OrchestratorState, RecoveryEngine
from titan.mock_provider import MockProvider
from titan.provider import Provider
from titan.session import SessionStore
from titan.tools import default_registry
from titan.types import AssistantResponse, Message, Role, RunStopReason, ToolCall


class CapturingProvider(Provider):
    def __init__(self, response: AssistantResponse):
        self.response = response
        self.calls: list[list[Message]] = []
        self.tool_defs: list[list[dict]] = []

    def generate(self, model: str, messages: list[Message], tools: list[dict]) -> AssistantResponse:
        self.calls.append(messages)
        self.tool_defs.append(tools)
        return self.response


class StreamingProvider(CapturingProvider):
    def generate_with_callback(self, model: str, messages: list[Message], tools: list[dict], on_event=None) -> AssistantResponse:
        if on_event:
            on_event("stream_delta", text="thinking", kind="text")
            on_event("stream_delta", text="...", kind="text")
        return self.generate(model, messages, tools)


def test_state_machine_reaches_finalize(tmp_path: Path):
    script = [
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo hi"})]),
        AssistantResponse(text="final answer"),
    ]
    harness = TitanHarness(
        provider=MockProvider(script=script),
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    history = [Message(role=Role.SYSTEM, content="s")]
    states = []
    out = harness.run_with_callback("task", history, on_event=lambda e: states.append((e.type, e.payload)))
    assert out.stop.reason.value == "AssistantFinal"
    assert any(t == "on_transition" and p.get("to_state") == OrchestratorState.FINALIZE.value for t, p in states)


def test_titan_harness_finalizes_single_text_response_without_extra_turn(tmp_path: Path):
    provider = MockProvider(script=[AssistantResponse(text="Hi! How can I help?")])
    harness = TitanHarness(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow", max_iterations=16),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    history = [Message(role=Role.SYSTEM, content="s")]
    states = []

    out = harness.run_with_callback("hi", history, on_event=lambda e: states.append((e.type, e.payload)))

    assert out.text == "Hi! How can I help?"
    assert out.stop.reason == RunStopReason.AssistantFinal
    assert out.stop.iterations == 1
    assert provider.idx == 1
    assert any(t == "on_transition" and p.get("to_state") == OrchestratorState.FINALIZE.value for t, p in states)


def test_titan_harness_routes_simple_chat_to_direct_reply_mode(tmp_path: Path):
    provider = CapturingProvider(AssistantResponse(text="Hi!"))
    harness = TitanHarness(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    events = []

    out = harness.run_with_callback("hi", [Message(role=Role.SYSTEM, content="You are Titan.")], on_event=lambda e: events.append((e.type, e.payload)))

    assert out.stop.reason == RunStopReason.AssistantFinal
    assert [p["state"] for t, p in events if t == "route_decision"] == [OrchestratorState.ACT.value]
    assert "Reply directly" in provider.calls[0][0].content


def test_titan_harness_instructs_image_tasks_to_use_attached_pixels_not_browser(tmp_path: Path):
    image = tmp_path / "Screenshot 2026-05-12 at 5.47.12 AM.png"
    image.write_bytes(b"fake-png")
    provider = CapturingProvider(AssistantResponse(text="I can see the image prompt."))
    harness = TitanHarness(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    events = []

    out = harness.run_with_callback(
        f"`{image}` Execute the prompt in the image.",
        [Message(role=Role.SYSTEM, content="You are Titan.")],
        on_event=lambda e: events.append((e.type, e.payload)),
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    system_prompt = provider.calls[0][0].content
    assert "Local image attachment guidance" in system_prompt
    assert "analyze the attached image pixels directly" in system_prompt
    assert "Do not call browser_navigate" in system_prompt
    assert "do not ask the user to upload it again" in system_prompt
    tool_names = [tool.get("function", {}).get("name") for tool in provider.tool_defs[0]]
    assert "browser_navigate" not in tool_names
    assert [payload["count"] for event_type, payload in events if event_type == "image_attachments_detected"] == [1]


def test_titan_harness_routes_delegation_requests_to_delegate_mode(tmp_path: Path):
    provider = CapturingProvider(AssistantResponse(text="I will split this into workers."))
    harness = TitanHarness(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    events = []

    harness.run_with_callback(
        "spawn subagents to review the independent modules in parallel",
        [Message(role=Role.SYSTEM, content="You are Titan.")],
        on_event=lambda e: events.append((e.type, e.payload)),
    )

    assert [p["state"] for t, p in events if t == "route_decision"] == [OrchestratorState.DELEGATE.value]
    assert "use delegate_task" in provider.calls[0][0].content


def test_titan_harness_injects_iteration_budgeted_phase_plan_for_work_tasks(tmp_path: Path):
    provider = CapturingProvider(AssistantResponse(text="Plan:\n1. Inspect.\n2. Fix.\n3. Verify."))
    harness = TitanHarness(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow", max_iterations=20),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    events = []

    out = harness.run_with_callback(
        "review and improve planning/decompose behavior",
        [Message(role=Role.SYSTEM, content="You are Titan.")],
        on_event=lambda e: events.append((e.type, e.payload)),
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    budget_events = [payload for event_type, payload in events if event_type == "plan_budget"]
    assert budget_events
    assert budget_events[0]["max_iterations"] == 20
    assert budget_events[0]["reserved_finalization_iterations"] == 1
    assert sum(phase["iterations"] for phase in budget_events[0]["phases"]) <= 20
    system_prompt = provider.calls[0][0].content
    assert "Iteration budget plan" in system_prompt
    assert "Stay within the user's configured max_iterations=20" in system_prompt
    assert "Phase 1" in system_prompt


def test_titan_harness_uses_last_iteration_to_finalize_instead_of_raw_iteration_error(tmp_path: Path):
    script = [
        AssistantResponse(text="Inspecting", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo inspect"})]),
        AssistantResponse(text="Fixing", tool_calls=[ToolCall(id="c2", name="shell", arguments={"command": "echo fix"})]),
        AssistantResponse(text="Summary:\n- Partial work completed.\n\nNext best step:\n- Continue verification."),
    ]
    provider = MockProvider(script=script)
    harness = TitanHarness(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow", max_iterations=3),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    events = []

    out = harness.run_with_callback(
        "fix a medium coding task",
        [Message(role=Role.SYSTEM, content="s")],
        on_event=lambda e: events.append((e.type, e.payload)),
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    assert out.stop.iterations == 3
    assert "iteration budget" not in out.text.lower()
    assert [payload["remaining_iterations"] for event_type, payload in events if event_type == "budget_finalization_requested"] == [1]
    budget_events = [payload for event_type, payload in events if event_type == "plan_budget"]
    assert sum(phase["iterations"] for phase in budget_events[0]["phases"]) <= 3
    final_request = provider.idx
    assert final_request == 3


def test_titan_harness_emits_plan_text_before_tool_execution(tmp_path: Path):
    script = [
        AssistantResponse(
            text="Plan:\n1. Inspect config.\n2. Run targeted tests.",
            tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo ok"})],
        ),
        AssistantResponse(text="done"),
    ]
    harness = TitanHarness(
        provider=MockProvider(script=script),
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    events = []

    out = harness.run_with_callback(
        "fix and test the harness loop behavior",
        [Message(role=Role.SYSTEM, content="s")],
        on_event=lambda e: events.append((e.type, e.payload)),
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    assistant_events = [(t, p) for t, p in events if t == "assistant_message"]
    tool_events = [(t, p) for t, p in events if t == "tool_call"]
    assert assistant_events
    assert tool_events
    assert events.index(assistant_events[0]) < events.index(tool_events[0])
    assert assistant_events[0][1]["state"] == OrchestratorState.PLAN.value
    assert assistant_events[0][1]["has_tool_calls"] is True


def test_titan_harness_emits_provider_request_before_blocking_generate(tmp_path: Path):
    provider = CapturingProvider(AssistantResponse(text="done"))
    harness = TitanHarness(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    events = []

    out = harness.run_with_callback(
        "build a website",
        [Message(role=Role.SYSTEM, content="s")],
        on_event=lambda e: events.append((e.type, e.payload)),
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    event_types = [event_type for event_type, _payload in events]
    assert "provider_request" in event_types
    assert event_types.index("provider_request") < event_types.index("assistant_message")
    provider_request = [payload for event_type, payload in events if event_type == "provider_request"][0]
    assert provider_request["iteration"] == 1
    assert provider_request["state"] == OrchestratorState.PLAN.value
    assert provider_request["tool_calls_total"] == 0
    assert provider_request["tool_calls_this_turn"] == 0
    assert "tool_count" not in provider_request


def test_titan_harness_tool_call_events_include_per_turn_count(tmp_path: Path):
    script = [
        AssistantResponse(
            text="",
            tool_calls=[
                ToolCall(id="c1", name="shell", arguments={"command": "echo one"}),
                ToolCall(id="c2", name="shell", arguments={"command": "echo two"}),
            ],
        ),
        AssistantResponse(text="done"),
    ]
    harness = TitanHarness(
        provider=MockProvider(script=script),
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    events = []

    out = harness.run_with_callback(
        "build a website",
        [Message(role=Role.SYSTEM, content="s")],
        on_event=lambda e: events.append((e.type, e.payload)),
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    tool_calls = [payload for event_type, payload in events if event_type == "tool_call"]
    assert [payload["tool_calls_this_turn"] for payload in tool_calls] == [1, 2]
    assert [payload["tool_calls_total"] for payload in tool_calls] == [1, 2]


def test_titan_harness_forwards_provider_stream_events_before_final(tmp_path: Path):
    provider = StreamingProvider(AssistantResponse(text="done"))
    harness = TitanHarness(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    events = []

    out = harness.run_with_callback(
        "build a website",
        [Message(role=Role.SYSTEM, content="s")],
        on_event=lambda e: events.append((e.type, e.payload)),
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    event_types = [event_type for event_type, _payload in events]
    assert event_types.count("provider_stream_delta") == 2
    assert event_types.index("provider_stream_delta") < event_types.index("assistant_message")
    assert [p["text"] for t, p in events if t == "provider_stream_delta"] == ["thinking", "..."]


def test_recovery_engine_escalates_human_after_retries():
    r = RecoveryEngine(max_retries=1)
    failures = []
    s1 = r.classify(failures, recovery_count=0)
    s2 = r.classify(failures, recovery_count=2)
    assert s1 == OrchestratorState.RECOVER
    assert s2 == OrchestratorState.REFLECT


def test_checkpoint_created_without_default_skill_distillation(tmp_path: Path):
    # 6 tool calls used to trigger surprise distillation; default must not create skill files.
    tool_calls = [ToolCall(id=f"c{i}", name="shell", arguments={"command": "echo ok"}) for i in range(6)]
    script = [AssistantResponse(text="", tool_calls=tool_calls), AssistantResponse(text="done")]
    sess = SessionStore(str(tmp_path / "session.jsonl"))
    harness = TitanHarness(
        provider=MockProvider(script=script),
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
        session_store=sess,
    )
    harness.learning.skills_dir = tmp_path / "skills"
    harness.learning.trajectories_dir = tmp_path / "trajectories"
    harness.learning.__post_init__()

    history = [Message(role=Role.SYSTEM, content="s")]
    out = harness.run_with_callback("task", history)
    assert out.stop.reason.value == "AssistantFinal"
    assert sess.checkpoints_path.exists()
    assert not any((tmp_path / "skills").glob("*.md"))
    assert not any((tmp_path / "trajectories").glob("*.json"))


def test_skill_distillation_is_explicit_opt_in(tmp_path: Path):
    tool_calls = [ToolCall(id=f"c{i}", name="shell", arguments={"command": "echo ok"}) for i in range(6)]
    script = [AssistantResponse(text="", tool_calls=tool_calls), AssistantResponse(text="done")]
    sess = SessionStore(str(tmp_path / "session.jsonl"))
    harness = TitanHarness(
        provider=MockProvider(script=script),
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow", learning_enabled=True),
        session_store=sess,
    )
    harness.learning.skills_dir = tmp_path / "skills"
    harness.learning.trajectories_dir = tmp_path / "trajectories"
    harness.learning.__post_init__()

    history = [Message(role=Role.SYSTEM, content="s")]
    out = harness.run_with_callback("task", history)
    assert out.stop.reason.value == "AssistantFinal"
    assert any(p.name.endswith(".md") for p in (tmp_path / "skills").glob("*.md"))
    assert any(p.name.endswith(".json") for p in (tmp_path / "trajectories").glob("*.json"))


def test_titan_harness_enforces_per_iteration_tool_cap(tmp_path: Path):
    tool_calls = [
        ToolCall(id="c1", name="shell", arguments={"command": "echo should-not-run-1"}),
        ToolCall(id="c2", name="shell", arguments={"command": "echo should-not-run-2"}),
    ]
    harness = TitanHarness(
        provider=MockProvider(script=[AssistantResponse(text="", tool_calls=tool_calls)]),
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow", max_tool_calls_per_iteration=1),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    history = [Message(role=Role.SYSTEM, content="s")]
    events = []

    out = harness.run_with_callback("task", history, on_event=lambda e: events.append((e.type, e.payload)))

    assert out.stop.reason == RunStopReason.BudgetToolsIteration
    assert out.stop.iterations == 1
    assert out.stop.tool_calls_total == 0
    assert out.stop.notes == "tool calls per iteration exceeded"
    assert not any(msg.role == Role.TOOL for msg in history)
    event_types = [event_type for event_type, _payload in events]
    assert "tool_batch_started" not in event_types
    assert event_types.count("tool_call_rejected") == 2
    rejected = [payload for event_type, payload in events if event_type == "tool_call_rejected"]
    assert [payload["name"] for payload in rejected] == ["shell", "shell"]
    assert rejected[0]["arguments"] == {"command": "echo should-not-run-1"}
    assert rejected[1]["arguments"] == {"command": "echo should-not-run-2"}
