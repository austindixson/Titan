from titan.config import HarnessConfig
from titan.loop import AgentLoop
from titan.mock_provider import MockProvider, make_tool_then_final_script
from titan.session import SessionStore
from titan.tools import default_registry
from titan.types import Message, Role, AssistantResponse, ToolCall, RunStopReason


def test_tool_then_final_completes(tmp_path):
    cfg = HarnessConfig()
    provider = MockProvider(script=make_tool_then_final_script())
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("say hi", hist)
    assert out.stop.reason == RunStopReason.AssistantFinal
    assert "Done" in out.text


def test_empty_final_after_tools_recovers_and_completes(tmp_path):
    cfg = HarnessConfig(max_iterations=6, max_consecutive_empty_turns=3)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="Recovered final answer", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.AssistantFinal
    assert out.text == "Recovered final answer"


def test_empty_final_after_tools_returns_diagnostic_when_recovery_exhausted(tmp_path):
    cfg = HarnessConfig(max_iterations=8, max_consecutive_empty_turns=2)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="   ", tool_calls=[]),
        AssistantResponse(text="", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2b.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.ErrorRecoveryExhausted
    assert "empty assistant turns" in out.text
    assert out.stop.notes


def test_empty_final_before_any_tools_stays_final(tmp_path):
    cfg = HarnessConfig(max_iterations=3)
    provider = MockProvider(script=[AssistantResponse(text="", tool_calls=[])])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2c.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.AssistantFinal
    assert "Titan stopped before completion" in out.text


def test_tool_error_then_empty_turn_can_recover(tmp_path):
    cfg = HarnessConfig(max_iterations=6, max_consecutive_empty_turns=2, permission_mode="deny")
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="Recovered despite tool failure", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2d.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.AssistantFinal
    assert "Recovered despite tool failure" in out.text


def test_recovery_prompts_are_persisted_to_session(tmp_path):
    session = SessionStore(str(tmp_path / "persist.jsonl"))
    cfg = HarnessConfig(max_iterations=6, max_consecutive_empty_turns=2)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="Recovered final answer", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, session)
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)

    assert out.stop.reason == RunStopReason.AssistantFinal
    rows = (tmp_path / "persist.jsonl").read_text()
    assert "Recovery attempt after an empty assistant turn" in rows
    assert "Continue and finish the task using the available context and tools." in rows


def test_explicit_empty_final_reason_is_now_recovered_not_emitted(tmp_path):
    cfg = HarnessConfig(max_iterations=6, max_consecutive_empty_turns=2)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="Recovered final answer", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2f.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.AssistantFinal
    assert out.stop.reason != RunStopReason.ErrorRecoveryExhausted


def test_total_tool_budget_returns_non_empty_text(tmp_path):
    cfg = HarnessConfig(max_iterations=6, max_tool_calls_total=1)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[
            ToolCall(id="c1", name="shell", arguments={"command": "echo 1"}),
            ToolCall(id="c2", name="shell", arguments={"command": "echo 2"}),
        ]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2g.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.BudgetToolsTotal
    assert out.text
    assert "tool-call budget" in out.text


def test_per_iteration_tool_budget_returns_non_empty_text(tmp_path):
    cfg = HarnessConfig(max_tool_calls_per_iteration=1)
    provider = MockProvider(script=[AssistantResponse(text="", tool_calls=[
        ToolCall(id="c1", name="shell", arguments={"command": "echo 1"}),
        ToolCall(id="c2", name="shell", arguments={"command": "echo 2"}),
    ])])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2h.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.BudgetToolsIteration
    assert out.text
    assert "single iteration" in out.text


def test_provider_error_returns_non_empty_text(tmp_path):
    class FailingProvider:
        def generate(self, model, messages, tools):
            from titan.provider import ProviderError
            raise ProviderError("boom", retryable=False)

    cfg = HarnessConfig(max_iterations=3)
    loop = AgentLoop(FailingProvider(), default_registry(), cfg, SessionStore(str(tmp_path / "s2i.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.ErrorNonRetryable
    assert "Provider error: boom" in out.text


def test_no_user_message_path_returns_non_empty_text(tmp_path):
    cfg = HarnessConfig(max_iterations=3)
    loop = AgentLoop(MockProvider(script=[]), default_registry(), cfg, SessionStore(str(tmp_path / "s2j.jsonl")))
    out = loop.run_with_callback("", [], on_event=None)
    assert out.text
    assert out.stop.reason in {RunStopReason.ErrorNonRetryable, RunStopReason.AssistantFinal}


def test_whitespace_final_answer_is_treated_as_empty_after_tools(tmp_path):
    cfg = HarnessConfig(max_iterations=6, max_consecutive_empty_turns=1)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="   ", tool_calls=[]),
        AssistantResponse(text="", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2k.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.ErrorRecoveryExhausted


def test_default_config_values_are_long_task_friendly():
    cfg = HarnessConfig()
    assert cfg.max_iterations == 75
    assert cfg.max_wall_clock_ms == 600000
    assert cfg.max_tool_calls_total == 256
    assert cfg.max_consecutive_empty_turns == 3


def test_empty_before_tools_keeps_non_empty_diagnostic_text(tmp_path):
    cfg = HarnessConfig(max_iterations=3)
    provider = MockProvider(script=[AssistantResponse(text="   ", tool_calls=[])])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2l.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.AssistantFinal
    assert out.text


def test_recovery_event_payload_is_emitted(tmp_path):
    cfg = HarnessConfig(max_iterations=6, max_consecutive_empty_turns=2)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="Recovered final answer", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2m.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    events = []
    out = loop.run_with_callback("do", hist, on_event=lambda e: events.append((e.type, e.payload)))
    assert out.stop.reason == RunStopReason.AssistantFinal
    recovery = [payload for event_type, payload in events if event_type == "empty_turn_recovery"]
    assert recovery
    assert recovery[0]["consecutive_empty_turns"] == 1
    assert recovery[0]["tool_calls_total"] == 1


def test_recovery_does_not_reset_tool_budget_enforcement(tmp_path):
    cfg = HarnessConfig(max_iterations=8, max_consecutive_empty_turns=2, max_tool_calls_total=2)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo 1"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="", tool_calls=[
            ToolCall(id="c2", name="shell", arguments={"command": "echo 2"}),
            ToolCall(id="c3", name="shell", arguments={"command": "echo 3"}),
        ]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2n.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.BudgetToolsTotal
    assert out.text


def test_recovery_prompts_preserve_original_user_task_in_history(tmp_path):
    session = SessionStore(str(tmp_path / "persist2.jsonl"))
    cfg = HarnessConfig(max_iterations=6, max_consecutive_empty_turns=2)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="Recovered final answer", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, session)
    hist = [Message(role=Role.SYSTEM, content="sys")]
    loop.run("solve the big task", hist)
    joined = "\n".join(m.content for m in hist if m.content)
    assert "solve the big task" in joined
    assert "Recovery attempt after an empty assistant turn" in joined


def test_recovery_exhausted_text_mentions_recovery(tmp_path):
    cfg = HarnessConfig(max_iterations=8, max_consecutive_empty_turns=1)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2o.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert "could not recover" in out.text


def test_recovery_respects_iteration_budget(tmp_path):
    cfg = HarnessConfig(max_iterations=1, max_consecutive_empty_turns=5)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo 1"})]),
        AssistantResponse(text="", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2p.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason in {RunStopReason.BudgetIterations, RunStopReason.ErrorRecoveryExhausted, RunStopReason.AssistantFinal}
    assert out.text


def test_recovery_respects_wall_clock_budget(tmp_path):
    cfg = HarnessConfig(max_iterations=6, max_consecutive_empty_turns=2, max_wall_clock_ms=-1)
    provider = MockProvider(script=[AssistantResponse(text="", tool_calls=[])])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2q.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.BudgetWallClock
    assert out.text


def test_recovery_diagnostic_includes_notes(tmp_path):
    cfg = HarnessConfig(max_iterations=8, max_consecutive_empty_turns=1)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2r.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.notes in out.text


def test_recovery_keeps_usage_dict_on_final(tmp_path):
    cfg = HarnessConfig(max_iterations=6, max_consecutive_empty_turns=2)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})], input_tokens=10, output_tokens=3),
        AssistantResponse(text="", tool_calls=[], input_tokens=11, output_tokens=0),
        AssistantResponse(text="Recovered final answer", tool_calls=[], input_tokens=12, output_tokens=5),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2s.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.AssistantFinal
    assert out.usage["input_tokens"] == 33
    assert out.usage["output_tokens"] == 8


def test_recovery_keeps_usage_dict_on_recovery_exhausted(tmp_path):
    cfg = HarnessConfig(max_iterations=8, max_consecutive_empty_turns=1)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})], input_tokens=10, output_tokens=3),
        AssistantResponse(text="", tool_calls=[], input_tokens=11, output_tokens=0),
        AssistantResponse(text="", tool_calls=[], input_tokens=12, output_tokens=0),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2t.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.ErrorRecoveryExhausted
    assert out.usage["input_tokens"] == 33
    assert out.usage["output_tokens"] == 3


def test_recovery_prompt_is_user_internal_note(tmp_path):
    cfg = HarnessConfig(max_iterations=6, max_consecutive_empty_turns=2)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="Recovered final answer", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2u.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    loop.run("do", hist)
    system_messages = [m for m in hist if m.role == Role.SYSTEM]
    notes = [m for m in hist if m.role == Role.USER and m.content.startswith("Titan internal note:")]
    assert len(system_messages) == 1
    assert notes
    assert "Recovery attempt after an empty assistant turn" in notes[0].content
    assert "Continue and finish the task" in notes[0].content


def test_recovery_occurs_only_after_tools(tmp_path):
    cfg = HarnessConfig(max_iterations=3, max_consecutive_empty_turns=2)
    provider = MockProvider(script=[AssistantResponse(text="", tool_calls=[]), AssistantResponse(text="recovered", tool_calls=[])])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2v.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    events = []
    out = loop.run_with_callback("do", hist, on_event=lambda e: events.append(e.type))
    assert out.stop.reason == RunStopReason.AssistantFinal
    assert "empty_turn_recovery" not in events


def test_recovery_preserves_session_logging_with_tools(tmp_path):
    session = SessionStore(str(tmp_path / "persist3.jsonl"))
    cfg = HarnessConfig(max_iterations=6, max_consecutive_empty_turns=2)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="Recovered final answer", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, session)
    hist = [Message(role=Role.SYSTEM, content="sys")]
    loop.run("do", hist)
    rows = (tmp_path / "persist3.jsonl").read_text()
    assert '"tool_name": "shell"' in rows
    assert "Recovered final answer" in rows


def test_configurable_empty_turn_limit_from_file(monkeypatch, tmp_path):
    from titan.config import resolve_config_path, write_default_config, update_config_key, load_harness_config

    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(cfg_path))
    write_default_config(resolve_config_path(), force=True)
    update_config_key(resolve_config_path(), "max_consecutive_empty_turns", "7")
    cfg = load_harness_config()
    assert cfg.max_consecutive_empty_turns == 7


def test_error_recovery_stop_reason_enum_exists():
    assert RunStopReason.ErrorRecoveryExhausted.value == "ErrorRecoveryExhausted"


def test_tiny_tool_budget_still_reports_helpfully(tmp_path):
    cfg = HarnessConfig(max_iterations=6, max_tool_calls_total=0)
    provider = MockProvider(script=[AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo 1"})])])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2w.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.BudgetToolsTotal
    assert out.text


def test_whitespace_budget_text_is_replaced_with_diagnostic(tmp_path):
    cfg = HarnessConfig(max_iterations=1)
    provider = MockProvider(script=[
        AssistantResponse(text=" ", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo loop"})]),
        AssistantResponse(text=" ", tool_calls=[ToolCall(id="c2", name="shell", arguments={"command": "echo loop2"})]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2x.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.BudgetIterations
    assert out.text


def test_recovery_path_emits_run_completed_with_notes(tmp_path):
    cfg = HarnessConfig(max_iterations=8, max_consecutive_empty_turns=1)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="", tool_calls=[]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2y.jsonl")))
    seen = []
    hist = [Message(role=Role.SYSTEM, content="sys")]
    loop.run_with_callback("do", hist, on_event=lambda e: seen.append((e.type, e.payload)))
    completed = [payload for event_type, payload in seen if event_type == "run_completed"]
    assert completed
    assert completed[-1]["notes"]


def test_empty_toolless_final_without_history_user_still_returns_text(tmp_path):
    cfg = HarnessConfig(max_iterations=3)
    loop = AgentLoop(MockProvider(script=[]), default_registry(), cfg, SessionStore(str(tmp_path / "s2z.jsonl")))
    out = loop.run("do", [Message(role=Role.SYSTEM, content="sys")])
    assert out.text


def test_budget_stop_returns_non_empty_text(tmp_path):
    cfg = HarnessConfig(max_iterations=1)
    provider = MockProvider(script=[
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo loop"})]),
        AssistantResponse(text="", tool_calls=[ToolCall(id="c2", name="shell", arguments={"command": "echo loop2"})]),
    ])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s2e.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.BudgetIterations
    assert out.text
    assert "iteration budget" in out.text


def test_tool_cap_hits(tmp_path):
    cfg = HarnessConfig(max_tool_calls_per_iteration=1)
    provider = MockProvider(script=[AssistantResponse(text="", tool_calls=[
        ToolCall(id="c1", name="shell", arguments={"command":"echo 1"}),
        ToolCall(id="c2", name="shell", arguments={"command":"echo 2"}),
    ])])
    loop = AgentLoop(provider, default_registry(), cfg, SessionStore(str(tmp_path / "s3.jsonl")))
    hist = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run("do", hist)
    assert out.stop.reason == RunStopReason.BudgetToolsIteration


def test_interrupt_flag_stops_before_provider(tmp_path):
    provider = MockProvider(script=[AssistantResponse(text="should not run")])
    loop = AgentLoop(provider, default_registry(), HarnessConfig(), SessionStore(str(tmp_path / "s-int.jsonl")))
    loop.request_interrupt()
    out = loop.run("do", [Message(role=Role.SYSTEM, content="sys")])
    assert out.stop.reason == RunStopReason.Interrupted
    assert provider.idx == 0
