from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import HarnessConfig
from .loop import AgentLoop
from .mock_provider import MockProvider
from .session import SessionStore
from .tools import default_registry
from .types import AssistantResponse, Message, Role, RunStopReason, ToolCall


@dataclass
class EvalCaseResult:
    name: str
    expected: RunStopReason
    actual: RunStopReason
    passed: bool


def _run_case(name: str, script: list[AssistantResponse], cfg: HarnessConfig, expected: RunStopReason) -> EvalCaseResult:
    loop = AgentLoop(
        provider=MockProvider(script=script),
        tools=default_registry(),
        config=cfg,
        session=SessionStore(str(Path(".titan") / f"eval-{name}.jsonl")),
    )
    history = [Message(role=Role.SYSTEM, content="You are Titan.")]
    out = loop.run("execute", history)
    return EvalCaseResult(name=name, expected=expected, actual=out.stop.reason, passed=(out.stop.reason == expected))


def run_accuracy_eval() -> list[EvalCaseResult]:
    results: list[EvalCaseResult] = []

    # assistant final
    cfg = HarnessConfig(permission_mode="allow")
    script = [
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo hi"})]),
        AssistantResponse(text="done"),
    ]
    results.append(_run_case("assistant_final", script, cfg, RunStopReason.AssistantFinal))

    # empty final after tools now recovers and can still finalize
    cfg = HarnessConfig(permission_mode="allow", max_iterations=6, max_consecutive_empty_turns=2)
    script = [
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="recovered", tool_calls=[]),
    ]
    results.append(_run_case("empty_final_after_tools_recovery", script, cfg, RunStopReason.AssistantFinal))

    # repeated empty post-tool turns exhaust recovery
    cfg = HarnessConfig(permission_mode="allow", max_iterations=6, max_consecutive_empty_turns=1)
    script = [
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo x"})]),
        AssistantResponse(text="", tool_calls=[]),
        AssistantResponse(text="", tool_calls=[]),
    ]
    results.append(_run_case("empty_final_recovery_exhausted", script, cfg, RunStopReason.ErrorRecoveryExhausted))

    # per-iteration tool budget exceeded
    cfg = HarnessConfig(permission_mode="allow", max_tool_calls_per_iteration=1)
    script = [
        AssistantResponse(
            text="",
            tool_calls=[
                ToolCall(id="c1", name="shell", arguments={"command": "echo 1"}),
                ToolCall(id="c2", name="shell", arguments={"command": "echo 2"}),
            ],
        )
    ]
    results.append(_run_case("budget_tools_iteration", script, cfg, RunStopReason.BudgetToolsIteration))

    # max iterations budget
    cfg = HarnessConfig(permission_mode="allow", max_iterations=1)
    script = [
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo loop"})]),
        AssistantResponse(text="", tool_calls=[ToolCall(id="c2", name="shell", arguments={"command": "echo loop2"})]),
    ]
    results.append(_run_case("budget_iterations", script, cfg, RunStopReason.BudgetIterations))

    return results
