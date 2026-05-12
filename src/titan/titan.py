from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .config import HarnessConfig
from .loop import AgentEvent
from .permissions import PermissionError, PermissionPolicy
from .provider import Provider, ProviderError, retry_call
from .session import SessionStore
from .tools import ToolRegistry
from .types import Message, Role, RunOutcome, RunStopContract, RunStopReason, ToolResult


class OrchestratorState(str, Enum):
    PLAN = "PLAN"
    ACT = "ACT"
    REFLECT = "REFLECT"
    RECOVER = "RECOVER"
    DELEGATE = "DELEGATE"
    FINALIZE = "FINALIZE"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}

    def on(self, event: str, handler: Callable[[dict[str, Any]], None]) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, payload: Optional[dict[str, Any]] = None) -> None:
        data = payload or {}
        for h in self._handlers.get(event, []):
            h(data)


@dataclass
class LearningLoop:
    skills_dir: Path = Path(".titan/skills")
    trajectories_dir: Path = Path(".titan/trajectories")

    def __post_init__(self) -> None:
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)

    def distill_skill(self, trace_id: str, trajectory: list[dict[str, Any]], success_score: float) -> Path:
        name = f"skill-{trace_id}.md"
        p = self.skills_dir / name
        p.write_text(
            "# Auto-distilled skill\n\n"
            f"trace_id: {trace_id}\n"
            f"success_score: {success_score:.2f}\n\n"
            "## Procedure\n"
            "1. Reuse this flow when similar failures recur.\n"
            "2. Keep tool calls bounded and checkpoint often.\n"
        )
        traj = self.trajectories_dir / f"{trace_id}.json"
        traj.write_text(json.dumps({"trace_id": trace_id, "trajectory": trajectory, "success": success_score}, indent=2))
        return p


def _should_distill_skill(config: HarnessConfig, tool_calls_total: int, turn: int) -> bool:
    """Auto-distillation is opt-in; default Titan loop must not create surprise files.

    Pi keeps tool execution/result handling separate from extension/skill management:
    unknown or failed tool calls are returned as tool results, not converted into new tools or
    skills. Titan follows that default and only writes distilled skill artifacts when explicitly
    enabled in config.
    """
    return bool(config.learning_enabled) and tool_calls_total > 5 and turn >= 2


class MemoryManager:
    def __init__(self) -> None:
        self.tool_results: list[str] = []

    def get_relevant_context(self, query: str, max_tokens: int = 4096) -> str:
        # minimal relevance strategy: keep tail of tool evidence
        tail = self.tool_results[-8:]
        joined = "\n".join(tail)
        return (query + "\n" + joined)[: max_tokens * 4]

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.tool_results.append(f"{r.tool_name}: {r.content[:300]}")


class RecoveryEngine:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def classify(self, failures: list[ToolResult], recovery_count: int) -> OrchestratorState:
        if recovery_count < self.max_retries:
            return OrchestratorState.RECOVER
        if any("permission" in (f.content or "").lower() for f in failures):
            return OrchestratorState.HUMAN_APPROVAL
        return OrchestratorState.REFLECT


@dataclass
class TitanSession:
    goal: str
    trace_id: str
    current_state: OrchestratorState = OrchestratorState.PLAN
    turn: int = 0
    recovery_count: int = 0
    context_budget: int = 4096
    strict_mode: bool = True
    risk_threshold: float = 0.7
    trace: list[dict[str, Any]] = field(default_factory=list)


class Supervisor:
    def decide_next_state(self, current_state: OrchestratorState, has_tools: bool, has_failures: bool, final_text: str) -> OrchestratorState:
        if has_failures:
            return OrchestratorState.RECOVER
        if current_state == OrchestratorState.PLAN:
            return OrchestratorState.ACT
        if has_tools:
            return OrchestratorState.REFLECT
        if final_text.strip():
            return OrchestratorState.FINALIZE
        return OrchestratorState.ACT


@dataclass
class RouteDecision:
    state: OrchestratorState
    reason: str
    instruction: str


class IntentRouter:
    DIRECT_KEYWORDS = {
        "hi", "hello", "hey", "yo", "thanks", "thank you", "ok", "okay", "yes", "no",
        "what can you do", "help", "who are you",
    }
    DELEGATE_KEYWORDS = {
        "delegate", "subagent", "subagents", "spawn", "parallel", "independent", "reviewers", "workers",
    }
    PLAN_KEYWORDS = {
        "build", "implement", "fix", "debug", "refactor", "design", "plan", "add", "create", "continue",
        "test", "wire", "integrate", "ship", "long task", "multi-step", "milestone",
    }

    def decide(self, task: str) -> RouteDecision:
        normalized = " ".join(task.lower().strip().split())
        word_count = len(normalized.split())
        if not normalized:
            return RouteDecision(OrchestratorState.FINALIZE, "empty input", "Ask for a concrete task; do not run tools.")
        if any(k in normalized for k in self.DELEGATE_KEYWORDS):
            return RouteDecision(
                OrchestratorState.DELEGATE,
                "delegation cue detected",
                "Decompose the work, use delegate_task for independent subtasks when useful, then integrate and verify results.",
            )
        if normalized in self.DIRECT_KEYWORDS or (word_count <= 4 and not any(k in normalized for k in self.PLAN_KEYWORDS)):
            return RouteDecision(
                OrchestratorState.ACT,
                "simple conversational input",
                "Reply directly and concisely. Do not make a plan and do not call tools unless the user asks for current facts or system state.",
            )
        if any(k in normalized for k in self.PLAN_KEYWORDS) or word_count >= 12:
            return RouteDecision(
                OrchestratorState.PLAN,
                "implementation or multi-step work cue detected",
                "Start with a brief plan only when it reduces risk, then execute. Prefer tool use for code/file/system work and verify changes.",
            )
        return RouteDecision(
            OrchestratorState.ACT,
            "default actionable reply",
            "Answer directly. Escalate to planning only if the task reveals multiple dependent steps.",
        )


class TitanHarness:
    def __init__(self, provider: Provider, tools: ToolRegistry, config: HarnessConfig, session_store: Optional[SessionStore] = None):
        self.provider = provider
        self.tools = tools
        self.config = config
        self.session_store = session_store or SessionStore(".titan/session.jsonl")
        self.policy = PermissionPolicy(config.permission_mode)
        self.event_bus = EventBus()
        self.supervisor = Supervisor()
        self.router = IntentRouter()
        self.memory = MemoryManager()
        self.recovery = RecoveryEngine()
        self.learning = LearningLoop()

    def run_with_callback(self, task: str, history: list[Message], on_event: Optional[Callable[[AgentEvent], None]] = None) -> RunOutcome:
        trace_id = self.session_store.trace_id
        route_decision = self.router.decide(task)
        session = TitanSession(goal=task, trace_id=trace_id, current_state=route_decision.state)
        self.session_store.checkpoint(session.current_state.value, 0, f"run_start:{route_decision.reason}")
        self._append(history, Message(role=Role.USER, content=task))
        started = time.time()
        tool_calls_total = 0

        def emit(t: str, **payload: Any) -> None:
            self.event_bus.emit(t, payload)
            if on_event:
                on_event(AgentEvent(type=t, payload=payload))

        emit("route_decision", state=route_decision.state.value, reason=route_decision.reason, instruction=route_decision.instruction)

        while session.turn < self.config.max_iterations:
            session.turn += 1
            emit("on_state_enter", state=session.current_state.value, turn=session.turn)
            context = self.memory.get_relevant_context(session.goal, session.context_budget)
            session.trace.append({"turn": session.turn, "state": session.current_state.value, "context_len": len(context)})
            tools = self.tools.definitions()
            emit(
                "provider_request",
                iteration=session.turn,
                state=session.current_state.value,
                model=self.config.model,
                provider=self.config.provider,
                tool_count=len(tools),
            )

            try:
                resp = retry_call(
                    lambda: self.provider.generate_with_callback(
                        self.config.model,
                        self._provider_history(history, route_decision),
                        tools,
                        on_event=lambda event_type, **event_payload: emit(f"provider_{event_type}", **event_payload),
                    ),
                    self.config.retry,
                )
            except ProviderError as e:
                elapsed_ms = int((time.time() - started) * 1000)
                return RunOutcome(
                    text=f"Provider error: {e}",
                    stop=RunStopContract(
                        reason=RunStopReason.ErrorRetryExhausted if e.retryable else RunStopReason.ErrorNonRetryable,
                        iterations=session.turn,
                        tool_calls_total=tool_calls_total,
                        elapsed_ms=elapsed_ms,
                        notes=str(e),
                    ),
                )

            self._append(history, Message(role=Role.ASSISTANT, content=resp.text))
            if resp.text.strip():
                emit(
                    "assistant_message",
                    text=resp.text,
                    state=session.current_state.value,
                    has_tool_calls=bool(resp.tool_calls),
                )

            if not resp.tool_calls:
                elapsed_ms = int((time.time() - started) * 1000)
                if _should_distill_skill(self.config, tool_calls_total, session.turn):
                    skill_path = self.learning.distill_skill(trace_id, session.trace, success_score=1.0 if resp.text.strip() else 0.5)
                    emit("on_skill_created", path=str(skill_path))
                emit("on_transition", from_state=session.current_state.value, to_state=OrchestratorState.FINALIZE.value, turn=session.turn)
                emit("on_state_exit", state=session.current_state.value, turn=session.turn)
                session.current_state = OrchestratorState.FINALIZE
                self.session_store.checkpoint(session.current_state.value, session.turn, "finalize")
                return RunOutcome(
                    text=resp.text,
                    stop=RunStopContract(
                        reason=RunStopReason.AssistantFinal,
                        iterations=session.turn,
                        tool_calls_total=tool_calls_total,
                        elapsed_ms=elapsed_ms,
                        notes=f"trace_id={trace_id}",
                    ),
                )

            results: list[ToolResult] = []
            if resp.tool_calls:
                if len(resp.tool_calls) > self.config.max_tool_calls_per_iteration:
                    emit(
                        "tool_batch_rejected",
                        count=len(resp.tool_calls),
                        max_tool_calls_per_iteration=self.config.max_tool_calls_per_iteration,
                    )
                    for index, tc in enumerate(resp.tool_calls, start=1):
                        emit(
                            "tool_call_rejected",
                            id=tc.id,
                            name=tc.name,
                            arguments=tc.arguments,
                            index=index,
                            count=len(resp.tool_calls),
                            reason="max_tool_calls_per_iteration",
                        )
                    elapsed_ms = int((time.time() - started) * 1000)
                    return RunOutcome(
                        text="",
                        stop=RunStopContract(
                            reason=RunStopReason.BudgetToolsIteration,
                            iterations=session.turn,
                            tool_calls_total=tool_calls_total,
                            elapsed_ms=elapsed_ms,
                            notes="tool calls per iteration exceeded",
                        ),
                    )

                emit("tool_batch_started", count=len(resp.tool_calls))
                for tc in resp.tool_calls:
                    tool_calls_total += 1
                    emit("tool_call", id=tc.id, name=tc.name, arguments=tc.arguments, count=tool_calls_total)
                    try:
                        self.policy.authorize(tc.name)
                        tr = self.tools.execute(tc.id, tc.name, tc.arguments)
                    except PermissionError as e:
                        tr = ToolResult(call_id=tc.id, tool_name=tc.name, content=str(e), is_error=True)
                    self._append(history, Message(role=Role.TOOL, content=tr.content, tool_call_id=tr.call_id, tool_name=tr.tool_name, is_error=tr.is_error))
                    results.append(tr)
                    emit("tool_result", id=tr.call_id, name=tr.tool_name, is_error=tr.is_error, content=tr.content)

                self.memory.add_tool_results(results)

            failures = [r for r in results if r.is_error]
            next_state = self.supervisor.decide_next_state(
                session.current_state,
                has_tools=bool(resp.tool_calls),
                has_failures=bool(failures),
                final_text=resp.text,
            )

            if failures:
                session.recovery_count += 1
                next_state = self.recovery.classify(failures, session.recovery_count)

            emit("on_transition", from_state=session.current_state.value, to_state=next_state.value, turn=session.turn)
            emit("on_state_exit", state=session.current_state.value, turn=session.turn)
            session.current_state = next_state

            if session.turn % 5 == 0:
                self.session_store.checkpoint(session.current_state.value, session.turn, "periodic")

            if _should_distill_skill(self.config, tool_calls_total, session.turn):
                skill_path = self.learning.distill_skill(trace_id, session.trace, success_score=1.0 if resp.text.strip() else 0.5)
                emit("on_skill_created", path=str(skill_path))

            if session.current_state == OrchestratorState.FINALIZE:
                elapsed_ms = int((time.time() - started) * 1000)
                self.session_store.checkpoint(session.current_state.value, session.turn, "finalize")
                return RunOutcome(
                    text=resp.text,
                    stop=RunStopContract(
                        reason=RunStopReason.AssistantFinal,
                        iterations=session.turn,
                        tool_calls_total=tool_calls_total,
                        elapsed_ms=elapsed_ms,
                        notes=f"trace_id={trace_id}",
                    ),
                )

        elapsed_ms = int((time.time() - started) * 1000)
        return RunOutcome(
            text="",
            stop=RunStopContract(
                reason=RunStopReason.BudgetIterations,
                iterations=session.turn,
                tool_calls_total=tool_calls_total,
                elapsed_ms=elapsed_ms,
                notes="max_iterations",
            ),
        )

    def _provider_history(self, history: list[Message], decision: RouteDecision) -> list[Message]:
        route_text = (
            "Titan routing decision:\n"
            f"- mode: {decision.state.value}\n"
            f"- reason: {decision.reason}\n"
            f"- instruction: {decision.instruction}"
        )
        if history and history[0].role == Role.SYSTEM:
            return [Message(role=Role.SYSTEM, content=f"{history[0].content}\n\n{route_text}"), *history[1:]]
        return [Message(role=Role.SYSTEM, content=route_text), *history]

    def _append(self, history: list[Message], msg: Message) -> None:
        history.append(msg)
        self.session_store.append(msg)
