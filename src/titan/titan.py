from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .config import HarnessConfig
from .image_paths import candidate_image_paths_from_text, local_image_references_from_text
from .leftover import CONTINUE_LOOP, leftover_block_note
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


def _allocate_phase_iterations(max_iterations: int, phases: list[str]) -> list[dict[str, Any]]:
    usable = max(0, max_iterations - 1)
    count = max(1, len(phases))
    base = usable // count
    remaining = usable - (base * count)
    allocations: list[dict[str, Any]] = []
    for index, phase in enumerate(phases, start=1):
        iterations = base + (1 if remaining > 0 else 0)
        remaining = max(0, remaining - 1)
        allocations.append({"phase": index, "name": phase, "iterations": iterations})
    return allocations


def _phase_names_for_route(decision: RouteDecision) -> list[str]:
    if decision.state == OrchestratorState.DELEGATE:
        return ["scope", "delegate", "integrate", "verify", "finalize"]
    if decision.state == OrchestratorState.PLAN:
        return ["scope", "inspect", "implement", "verify", "finalize"]
    return ["answer"]


def _iteration_budget_plan(config: HarnessConfig, decision: RouteDecision) -> dict[str, Any]:
    phases = _phase_names_for_route(decision)
    allocations = _allocate_phase_iterations(config.max_iterations, phases)
    return {
        "max_iterations": config.max_iterations,
        "reserved_finalization_iterations": 1,
        "phases": allocations,
    }


def _format_iteration_budget_plan(plan: dict[str, Any]) -> str:
    phase_lines = []
    for phase in plan["phases"]:
        phase_lines.append(f"Phase {phase['phase']}: {phase['name']} — target {phase['iterations']} iteration(s)")
    return (
        "Iteration budget plan:\n"
        f"Stay within the user's configured max_iterations={plan['max_iterations']}.\n"
        f"Reserve {plan['reserved_finalization_iterations']} final iteration(s) for summary/next step instead of running into the limit.\n"
        + "\n".join(phase_lines)
        + "\nIf remaining iterations are low, stop taking new tool actions and produce a concise Summary + Next best step."
    )


def _format_image_attachment_guidance(paths: list[Path], refs: list[str]) -> str:
    if not refs:
        return ""
    path_lines = "\n".join(f"- {path}" for path in refs)
    attachment_note = ""
    if not paths:
        attachment_note = (
            "Note: one or more referenced local image paths may be missing/unreadable right now; "
            "do not use browser_navigate for them. Ask for a corrected local path if needed.\n"
        )
    return (
        "Local image attachment guidance:\n"
        "The user supplied local image path(s), and the provider request includes them as image attachments when supported.\n"
        f"{path_lines}\n"
        f"{attachment_note}"
        "For image/screenshot tasks, analyze the attached image pixels directly. "
        "Do not call browser_navigate or web tools for these local file paths. "
        "Do not say you cannot read the local path and do not ask the user to upload it again unless the provider returns an explicit image/vision error."
    )


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


@dataclass
class _HarnessRun:
    task: str
    history: list[Message]
    session: TitanSession
    route_decision: RouteDecision
    plan_budget: dict[str, Any]
    plan_budget_text: str
    image_paths: list[Path]
    image_refs: list[str]
    image_guidance_text: str
    started: float
    emit: Callable[..., None]
    tool_calls_total: int = 0
    tool_calls_this_turn: int = 0
    budget_finalization_requested: bool = False


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

    def _emit_harness(self, on_event: Optional[Callable[[AgentEvent], None]], event_type: str, **payload: Any) -> None:
        self.event_bus.emit(event_type, payload)
        if on_event:
            on_event(AgentEvent(type=event_type, payload=payload))

    def _assistant_final_outcome(self, run: _HarnessRun, text: str, notes: str) -> RunOutcome:
        elapsed_ms = int((time.time() - run.started) * 1000)
        return RunOutcome(
            text=text,
            stop=RunStopContract(
                reason=RunStopReason.AssistantFinal,
                iterations=run.session.turn,
                tool_calls_total=run.tool_calls_total,
                elapsed_ms=elapsed_ms,
                notes=notes,
            ),
        )

    def _maybe_distill_skill(self, run: _HarnessRun, resp) -> None:
        if not _should_distill_skill(self.config, run.tool_calls_total, run.session.turn):
            return
        score = 1.0 if resp.text.strip() else 0.5
        skill_path = self.learning.distill_skill(run.session.trace_id, run.session.trace, success_score=score)
        run.emit("on_skill_created", path=str(skill_path))

    def _start_harness_run(self, task: str, history: list[Message], on_event: Optional[Callable[[AgentEvent], None]]) -> _HarnessRun:
        trace_id = self.session_store.trace_id
        route_decision = self.router.decide(task)
        plan_budget = _iteration_budget_plan(self.config, route_decision)
        plan_budget_text = _format_iteration_budget_plan(plan_budget)
        image_paths = candidate_image_paths_from_text(task)
        image_refs = local_image_references_from_text(task)
        image_guidance_text = _format_image_attachment_guidance(image_paths, image_refs)
        session = TitanSession(goal=task, trace_id=trace_id, current_state=route_decision.state)
        self.session_store.checkpoint(session.current_state.value, 0, f"run_start:{route_decision.reason}")
        self._append(history, Message(role=Role.USER, content=task))
        run = _HarnessRun(
            task=task,
            history=history,
            session=session,
            route_decision=route_decision,
            plan_budget=plan_budget,
            plan_budget_text=plan_budget_text,
            image_paths=image_paths,
            image_refs=image_refs,
            image_guidance_text=image_guidance_text,
            started=time.time(),
            emit=lambda event_type, **payload: self._emit_harness(on_event, event_type, **payload),
        )
        run.emit(
            "route_decision",
            state=route_decision.state.value,
            reason=route_decision.reason,
            instruction=route_decision.instruction,
        )
        run.emit("plan_budget", **plan_budget)
        if image_refs:
            run.emit("image_attachments_detected", count=len(image_paths), paths=[str(path) for path in image_paths])
        return run

    def _begin_harness_turn(self, run: _HarnessRun) -> list[dict]:
        run.tool_calls_this_turn = 0
        session = run.session
        run.emit("on_state_enter", state=session.current_state.value, turn=session.turn)
        remaining_iterations = self.config.max_iterations - session.turn + 1
        if run.tool_calls_total > 0 and remaining_iterations <= 1 and not run.budget_finalization_requested:
            run.budget_finalization_requested = True
            session.current_state = OrchestratorState.FINALIZE
            finalization_prompt = (
                "Titan is on the reserved finalization iteration. Do not call tools. "
                "Return a concise Summary and Next best step using the work already completed."
            )
            self._append(run.history, Message(role=Role.SYSTEM, content=finalization_prompt))
            run.emit(
                "budget_finalization_requested",
                remaining_iterations=remaining_iterations,
                max_iterations=self.config.max_iterations,
                tool_calls_total=run.tool_calls_total,
            )
        context = self.memory.get_relevant_context(session.goal, session.context_budget)
        session.trace.append({"turn": session.turn, "state": session.current_state.value, "context_len": len(context)})
        tools = self.tools.definitions()
        if run.image_refs:
            tools = self._without_browser_navigate(tools)
        run.emit(
            "provider_request",
            iteration=session.turn,
            state=session.current_state.value,
            model=self.config.model,
            provider=self.config.provider,
            tool_calls_total=run.tool_calls_total,
            tool_calls_this_turn=run.tool_calls_this_turn,
        )
        return tools

    def _without_browser_navigate(self, tools: list[dict]) -> list[dict]:
        filtered = []
        for tool in tools:
            name = (tool.get("function", {}) or {}).get("name")
            if name == "browser_navigate":
                continue
            filtered.append(tool)
        return filtered

    def _generate_harness_turn(self, run: _HarnessRun, tools: list[dict]):
        try:
            resp = retry_call(
                lambda: self.provider.generate_with_callback(
                    self.config.model,
                    self._provider_history(run.history, run.route_decision, run.plan_budget_text, run.image_guidance_text),
                    tools,
                    on_event=lambda event_type, **event_payload: run.emit(f"provider_{event_type}", **event_payload),
                ),
                self.config.retry,
            )
            return resp, None
        except ProviderError as exc:
            elapsed_ms = int((time.time() - run.started) * 1000)
            reason = RunStopReason.ErrorRetryExhausted if exc.retryable else RunStopReason.ErrorNonRetryable
            return None, RunOutcome(
                text=f"Provider error: {exc}",
                stop=RunStopContract(
                    reason=reason,
                    iterations=run.session.turn,
                    tool_calls_total=run.tool_calls_total,
                    elapsed_ms=elapsed_ms,
                    notes=str(exc),
                ),
            )

    def _record_harness_assistant(self, run: _HarnessRun, resp) -> None:
        self._append(run.history, Message(role=Role.ASSISTANT, content=resp.text))
        if resp.text.strip():
            run.emit(
                "assistant_message",
                text=resp.text,
                state=run.session.current_state.value,
                has_tool_calls=bool(resp.tool_calls),
            )

    def _finalize_text_response(self, run: _HarnessRun, resp):
        blocked = leftover_block_note(run.task, lambda msg: self._append(run.history, msg), run.emit)
        if blocked is not None:
            return blocked
        self._maybe_distill_skill(run, resp)
        run.emit(
            "on_transition",
            from_state=run.session.current_state.value,
            to_state=OrchestratorState.FINALIZE.value,
            turn=run.session.turn,
        )
        run.emit("on_state_exit", state=run.session.current_state.value, turn=run.session.turn)
        run.session.current_state = OrchestratorState.FINALIZE
        self.session_store.checkpoint(run.session.current_state.value, run.session.turn, "finalize")
        return self._assistant_final_outcome(run, resp.text, notes=f"trace_id={run.session.trace_id}")

    def _reserved_finalization_stop(self, run: _HarnessRun, resp):
        if not run.budget_finalization_requested:
            return None
        text = resp.text.strip() or (
            "Summary:\n"
            "- Stopped cleanly on the reserved finalization pass before taking more tool actions.\n\n"
            "Next best step:\n"
            "- Continue the task with a fresh iteration budget."
        )
        run.emit("tool_batch_rejected", count=len(resp.tool_calls), reason="reserved_finalization_iteration")
        run.emit(
            "on_transition",
            from_state=run.session.current_state.value,
            to_state=OrchestratorState.FINALIZE.value,
            turn=run.session.turn,
        )
        run.emit("on_state_exit", state=run.session.current_state.value, turn=run.session.turn)
        self.session_store.checkpoint(OrchestratorState.FINALIZE.value, run.session.turn, "reserved_finalization")
        return self._assistant_final_outcome(
            run,
            text,
            notes=f"trace_id={run.session.trace_id}; reserved_finalization_iteration",
        )

    def _tool_cap_stop(self, run: _HarnessRun, resp):
        if len(resp.tool_calls) <= self.config.max_tool_calls_per_iteration:
            return None
        run.emit(
            "tool_batch_rejected",
            count=len(resp.tool_calls),
            max_tool_calls_per_iteration=self.config.max_tool_calls_per_iteration,
        )
        for index, tool_call in enumerate(resp.tool_calls, start=1):
            run.emit(
                "tool_call_rejected",
                id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                index=index,
                count=len(resp.tool_calls),
                reason="max_tool_calls_per_iteration",
            )
        elapsed_ms = int((time.time() - run.started) * 1000)
        return RunOutcome(
            text="",
            stop=RunStopContract(
                reason=RunStopReason.BudgetToolsIteration,
                iterations=run.session.turn,
                tool_calls_total=run.tool_calls_total,
                elapsed_ms=elapsed_ms,
                notes="tool calls per iteration exceeded",
            ),
        )

    def _execute_tool_batch(self, run: _HarnessRun, resp) -> list[ToolResult]:
        results: list[ToolResult] = []
        run.emit("tool_batch_started", count=len(resp.tool_calls))
        for tool_call in resp.tool_calls:
            run.tool_calls_total += 1
            run.tool_calls_this_turn += 1
            run.emit(
                "tool_call",
                id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                count=run.tool_calls_total,
                tool_calls_total=run.tool_calls_total,
                tool_calls_this_turn=run.tool_calls_this_turn,
            )
            try:
                self.policy.authorize(tool_call.name)
                result = self.tools.execute(tool_call.id, tool_call.name, tool_call.arguments)
            except PermissionError as exc:
                result = ToolResult(call_id=tool_call.id, tool_name=tool_call.name, content=str(exc), is_error=True)
            self._append(
                run.history,
                Message(
                    role=Role.TOOL,
                    content=result.content,
                    tool_call_id=result.call_id,
                    tool_name=result.tool_name,
                    is_error=result.is_error,
                ),
            )
            results.append(result)
            run.emit(
                "tool_result",
                id=result.call_id,
                name=result.tool_name,
                is_error=result.is_error,
                content=result.content,
                tool_calls_total=run.tool_calls_total,
                tool_calls_this_turn=run.tool_calls_this_turn,
            )
        self.memory.add_tool_results(results)
        return results

    def _finalize_orchestrator_state(self, run: _HarnessRun, resp):
        blocked = leftover_block_note(run.task, lambda msg: self._append(run.history, msg), run.emit)
        if blocked is not None:
            run.session.current_state = OrchestratorState.ACT
            return blocked
        self.session_store.checkpoint(run.session.current_state.value, run.session.turn, "finalize")
        return self._assistant_final_outcome(run, resp.text, notes=f"trace_id={run.session.trace_id}")

    def _advance_after_tools(self, run: _HarnessRun, resp, results: list[ToolResult]):
        failures = [result for result in results if result.is_error]
        next_state = self.supervisor.decide_next_state(
            run.session.current_state,
            has_tools=bool(resp.tool_calls),
            has_failures=bool(failures),
            final_text=resp.text,
        )
        if failures:
            run.session.recovery_count += 1
            next_state = self.recovery.classify(failures, run.session.recovery_count)
        run.emit("on_transition", from_state=run.session.current_state.value, to_state=next_state.value, turn=run.session.turn)
        run.emit("on_state_exit", state=run.session.current_state.value, turn=run.session.turn)
        run.session.current_state = next_state
        if run.session.turn % 5 == 0:
            self.session_store.checkpoint(run.session.current_state.value, run.session.turn, "periodic")
        self._maybe_distill_skill(run, resp)
        if run.session.current_state == OrchestratorState.FINALIZE:
            return self._finalize_orchestrator_state(run, resp)
        return None

    def _process_harness_response(self, run: _HarnessRun, resp):
        if not resp.tool_calls:
            return self._finalize_text_response(run, resp)
        reserved = self._reserved_finalization_stop(run, resp)
        if reserved is not None:
            return reserved
        capped = self._tool_cap_stop(run, resp)
        if capped is not None:
            return capped
        results = self._execute_tool_batch(run, resp)
        return self._advance_after_tools(run, resp, results)

    def _budget_iterations_outcome(self, run: _HarnessRun) -> RunOutcome:
        elapsed_ms = int((time.time() - run.started) * 1000)
        return RunOutcome(
            text="",
            stop=RunStopContract(
                reason=RunStopReason.BudgetIterations,
                iterations=run.session.turn,
                tool_calls_total=run.tool_calls_total,
                elapsed_ms=elapsed_ms,
                notes="max_iterations",
            ),
        )

    def run_with_callback(self, task: str, history: list[Message], on_event: Optional[Callable[[AgentEvent], None]] = None) -> RunOutcome:
        run = self._start_harness_run(task, history, on_event)
        while run.session.turn < self.config.max_iterations:
            run.session.turn += 1
            tools = self._begin_harness_turn(run)
            resp, error_outcome = self._generate_harness_turn(run, tools)
            if error_outcome is not None:
                return error_outcome
            self._record_harness_assistant(run, resp)
            outcome = self._process_harness_response(run, resp)
            if outcome is CONTINUE_LOOP:
                continue
            if outcome is not None:
                return outcome
        return self._budget_iterations_outcome(run)

    def _provider_history(
        self,
        history: list[Message],
        decision: RouteDecision,
        plan_budget_text: str = "",
        image_guidance_text: str = "",
    ) -> list[Message]:
        route_text = (
            "Titan routing decision:\n"
            f"- mode: {decision.state.value}\n"
            f"- reason: {decision.reason}\n"
            f"- instruction: {decision.instruction}"
        )
        extra_sections = [section for section in (plan_budget_text, image_guidance_text) if section]
        if extra_sections:
            route_text = f"{route_text}\n\n" + "\n\n".join(extra_sections)
        if history and history[0].role == Role.SYSTEM:
            return [Message(role=Role.SYSTEM, content=f"{history[0].content}\n\n{route_text}"), *history[1:]]
        return [Message(role=Role.SYSTEM, content=route_text), *history]

    def _append(self, history: list[Message], msg: Message) -> None:
        history.append(msg)
        self.session_store.append(msg)
