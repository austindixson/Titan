from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .config import HarnessConfig
from .image_paths import candidate_image_paths_from_text, local_image_references_from_text
from .loop import AgentEvent, AgentLoop
from .provider import Provider
from .session import SessionStore
from .tools import ToolRegistry
from .types import Message, Role, RunOutcome, RunStopReason, ToolResult


TITAN_SYSTEM_PROMPT = "You are Titan. Be resilient and tool-first."
_ROUTING_MARKER = "Titan routing decision:"


class OrchestratorState(str, Enum):
    PLAN = "PLAN"
    ACT = "ACT"
    REFLECT = "REFLECT"
    RECOVER = "RECOVER"
    FINALIZE = "FINALIZE"


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
                OrchestratorState.PLAN,
                "delegation cue detected; stay in-process",
                "Start with a brief plan only when it reduces risk, then execute in-process. "
                "Do not call delegate_task. Prefer tool use for code/file/system work and verify changes.",
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


def _system_prompt_base(content: str) -> str:
    marker = f"\n\n{_ROUTING_MARKER}"
    if marker in content:
        return content.split(marker, 1)[0]
    if content.startswith(_ROUTING_MARKER):
        return TITAN_SYSTEM_PROMPT
    return content


def _route_system_text(decision: RouteDecision, plan_budget_text: str, image_guidance_text: str) -> str:
    route_text = (
        f"{_ROUTING_MARKER}\n"
        f"- mode: {decision.state.value}\n"
        f"- reason: {decision.reason}\n"
        f"- instruction: {decision.instruction}"
    )
    extra_sections = [section for section in (plan_budget_text, image_guidance_text) if section]
    if extra_sections:
        route_text = f"{route_text}\n\n" + "\n\n".join(extra_sections)
    return route_text


def _merge_routing_into_system(
    history: list[Message],
    decision: RouteDecision,
    plan_budget_text: str,
    image_guidance_text: str,
) -> None:
    route_text = _route_system_text(decision, plan_budget_text, image_guidance_text)
    if history and history[0].role == Role.SYSTEM:
        base = _system_prompt_base(history[0].content)
        history[0] = Message(role=Role.SYSTEM, content=f"{base}\n\n{route_text}")
        return
    history.insert(0, Message(role=Role.SYSTEM, content=f"{TITAN_SYSTEM_PROMPT}\n\n{route_text}"))


@dataclass
class _FacadeRun:
    task: str
    history: list[Message]
    route_decision: RouteDecision
    plan_budget: dict[str, Any]
    emit: Callable[..., None]
    loop: AgentLoop
    on_engine_event: Callable[[AgentEvent], None]


class TitanHarness:
    """Product-path facade. AgentLoop is the only orchestrator; this class does not append USER."""

    def __init__(self, provider: Provider, tools: ToolRegistry, config: HarnessConfig, session_store: Optional[SessionStore] = None):
        self.provider = provider
        self.tools = tools
        self.config = config
        self.session_store = session_store or SessionStore(".titan/session.jsonl")
        self.event_bus = EventBus()
        self.supervisor = Supervisor()
        self.router = IntentRouter()
        self.memory = MemoryManager()
        self.recovery = RecoveryEngine()
        self.learning = LearningLoop()
        self.interrupt_flag = False
        self._active_loop: Optional[AgentLoop] = None

    def request_interrupt(self) -> None:
        self.interrupt_flag = True
        if self._active_loop is not None:
            self._active_loop.request_interrupt()

    def _emit_harness(self, on_event: Optional[Callable[[AgentEvent], None]], event_type: str, **payload: Any) -> None:
        self.event_bus.emit(event_type, payload)
        if on_event:
            on_event(AgentEvent(type=event_type, payload=payload))

    def _bridge_engine_event(self, on_event: Optional[Callable[[AgentEvent], None]], state: str, event: AgentEvent) -> None:
        if event.type in {"provider_request", "assistant_message"}:
            event.payload.setdefault("state", state)
        self.event_bus.emit(event.type, event.payload)
        if on_event:
            on_event(event)

    def _route_and_budget(self, task: str) -> tuple[RouteDecision, dict[str, Any], str, str]:
        decision = self.router.decide(task)
        plan_budget = _iteration_budget_plan(self.config, decision)
        plan_budget_text = _format_iteration_budget_plan(plan_budget)
        image_paths = candidate_image_paths_from_text(task)
        image_refs = local_image_references_from_text(task)
        image_guidance = _format_image_attachment_guidance(image_paths, image_refs)
        return decision, plan_budget, plan_budget_text, image_guidance

    def _prepare_facade_run(
        self,
        task: str,
        history: list[Message],
        on_event: Optional[Callable[[AgentEvent], None]],
    ) -> _FacadeRun:
        decision, plan_budget, plan_budget_text, image_guidance = self._route_and_budget(task)
        _merge_routing_into_system(history, decision, plan_budget_text, image_guidance)
        self.session_store.checkpoint(decision.state.value, 0, f"run_start:{decision.reason}")
        emit = lambda event_type, **payload: self._emit_harness(on_event, event_type, **payload)
        emit(
            "route_decision",
            state=decision.state.value,
            reason=decision.reason,
            instruction=decision.instruction,
        )
        emit("plan_budget", **plan_budget)
        emit("on_state_enter", state=decision.state.value, turn=0)
        loop = AgentLoop(self.provider, self.tools, self.config, session=self.session_store)
        loop.interrupt_flag = self.interrupt_flag
        self._active_loop = loop

        def on_engine_event(event: AgentEvent) -> None:
            self._bridge_engine_event(on_event, decision.state.value, event)

        return _FacadeRun(
            task=task,
            history=history,
            route_decision=decision,
            plan_budget=plan_budget,
            emit=emit,
            loop=loop,
            on_engine_event=on_engine_event,
        )

    def _maybe_distill_skill(self, run: _FacadeRun, outcome: RunOutcome) -> None:
        if not _should_distill_skill(self.config, outcome.stop.tool_calls_total, outcome.stop.iterations):
            return
        score = 1.0 if outcome.text.strip() else 0.5
        skill_path = self.learning.distill_skill(self.session_store.trace_id, [], success_score=score)
        run.emit("on_skill_created", path=str(skill_path))

    def _finish_facade_run(self, run: _FacadeRun, outcome: RunOutcome) -> RunOutcome:
        if outcome.stop.reason == RunStopReason.AssistantFinal:
            run.emit(
                "on_transition",
                from_state=run.route_decision.state.value,
                to_state=OrchestratorState.FINALIZE.value,
                turn=outcome.stop.iterations,
            )
            run.emit("on_state_exit", state=run.route_decision.state.value, turn=outcome.stop.iterations)
            self.session_store.checkpoint(OrchestratorState.FINALIZE.value, outcome.stop.iterations, "finalize")
            self._maybe_distill_skill(run, outcome)
        return outcome

    def run_with_callback(self, task: str, history: list[Message], on_event: Optional[Callable[[AgentEvent], None]] = None) -> RunOutcome:
        run = self._prepare_facade_run(task, history, on_event)
        try:
            outcome = run.loop.run_with_callback(task, history, on_event=run.on_engine_event)
            return self._finish_facade_run(run, outcome)
        finally:
            self._active_loop = None
