from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import HarnessConfig
from .leftover import CONTINUE_LOOP, INTERNAL_NOTE_PREFIX, leftover_block_note
from .permissions import PermissionError, PermissionPolicy
from .provider import Provider, ProviderError, retry_call
from .session import SessionStore
from .tools import ToolRegistry
from .types import Message, Role, RunOutcome, RunStopContract, RunStopReason, ToolResult
from .image_paths import local_image_references_from_text


EMPTY_TURN_RECOVERY_TEMPLATE = (
    "The previous assistant turn was empty after using tools. "
    "Continue the task without apologizing or stopping. "
    "Summarize concrete progress, identify the obstacle, choose a workaround, and either finish the task or take the next best tool action."
)

RESERVED_FINALIZATION_NOTE = (
    f"{INTERNAL_NOTE_PREFIX}"
    "Titan is on the reserved finalization iteration. Do not call tools. "
    "Return a concise Summary and Next best step using the work already completed."
)

RESERVED_FINALIZATION_FALLBACK = (
    "Summary:\n"
    "- Stopped cleanly on the reserved finalization pass before taking more tool actions.\n\n"
    "Next best step:\n"
    "- Continue the task with a fresh iteration budget."
)


def _diagnostic_text(reason: RunStopReason, notes: str) -> str:
    reason_map = {
        RunStopReason.BudgetIterations: "Titan stopped after reaching the iteration budget before it could finish.",
        RunStopReason.BudgetWallClock: "Titan stopped after reaching the wall-clock budget before it could finish.",
        RunStopReason.BudgetToolsIteration: "Titan requested too many tool calls in a single iteration.",
        RunStopReason.BudgetToolsTotal: "Titan stopped after reaching the total tool-call budget before it could finish.",
        RunStopReason.ErrorRecoveryExhausted: "Titan hit repeated empty assistant turns after tool use and could not recover further.",
        RunStopReason.ErrorRetryExhausted: "Titan stopped after provider retries were exhausted.",
        RunStopReason.ErrorNonRetryable: "Titan stopped due to a non-retryable provider error.",
        RunStopReason.Interrupted: "Titan was interrupted before completion.",
    }
    base = reason_map.get(reason, "Titan stopped before completion.")
    if notes:
        return f"{base} Details: {notes}"
    return base


def _recovery_message(user_input: str, iterations: int, tool_calls_total: int) -> str:
    return (
        f"Original task: {user_input}\n"
        f"Recovery attempt after an empty assistant turn. Iteration={iterations}, tool_calls={tool_calls_total}.\n"
        f"{EMPTY_TURN_RECOVERY_TEMPLATE}"
    )


def _non_empty_text(text: str, reason: RunStopReason, notes: str) -> str:
    stripped = text.strip()
    return stripped if stripped else _diagnostic_text(reason, notes)


def _stop_outcome(
    text: str,
    reason: RunStopReason,
    iterations: int,
    tool_calls_total: int,
    elapsed_ms: int,
    notes: str,
    usage: dict[str, int] | None = None,
) -> RunOutcome:
    return RunOutcome(
        text=_non_empty_text(text, reason, notes),
        stop=RunStopContract(reason, iterations, tool_calls_total, elapsed_ms, notes),
        usage=usage or {},
    )


def _append_recovery_prompt(append_message: Callable[[list[Message], Message], None], history: list[Message], recovery_message: str) -> None:
    append_message(
        history,
        Message(
            role=Role.USER,
            content=(
                f"{INTERNAL_NOTE_PREFIX}{recovery_message}\n"
                "Continue and finish the task using the available context and tools."
            ),
        ),
    )


def _usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _emit_recovery(
    emit: Callable[[str, Any], None],
    consecutive_empty_turns: int,
    iterations: int,
    tool_calls_total: int,
) -> None:
    emit(
        "empty_turn_recovery",
        consecutive_empty_turns=consecutive_empty_turns,
        iteration=iterations,
        tool_calls_total=tool_calls_total,
    )


def _emit_completed(
    emit: Callable[[str, Any], None], reason: RunStopReason, elapsed_ms: int, notes: str = ""
) -> None:
    emit("run_completed", reason=reason.value, elapsed_ms=elapsed_ms, notes=notes)


@dataclass
class AgentEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class _LoopCounters:
    iterations: int = 0
    tool_calls_total: int = 0
    tool_calls_this_turn: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    had_tools: bool = False
    consecutive_empty_turns: int = 0
    budget_finalization_requested: bool = False


def _emit_agent_event(on_event: Optional[Callable[[AgentEvent], None]], event_type: str, **payload: Any) -> None:
    if on_event is not None:
        on_event(AgentEvent(type=event_type, payload=payload))


def _loop_finalize(
    emit: Callable[..., None],
    counters: _LoopCounters,
    text: str,
    reason: RunStopReason,
    elapsed_ms: int,
    notes: str,
    include_usage: bool = False,
) -> RunOutcome:
    usage = _usage(counters.input_tokens, counters.output_tokens) if include_usage else None
    outcome = _stop_outcome(
        text=text,
        reason=reason,
        iterations=counters.iterations,
        tool_calls_total=counters.tool_calls_total,
        elapsed_ms=elapsed_ms,
        notes=notes,
        usage=usage,
    )
    _emit_completed(emit, reason, elapsed_ms, notes)
    return outcome


def _schedule_empty_turn_recovery(
    loop: "AgentLoop",
    user_input: str,
    history: list[Message],
    emit: Callable[..., None],
    counters: _LoopCounters,
    elapsed_ms: int,
) -> RunOutcome | None:
    counters.consecutive_empty_turns += 1
    if counters.consecutive_empty_turns > loop.config.max_consecutive_empty_turns:
        notes = (
            "empty assistant output persisted after tool use "
            f"for {counters.consecutive_empty_turns} consecutive turns"
        )
        return _loop_finalize(
            emit,
            counters,
            text="",
            reason=RunStopReason.ErrorRecoveryExhausted,
            elapsed_ms=elapsed_ms,
            notes=notes,
            include_usage=True,
        )

    recovery_message = _recovery_message(user_input, counters.iterations, counters.tool_calls_total)
    _append_recovery_prompt(loop._append, history, recovery_message)
    _emit_recovery(emit, counters.consecutive_empty_turns, counters.iterations, counters.tool_calls_total)
    return None


def _tool_defs_for_history(tool_defs: list[dict], history: list[Message], emit: Callable[..., None]) -> list[dict]:
    local_image_refs: list[str] = []
    for message in history:
        if message.role != Role.USER:
            continue
        local_image_refs.extend(local_image_references_from_text(message.content))
    if not local_image_refs:
        return tool_defs
    emit("image_attachments_detected", count=len(local_image_refs), references=local_image_refs)
    filtered = []
    for tool_def in tool_defs:
        name = (tool_def.get("function", {}) or {}).get("name")
        if name == "browser_navigate":
            continue
        filtered.append(tool_def)
    return filtered


def _invoke_provider(loop: "AgentLoop", history: list[Message], tool_defs: list[dict], emit: Callable[..., None]):
    def on_provider_event(event_type: str, **payload: Any) -> None:
        emit(f"provider_{event_type}", **payload)

    callback = getattr(loop.provider, "generate_with_callback", None)
    if callback is not None:
        return callback(loop.config.model, history, tool_defs, on_event=on_provider_event)
    return loop.provider.generate(loop.config.model, history, tool_defs)


def _loop_call_provider(
    loop: "AgentLoop",
    history: list[Message],
    emit: Callable[..., None],
    counters: _LoopCounters,
    elapsed_ms: int,
):
    try:
        tool_defs = _tool_defs_for_history(loop.tools.definitions(), history, emit)
        emit(
            "provider_request",
            iteration=counters.iterations,
            model=loop.config.model,
            provider=loop.config.provider,
            tool_calls_total=counters.tool_calls_total,
            tool_calls_this_turn=counters.tool_calls_this_turn,
        )
        resp = retry_call(lambda: _invoke_provider(loop, history, tool_defs, emit), loop.config.retry)
        return resp, None
    except ProviderError as exc:
        reason = RunStopReason.ErrorRetryExhausted if exc.retryable else RunStopReason.ErrorNonRetryable
        emit("provider_error", error=str(exc), retryable=exc.retryable)
        outcome = _loop_finalize(
            emit,
            counters,
            text=f"Provider error: {str(exc)}",
            reason=reason,
            elapsed_ms=elapsed_ms,
            notes=str(exc),
        )
        return None, outcome


def _handle_empty_tool_calls(
    loop: "AgentLoop",
    user_input: str,
    history: list[Message],
    emit: Callable[..., None],
    counters: _LoopCounters,
    resp,
    elapsed_ms: int,
):
    if resp.tool_calls:
        return None
    if counters.had_tools and not resp.text.strip():
        recovered = _schedule_empty_turn_recovery(loop, user_input, history, emit, counters, elapsed_ms)
        if recovered is not None:
            return recovered
        return CONTINUE_LOOP
    blocked = leftover_block_note(user_input, lambda msg: loop._append(history, msg), emit)
    if blocked is not None:
        return blocked
    return _loop_finalize(
        emit,
        counters,
        text=resp.text,
        reason=RunStopReason.AssistantFinal,
        elapsed_ms=elapsed_ms,
        notes="",
        include_usage=True,
    )


def _reject_tool_batch_over_iteration_cap(
    loop: "AgentLoop",
    emit: Callable[..., None],
    counters: _LoopCounters,
    resp,
    elapsed_ms: int,
) -> RunOutcome | None:
    limit = loop.config.max_tool_calls_per_iteration
    if len(resp.tool_calls) <= limit:
        return None
    emit("tool_batch_rejected", count=len(resp.tool_calls), max_tool_calls_per_iteration=limit)
    for index, tool_call in enumerate(resp.tool_calls, start=1):
        emit(
            "tool_call_rejected",
            id=tool_call.id,
            name=tool_call.name,
            arguments=tool_call.arguments,
            index=index,
            count=len(resp.tool_calls),
            reason="max_tool_calls_per_iteration",
        )
    return _loop_finalize(
        emit,
        counters,
        text="",
        reason=RunStopReason.BudgetToolsIteration,
        elapsed_ms=elapsed_ms,
        notes="tool calls per iteration exceeded",
    )


def _execute_authorized_tool(loop: "AgentLoop", tool_call) -> ToolResult:
    try:
        loop.policy.authorize(tool_call.name)
        return loop.tools.execute(tool_call.id, tool_call.name, tool_call.arguments)
    except PermissionError as exc:
        return ToolResult(call_id=tool_call.id, tool_name=tool_call.name, content=str(exc), is_error=True)


def _loop_run_tool_calls(
    loop: "AgentLoop",
    history: list[Message],
    emit: Callable[..., None],
    counters: _LoopCounters,
    resp,
    elapsed_ms: int,
):
    counters.consecutive_empty_turns = 0
    capped = _reject_tool_batch_over_iteration_cap(loop, emit, counters, resp, elapsed_ms)
    if capped is not None:
        return capped

    counters.had_tools = True
    emit("tool_batch_started", count=len(resp.tool_calls))
    for tool_call in resp.tool_calls:
        counters.tool_calls_total += 1
        counters.tool_calls_this_turn += 1
        emit(
            "tool_call",
            id=tool_call.id,
            name=tool_call.name,
            arguments=tool_call.arguments,
            count=counters.tool_calls_total,
            tool_calls_total=counters.tool_calls_total,
            tool_calls_this_turn=counters.tool_calls_this_turn,
        )
        if counters.tool_calls_total > loop.config.max_tool_calls_total:
            return _loop_finalize(
                emit,
                counters,
                text="",
                reason=RunStopReason.BudgetToolsTotal,
                elapsed_ms=elapsed_ms,
                notes="tool calls total exceeded",
            )
        result = _execute_authorized_tool(loop, tool_call)
        loop._append(
            history,
            Message(
                role=Role.TOOL,
                content=result.content,
                tool_call_id=result.call_id,
                tool_name=result.tool_name,
                is_error=result.is_error,
            ),
        )
        emit(
            "tool_result",
            id=result.call_id,
            name=result.tool_name,
            is_error=result.is_error,
            content=result.content,
            tool_calls_total=counters.tool_calls_total,
            tool_calls_this_turn=counters.tool_calls_this_turn,
        )
    return None


def _maybe_request_reserved_finalization(
    loop: "AgentLoop",
    history: list[Message],
    emit: Callable[..., None],
    counters: _LoopCounters,
) -> None:
    if counters.budget_finalization_requested:
        return
    remaining = loop.config.max_iterations - counters.iterations + 1
    if counters.tool_calls_total <= 0 or remaining > 1:
        return
    counters.budget_finalization_requested = True
    loop._append(history, Message(role=Role.USER, content=RESERVED_FINALIZATION_NOTE))
    emit(
        "budget_finalization_requested",
        remaining_iterations=remaining,
        max_iterations=loop.config.max_iterations,
        tool_calls_total=counters.tool_calls_total,
    )


def _reserved_finalization_reject(
    emit: Callable[..., None],
    counters: _LoopCounters,
    resp,
    elapsed_ms: int,
) -> RunOutcome | None:
    if not counters.budget_finalization_requested or not resp.tool_calls:
        return None
    emit("tool_batch_rejected", count=len(resp.tool_calls), reason="reserved_finalization_iteration")
    text = resp.text.strip() or RESERVED_FINALIZATION_FALLBACK
    return _loop_finalize(
        emit,
        counters,
        text=text,
        reason=RunStopReason.AssistantFinal,
        elapsed_ms=elapsed_ms,
        notes="reserved_finalization_iteration",
        include_usage=True,
    )


def _loop_user_invariant(
    emit: Callable[..., None],
    counters: _LoopCounters,
    history: list[Message],
) -> RunOutcome | None:
    if any(message.role == Role.USER for message in history):
        return None
    return _loop_finalize(
        emit,
        counters,
        text="",
        reason=RunStopReason.ErrorNonRetryable,
        elapsed_ms=0,
        notes="I1 no user message",
    )


def _loop_begin_iteration(
    loop: "AgentLoop",
    history: list[Message],
    emit: Callable[..., None],
    counters: _LoopCounters,
    elapsed_ms: int,
) -> RunOutcome | None:
    if loop.interrupt_flag:
        return _loop_finalize(
            emit,
            counters,
            text="",
            reason=RunStopReason.Interrupted,
            elapsed_ms=elapsed_ms,
            notes="interrupt_flag",
        )
    if elapsed_ms > loop.config.max_wall_clock_ms:
        return _loop_finalize(
            emit,
            counters,
            text="",
            reason=RunStopReason.BudgetWallClock,
            elapsed_ms=elapsed_ms,
            notes="wall clock exceeded",
        )
    counters.iterations += 1
    counters.tool_calls_this_turn = 0
    emit("iteration_started", iteration=counters.iterations, elapsed_ms=elapsed_ms)
    if counters.iterations > loop.config.max_iterations:
        return _loop_finalize(
            emit,
            counters,
            text="",
            reason=RunStopReason.BudgetIterations,
            elapsed_ms=elapsed_ms,
            notes="max iterations",
        )
    _maybe_request_reserved_finalization(loop, history, emit, counters)
    return None


def _record_assistant_turn(
    loop: "AgentLoop",
    history: list[Message],
    emit: Callable[..., None],
    counters: _LoopCounters,
    resp,
) -> None:
    counters.input_tokens += resp.input_tokens
    counters.output_tokens += resp.output_tokens
    loop._append(history, Message(role=Role.ASSISTANT, content=resp.text))
    if resp.text.strip():
        counters.consecutive_empty_turns = 0
        emit("assistant_message", text=resp.text, has_tool_calls=bool(resp.tool_calls))


class AgentLoop:
    def __init__(self, provider: Provider, tools: ToolRegistry, config: HarnessConfig, session: Optional[SessionStore] = None):
        self.provider = provider
        self.tools = tools
        self.config = config
        self.policy = PermissionPolicy(config.permission_mode)
        self.session = session
        self.interrupt_flag = False

    def request_interrupt(self) -> None:
        self.interrupt_flag = True

    def _append(self, history: list[Message], msg: Message):
        history.append(msg)
        if self.session:
            self.session.append(msg)

    def run(self, user_input: str, history: list[Message]) -> RunOutcome:
        return self.run_with_callback(user_input, history, on_event=None)

    def run_with_callback(
        self,
        user_input: str,
        history: list[Message],
        on_event: Optional[Callable[[AgentEvent], None]] = None,
    ) -> RunOutcome:
        emit = lambda event_type, **payload: _emit_agent_event(on_event, event_type, **payload)
        started = time.time()
        counters = _LoopCounters()
        self._append(history, Message(role=Role.USER, content=user_input))
        emit("user_message", text=user_input)
        emit("run_started")
        invariant = _loop_user_invariant(emit, counters, history)
        if invariant is not None:
            return invariant

        while True:
            elapsed_ms = int((time.time() - started) * 1000)
            pre = _loop_begin_iteration(self, history, emit, counters, elapsed_ms)
            if pre is not None:
                return pre

            resp, error_outcome = _loop_call_provider(self, history, emit, counters, elapsed_ms)
            if error_outcome is not None:
                return error_outcome

            _record_assistant_turn(self, history, emit, counters, resp)

            empty_outcome = _handle_empty_tool_calls(self, user_input, history, emit, counters, resp, elapsed_ms)
            if empty_outcome is CONTINUE_LOOP:
                continue
            if empty_outcome is not None:
                return empty_outcome

            reserved = _reserved_finalization_reject(emit, counters, resp, elapsed_ms)
            if reserved is not None:
                return reserved

            tool_outcome = _loop_run_tool_calls(self, history, emit, counters, resp, elapsed_ms)
            if tool_outcome is not None:
                return tool_outcome
