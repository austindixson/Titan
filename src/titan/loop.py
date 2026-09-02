from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import HarnessConfig
from .leftover import find_leftovers, leftover_user_note
from .permissions import PermissionError, PermissionPolicy
from .provider import Provider, ProviderError, retry_call
from .session import SessionStore
from .tools import ToolRegistry
from .types import Message, Role, RunOutcome, RunStopContract, RunStopReason
from .image_paths import local_image_references_from_text


EMPTY_TURN_RECOVERY_TEMPLATE = (
    "The previous assistant turn was empty after using tools. "
    "Continue the task without apologizing or stopping. "
    "Summarize concrete progress, identify the obstacle, choose a workaround, and either finish the task or take the next best tool action."
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
    append_message(history, Message(role=Role.SYSTEM, content=recovery_message))
    append_message(history, Message(role=Role.USER, content="Continue and finish the task using the available context and tools."))


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


class AgentLoop:
    def __init__(self, provider: Provider, tools: ToolRegistry, config: HarnessConfig, session: Optional[SessionStore] = None):
        self.provider = provider
        self.tools = tools
        self.config = config
        self.policy = PermissionPolicy(config.permission_mode)
        self.session = session

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
        def emit(event_type: str, **payload: Any) -> None:
            if on_event is not None:
                on_event(AgentEvent(type=event_type, payload=payload))

        started = time.time()
        iterations = 0
        tool_calls_total = 0
        input_tokens = 0
        output_tokens = 0
        had_tools = False
        consecutive_empty_turns = 0

        def finalize(
            text: str,
            reason: RunStopReason,
            elapsed_ms: int,
            notes: str,
            include_usage: bool = False,
        ) -> RunOutcome:
            outcome = _stop_outcome(
                text=text,
                reason=reason,
                iterations=iterations,
                tool_calls_total=tool_calls_total,
                elapsed_ms=elapsed_ms,
                notes=notes,
                usage=_usage(input_tokens, output_tokens) if include_usage else None,
            )
            _emit_completed(emit, reason, elapsed_ms, notes)
            return outcome

        def schedule_empty_turn_recovery(elapsed_ms: int) -> RunOutcome | None:
            nonlocal consecutive_empty_turns
            consecutive_empty_turns += 1
            if consecutive_empty_turns > self.config.max_consecutive_empty_turns:
                notes = (
                    "empty assistant output persisted after tool use "
                    f"for {consecutive_empty_turns} consecutive turns"
                )
                return finalize(
                    text="",
                    reason=RunStopReason.ErrorRecoveryExhausted,
                    elapsed_ms=elapsed_ms,
                    notes=notes,
                    include_usage=True,
                )

            recovery_message = _recovery_message(user_input, iterations, tool_calls_total)
            _append_recovery_prompt(self._append, history, recovery_message)
            _emit_recovery(emit, consecutive_empty_turns, iterations, tool_calls_total)
            return None

        self._append(history, Message(role=Role.USER, content=user_input))
        emit("user_message", text=user_input)
        emit("run_started")

        if not any(m.role == Role.USER for m in history):
            return finalize(
                text="",
                reason=RunStopReason.ErrorNonRetryable,
                elapsed_ms=0,
                notes="I1 no user message",
            )

        while True:
            elapsed_ms = int((time.time() - started) * 1000)
            if elapsed_ms > self.config.max_wall_clock_ms:
                return finalize(
                    text="",
                    reason=RunStopReason.BudgetWallClock,
                    elapsed_ms=elapsed_ms,
                    notes="wall clock exceeded",
                )

            iterations += 1
            emit("iteration_started", iteration=iterations, elapsed_ms=elapsed_ms)

            if iterations > self.config.max_iterations:
                return finalize(
                    text="",
                    reason=RunStopReason.BudgetIterations,
                    elapsed_ms=elapsed_ms,
                    notes="max iterations",
                )

            try:
                tool_defs = self.tools.definitions()
                local_image_refs: list[str] = []
                for m in history:
                    if m.role != Role.USER:
                        continue
                    local_image_refs.extend(local_image_references_from_text(m.content))
                if local_image_refs:
                    tool_defs = [
                        td
                        for td in tool_defs
                        if (td.get("function", {}) or {}).get("name") != "browser_navigate"
                    ]
                    emit("image_attachments_detected", count=len(local_image_refs), references=local_image_refs)

                emit("provider_request", iteration=iterations)
                resp = retry_call(lambda: self.provider.generate(self.config.model, history, tool_defs), self.config.retry)
            except ProviderError as e:
                reason = RunStopReason.ErrorRetryExhausted if e.retryable else RunStopReason.ErrorNonRetryable
                msg = f"Provider error: {str(e)}"
                emit("provider_error", error=str(e), retryable=e.retryable)
                return finalize(
                    text=msg,
                    reason=reason,
                    elapsed_ms=elapsed_ms,
                    notes=str(e),
                )

            input_tokens += resp.input_tokens
            output_tokens += resp.output_tokens
            self._append(history, Message(role=Role.ASSISTANT, content=resp.text))
            if resp.text.strip():
                consecutive_empty_turns = 0
                emit("assistant_message", text=resp.text)

            if not resp.tool_calls:
                if had_tools and not resp.text.strip():
                    maybe_outcome = schedule_empty_turn_recovery(elapsed_ms)
                    if maybe_outcome is not None:
                        return maybe_outcome
                    continue

                leftovers = find_leftovers(user_input)
                if leftovers:
                    self._append(history, Message(role=Role.USER, content=leftover_user_note(leftovers)))
                    emit("leftover_stop_blocked", leftovers=leftovers)
                    continue

                return finalize(
                    text=resp.text,
                    reason=RunStopReason.AssistantFinal,
                    elapsed_ms=elapsed_ms,
                    notes="",
                    include_usage=True,
                )

            consecutive_empty_turns = 0
            recovery_attempts = 0
            if len(resp.tool_calls) > self.config.max_tool_calls_per_iteration:
                return finalize(
                    text="",
                    reason=RunStopReason.BudgetToolsIteration,
                    elapsed_ms=elapsed_ms,
                    notes="tool calls per iteration exceeded",
                )

            had_tools = True
            for tc in resp.tool_calls:
                tool_calls_total += 1
                emit("tool_call", id=tc.id, name=tc.name, arguments=tc.arguments, count=tool_calls_total)

                if tool_calls_total > self.config.max_tool_calls_total:
                    return finalize(
                        text="",
                        reason=RunStopReason.BudgetToolsTotal,
                        elapsed_ms=elapsed_ms,
                        notes="tool calls total exceeded",
                    )

                try:
                    self.policy.authorize(tc.name)
                    tr = self.tools.execute(tc.id, tc.name, tc.arguments)
                except PermissionError as e:
                    from .types import ToolResult

                    tr = ToolResult(call_id=tc.id, tool_name=tc.name, content=str(e), is_error=True)

                self._append(
                    history,
                    Message(role=Role.TOOL, content=tr.content, tool_call_id=tr.call_id, tool_name=tr.tool_name, is_error=tr.is_error),
                )
                emit(
                    "tool_result",
                    id=tr.call_id,
                    name=tr.tool_name,
                    is_error=tr.is_error,
                    content=tr.content,
                )
