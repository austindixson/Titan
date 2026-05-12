from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    is_error: bool = False


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    content: str
    is_error: bool = False


@dataclass
class AssistantResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class RunStopReason(str, Enum):
    AssistantFinal = "AssistantFinal"
    BudgetIterations = "BudgetIterations"
    BudgetWallClock = "BudgetWallClock"
    BudgetToolsIteration = "BudgetToolsIteration"
    BudgetToolsTotal = "BudgetToolsTotal"
    ErrorNonRetryable = "ErrorNonRetryable"
    ErrorRetryExhausted = "ErrorRetryExhausted"
    ErrorRecoveryExhausted = "ErrorRecoveryExhausted"
    Interrupted = "Interrupted"


@dataclass
class RunStopContract:
    reason: RunStopReason
    iterations: int
    tool_calls_total: int
    elapsed_ms: int
    notes: str = ""


@dataclass
class RunOutcome:
    text: str
    stop: RunStopContract
    usage: dict[str, int] = field(default_factory=dict)
