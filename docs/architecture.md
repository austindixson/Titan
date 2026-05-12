# Titan Architecture

Goal: a simpler, more resilient ClaudeCode-inspired harness with deterministic stop contracts, robust tool calling, and practical terminal UX.

## Source-informed constraints (from claw-code + ferroclaw)

1. Orchestrator loop must be bounded and deterministic
- Loop increments iterations
- Hard max_iterations stop
- Stop reasons are explicit, machine-readable

2. Tool calls are first-class events
- Model output parsed into text + tool calls
- Every tool call must produce a result message (success or error)
- Tool results are fed back into next turn

3. Permission/authorization is centralized
- deny/ask/allow policy in one gate
- No direct tool bypass around the gate

4. Provider reliability wrapper
- Retry only retryable failures
- Exponential backoff + jitter
- Non-retryable fail fast

5. Session integrity
- Append-only message history per turn
- If tools were run and final text is empty, return deterministic error stop

## System modules

1) core/types.py
- Message, ToolCall, ToolResult
- RunStopReason, RunStopContract, RunOutcome
- AssistantResponse (parsed model output)

2) core/tools.py
- ToolRegistry
- Tool handlers (read_file, write_file, shell)
- Central capability + path policy enforcement

3) core/permissions.py
- PermissionMode and rules
- authorize(tool_name, args) -> allow/deny + reason

4) core/provider.py
- Provider interface
- OpenAI-compatible provider adapter
- retry_with_backoff wrapper

5) core/loop.py
- run_turn(user_input, history) orchestrator
- loop invariants:
  - I1: at least one user message
  - I2: every tool call yields a tool result
  - I3: max iteration/tool budgets enforced
  - I4: empty final text after tools => ErrorEmptyFinalAfterTools

6) core/session.py
- JSONL transcript writer/reader
- per-turn checkpoints

7) tui/app.py
- minimal terminal app
- streaming event log + input
- statuses: thinking, tool_running, complete, error

8) tests/
- unit tests for parser, retry, permissions, loop invariants
- integration test for end-to-end tool loop completion

## Execution flow

1. User input appended to history
2. Build model request (system + history + tool schema)
3. Provider call through retry wrapper
4. Parse assistant response
5. If no tools:
   - if text exists => AssistantFinal
   - if no text and tools were run => ErrorEmptyFinalAfterTools
6. If tools present:
   - enforce per-iteration and total caps
   - authorize each tool
   - execute each tool and append ToolResult message
   - continue next iteration
7. On budget/time interrupts return specific stop reason

## Stop contract

RunStopReason enum:
- AssistantFinal
- BudgetIterations
- BudgetWallClock
- BudgetToolsIteration
- BudgetToolsTotal
- ErrorNonRetryable
- ErrorRetryExhausted
- ErrorEmptyFinalAfterTools
- Interrupted

RunStopContract fields:
- reason
- iterations
- tool_calls_total
- elapsed_ms
- notes

RunOutcome fields:
- text
- stop
- usage

## Minimality decisions

Keep:
- deterministic loop
- tool/result accounting
- retry policy
- stop contracts
- simple TUI

Drop (v2 scope):
- MCP orchestration
- multi-channel gateway
- complex memory subsystems
- advanced compaction

These can be layered later without changing loop contract.