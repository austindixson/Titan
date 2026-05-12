# Product Requirements Document (PDR) — Titan

## Problem
Current harness behavior is unreliable:
- stops before task completion
- avoids tool use when tools are required
- weak error semantics

## Product objective
Deliver a resilient local agent harness that reliably completes tool-using tasks with deterministic stop semantics.

## Users
Primary: you (developer/operator) running local CLI/TUI harness sessions.

## Core requirements

R1. Deterministic completion contract
- Every run ends with RunOutcome + RunStopContract.

R2. Tool-first execution behavior
- If task implies actionable work, loop should attempt tools before giving up.

R3. Resilient provider calling
- Retry retryable provider failures.
- Exhaustion is explicit in stop reason.

R4. Strong loop invariants
- No user message => invariant stop.
- Each tool call maps to a result message.
- Empty final text after tool execution => explicit error stop.

R5. Practical local tools
- read_file
- write_file
- shell command

R6. Terminal UX
- simple interactive TUI
- event feed (rounds, tools, errors)
- no fragile/complex UI dependencies

R7. Testability
- Unit + integration tests covering loop reliability and failure modes.

## Non-goals (v2)
- full Claude feature parity
- multi-tenant service/gateway
- remote orchestrator swarm

## Success criteria
- 100% pass on included tests
- Integration scenario: model asks tool call, tool executes, model finalizes answer successfully
- Deterministic stop reason for every failure mode covered by tests

## Risks
- Provider response shape variance
Mitigation: robust parser for text/tool_calls and strict validation.

- Tool misuse/danger
Mitigation: centralized permission/path policy and default-safe modes.

## Milestones
M1: docs/spec/tasks complete
M2: core loop + tools + provider + types implemented
M3: TUI wired
M4: tests green
M5: final hardening pass