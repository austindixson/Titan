# Feature Spec — Titan

## 1) Core loop

Inputs:
- user_input: str
- history: list[Message]

Outputs:
- RunOutcome

Behavior:
- add user message
- iterate up to max_iterations
- call provider with tools and history
- parse assistant output into text/tool_calls/usage
- execute tool calls through registry (permission gate enforced)
- append assistant + tool messages back to history
- finish only with explicit stop contract

## 2) Stop reason mapping

- max iterations reached -> BudgetIterations
- wall clock exceeded -> BudgetWallClock
- tool calls in one round exceed cap -> BudgetToolsIteration
- total tool calls exceed cap -> BudgetToolsTotal
- provider non-retryable error -> ErrorNonRetryable
- provider retries exhausted -> ErrorRetryExhausted
- tool(s) executed but final assistant text empty -> ErrorEmptyFinalAfterTools
- normal text completion -> AssistantFinal

## 3) Provider contract

Provider.generate(request) -> AssistantResponse

AssistantResponse:
- text: str
- tool_calls: list[ToolCall]
- usage: dict(input_tokens, output_tokens)

OpenAI-compatible parsing:
- supports `tool_calls` and text in `choices[0].message`
- handles missing/null arrays safely

Retry policy:
- max_retries configurable
- exponential backoff + jitter
- retry on transport errors, 429, 5xx

## 4) Tool runtime

Built-ins:
- read_file(path, offset=1, limit=500)
- write_file(path, content)
- shell(command, timeout=60)

Rules:
- all tool execution goes through registry.execute()
- permission check before handler call
- failures return ToolResult(is_error=True)

## 5) Permission policy

Modes:
- prompt (default deny for dangerous ops in non-interactive mode)
- allow

v2 behavior:
- allow read_file by default
- write_file/shell require allow mode

## 6) Session persistence

- JSONL append after every message
- transcript contains timestamp, role, blocks

## 7) TUI

- simple event log and input prompt
- Enter sends, Ctrl+C exits
- displays:
  - round start
  - tool start/result
  - final output
  - stop reason

## 8) Configuration

config.yaml fields:
- model
- api_base
- api_key_env
- max_iterations
- max_wall_clock_ms
- max_tool_calls_per_iteration
- max_tool_calls_total
- retry.max_retries, retry.base_delay_ms, retry.max_delay_ms
- permission_mode

## 9) Test matrix

Unit:
- parser handles text-only, tools-only, mixed
- retry retries only retryable errors
- permission gate blocks/permits correctly
- stop reason mapping

Integration:
- mock provider: tool then final text path
- mock provider: endless tools capped by iteration/tool caps
- empty final after tools -> ErrorEmptyFinalAfterTools