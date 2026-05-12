# Titan -> Hermes Parity Plan

Status date: 2026-05-07

## Current parity snapshot

- setup/config: partial (implemented `titan setup`, `titan config path/show/set`)
- capability breadth: partial (read_file/write_file/shell/cd only)
- accuracy/reliability: partial (deterministic stops + tests green)
- extensibility: partial (provider abstraction + tool registry, no skills/plugin system parity)
- UX: partial (CLI + TUI, responsive fixes in progress)
- tested e2e: partial (unit/integration + local smoke, no full Hermes-equivalent e2e matrix)

## Gaps to full parity

1) Setup/config parity
- Add config get/unset/edit and schema validation
- Add migration/versioned config

2) Capability parity
- Expand tool catalog and contracts (todo, memory, browser/web/search/delegation equivalents)
- Add provider/model management parity and discovery

3) Accuracy parity
- Add benchmark suite against Hermes tasks and stop-contract compliance checks
- Add regression corpus for tool-call/retry/parser edge cases

4) Extensibility parity
- Add skills lifecycle (list/view/use/unuse/create/patch)
- Add plugin/toolset loading contracts

5) UX parity
- Add slash command palette, stateful plan pane, richer progress telemetry
- Complete responsive layout + compact mode quality gates

6) E2E parity
- Add scripted e2e flows for setup->run->tool->final across providers
- Add CI job with deterministic replay and artifact capture

## Completed in this phase

- Added config system and CLI:
  - `titan setup`
  - `titan config path`
  - `titan config show`
  - `titan config set <key> <value>`
- Added `titan capability` command
- Wired all entrypoints to config loader
- Added tests: `tests/test_config_cli.py`

## Verification

- `python -m pytest -q` => passing
- `titan setup --force` => creates config
- `titan config set model gpt-5.4-mini` => persists
- `titan capability` => reports effective runtime configuration
