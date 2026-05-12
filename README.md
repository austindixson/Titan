# Titan

Resilient local-first agent harness with Titan state-machine orchestration, checkpointing, replay, provider streaming, and a full-screen Textual TUI.


## Philosophy

Titan is for builders who want to talk conversationally and get working software back: finished MVPs, quick iterations, robust tests, and fewer walls of process output. It is intentionally local-first and harness-first, but not bloated. The chat stays primary; traceability is available when troubleshooting, not forced into every response.

I built Titan for myself because coding-agent harnesses were becoming too heavy: too many panes, too much ceremony, too much output, and not enough finished product. Titan keeps the loop resilient while preserving a simple conversation: ask for the thing, watch concise progress, inspect trace/diff when needed, and keep shipping.

Titan is for:
- Non-expert coders who want to build real products through conversation.
- Solo builders who care about MVP speed and working test suites.
- Agent/harness tinkerers who still need transparent traces when debugging.
- People who prefer concise chat over three pages of tool logs.

Titan is not trying to hide execution. It keeps trace and diff tabs because accountability matters. It just keeps that detail out of the way until you need it.

## What is implemented

- Deterministic stop contracts (`RunStopReason`, `RunStopContract`, `RunOutcome`)
- Bounded orchestrator loop with resilience invariants
- Tool runtime with centralized execution (`read_file`, `write_file`, `shell`)
- Permission gate (allow vs prompt/restricted)
- Provider retry wrapper (retryable vs non-retryable)
- JSONL session persistence
- Minimal terminal CLI and canonical `titan` TUI launcher
- Pasted multi-line input is compacted in the composer as `[pasted N lines]` while the full content is sent on submit
- Pasted or terminal-dropped local files/photos normalize to absolute paths
- Provider selection prompts for a hidden API-key input when no key is already saved/resolved
- Test suite covering core reliability behavior

## Docs

- `docs/INSTALL.md`
- `docs/architecture.md`
- `docs/pdr.md`
- `docs/feature-spec.md`
- `docs/task-list.md`

## One-shot install

macOS / Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/austindixson/Titan/main/scripts/install.sh | bash
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/austindixson/Titan/main/scripts/install.ps1 | iex
```

Local checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

See `docs/INSTALL.md` for PATH, credential, override, and uninstall details.

## Run tests

python -m pip install pytest
python -m pytest -q

Expected:
all tests passed

## Run CLI (real provider, OAuth-first)

titan setup
titan config show
titan config get model
titan config set model gpt-5.4-mini
titan config unset api_base
titan skills list
titan skills create demo-local "# Demo Local\nUse this workflow."
titan skills use demo-local
titan skills active
titan sessions recent --limit 5
titan sessions search "layout"
titan eval accuracy
titan todo set '[{"id":"t1","content":"ship parity","status":"in_progress"}]'
titan todo get
titan memory add "User prefers concise updates"
titan memory get --query concise
titan report parity --out .titan/parity-report.json
titan capability
titan run "Summarize current directory"

TUI slash shortcuts:
- /help
- /skills
- /active
- /use <slug>
- /unuse <slug>
- /todo
- /memory [query]
- /trace

Auth resolution order:
1) Provider-specific environment variable, for example `OPENAI_OAUTH_TOKEN`, `OPENAI_API_KEY`, `XAI_API_KEY`, or `OPENROUTER_API_KEY`
2) `~/.pi/agent/auth.json` when present
3) `~/.hermes/auth.json` for OpenAI/OpenAI Codex
4) Saved local `~/.titan/config.json` API key entered through the TUI provider prompt

Saved API keys are local-only and redacted from `titan config show`.


## Security / repository hygiene

- Runtime state lives in `~/.titan` or local `.titan/` and is ignored by git.
- Credentials are loaded from environment variables such as `OPENAI_OAUTH_TOKEN` or `OPENAI_API_KEY`; do not commit `.env` files or tokens.
- Chat recaps are disabled by default (`chat_recaps_enabled=false`) so the TUI does not append generic system-prompt-like next-step boilerplate.

## E2E parity smoke (local, deterministic)

python -m pytest -q tests/test_e2e_parity_cli.py

This test exercises setup/config/skills/capability/run end-to-end using provider=mock.

## Optional live-provider smoke

python -m pytest -q tests/test_e2e_live_provider_optional.py

- Auto-skips when no OPENAI_OAUTH_TOKEN / OPENAI_API_KEY is present.
- When credentials exist, validates a minimal real-provider AssistantFinal flow.

## CI parity test matrix

- GitHub Actions workflow: `.github/workflows/ci.yml`
- Python versions: 3.10, 3.11, 3.12
- Runs:
  - full test suite (`python -m pytest -q`)
  - deterministic e2e parity smoke (`tests/test_e2e_parity_cli.py`)

Local CI script:

./scripts/ci-local.sh

## Run full-screen TUI

titan doctor
titan

TUI controls:
- Enter: send
- Up/Down: scroll when composer is empty (otherwise edit input)
- Shift+Up/Down: fine scroll
- PgUp/PgDn: fast scroll
- Ctrl+Home/Ctrl+End: top/bottom
- Ctrl+D: switch top panel between Trace and Diff
- Ctrl+F: focus input
- Ctrl+L: clear transcript
- Paste multi-line text: composer shows `[pasted N lines]`; full paste is submitted
- Paste/drag local files or photos into the terminal: Titan normalizes them to paths

## Run minimal TUI demo (mock)

python -c "from titan.tui import demo_tui; demo_tui()"

## Notes about original ferroclaw TUI reuse

I took the same minimalist interaction pattern from your current TUI direction:
- transcript-style lines
- simple status words (`thinking`, `ready`, `error`)
- no heavy panes/chrome

But this v2 keeps UI intentionally thin while prioritizing loop reliability first.
