# Titan

Resilient local-first TUI agent harness with checkpoint/replay and provider switching.

## One-command install

macOS / Linux:
```bash
curl -fsSL https://raw.githubusercontent.com/austindixson/Titan/main/scripts/install.sh | bash
```

Windows PowerShell:
```powershell
irm https://raw.githubusercontent.com/austindixson/Titan/main/scripts/install.ps1 | iex
```

## First 60 seconds after download

If you already cloned Titan, run:
```bash
./scripts/install.sh
```

Then verify + launch:
```bash
titan doctor
titan config show
titan
```

## What users should expect in the interface

When Titan opens, you will see:
- top pane: Trace / Diff tabs
- main pane: chat output
- control row buttons: Stop, Provider, Clear, Trace, Quit
- composer at bottom: "Type task and press Enter"

Provider UX:
- click `Provider: <current>` or press `Ctrl+P` to cycle providers
- if the next provider has no key, Titan opens a hidden key input
- paste key, press Enter, Titan saves it to `~/.titan/config.json` (redacted in `titan config show`)

## Provider setup (real examples)

OpenAI Codex OAuth:
```bash
export OPENAI_OAUTH_TOKEN="..."
titan config set provider openai-codex
titan config set model gpt-5.4
titan
```

OpenAI API key:
```bash
export OPENAI_API_KEY="..."
titan config set provider openai
titan config set model gpt-5.4-mini
titan
```

xAI (Grok-compatible OpenAI API):
```bash
export XAI_API_KEY="..."
titan config set provider xai
titan config set model grok-4
titan
```

## Supported providers (built-in)

- openai
- openai-codex
- openrouter
- xai
- groq
- cerebras
- deepseek
- mistral
- zai
- moonshotai

## How to try it quickly (real flow)

1) Start Titan:
```bash
titan
```

2) Ask a concrete prompt:
```text
Summarize this repository and propose 3 improvements.
```

3) Switch provider live (`Ctrl+P`) and ask the same prompt again.

4) Compare behavior in Trace tab (request/tool/retry flow).

## Troubleshooting users will actually hit

`titan: command not found`
```bash
export PATH="$HOME/.local/bin:$PATH"
source ~/.zshrc  # or ~/.bashrc
```

Provider key not detected
- ensure env var name matches provider
- or use in-app hidden prompt when switching provider

Provider/model mismatch error
```bash
titan config set provider openai
titan config set model gpt-5.4-mini
```

UI rendering looks broken
- use a terminal with UTF-8 + truecolor
- widen terminal window and relaunch

## Architecture at a glance

- `src/titan/titan_tui.py` — full-screen TUI, provider cycling, hidden API-key prompt
- `src/titan/auth.py` — provider credential resolution (env, local auth files)
- `src/titan/titan_cli.py` — `titan config ...`, `run`, `doctor`, and command routing
- `tests/` — behavior and reliability checks

## Visual Overview

![Titan visual overview](docs/assets/visual-overview-titan.svg)

## Contributing

Open an issue first for significant changes, then submit a focused PR with reproducible validation steps.

## License

See `LICENSE`.
