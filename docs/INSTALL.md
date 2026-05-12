# Installing Titan

Titan is a Python CLI/TUI agent harness. The installer creates an isolated virtual environment under `~/.titan/venv`, installs Titan, creates `titan` and `titan-tui` launcher scripts under `~/.local/bin`, and writes the default config with chat recaps disabled. Run `titan` with no subcommand to launch the full-screen TUI.

## macOS / Linux

From GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/austindixson/Titan/main/scripts/install.sh | bash
```

From a local checkout:

```bash
./scripts/install.sh
```

If `~/.local/bin` is not on your PATH, add this to your shell profile:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Windows PowerShell

From GitHub:

```powershell
irm https://raw.githubusercontent.com/austindixson/Titan/main/scripts/install.ps1 | iex
```

From a local checkout:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

If `%USERPROFILE%\.local\bin` is not on your PATH, add it in Windows Environment Variables or run the generated `titan.cmd`/`titan-tui.cmd` directly from that folder.

## Verify

```bash
titan doctor
titan config show
titan
```


## First provider run

Run the TUI with:

```bash
titan
```

Use the Provider button / provider hotkey to cycle providers. If Titan cannot resolve a key from the environment, saved local config, Pi auth, or Hermes auth, it opens a hidden API-key input. Paste the key and press Enter; Titan saves it locally under `~/.titan/config.json` and redacts it from `titan config show`.

Multi-line pastes are compacted in the composer as `[pasted N lines]` so the UI stays readable, but the full paste is sent when you submit. Pasting or dragging local files/photos into the terminal normalizes them to absolute paths.

## Configure provider credentials

Titan does not store API keys in the repository. Provide credentials via environment variables, the hidden TUI provider prompt, or your local Titan config.

Preferred OpenAI Codex OAuth path:

```bash
export OPENAI_OAUTH_TOKEN="..."
titan config set provider openai-codex
titan config set model gpt-5.4
```

OpenAI-compatible API key path:

```bash
export OPENAI_API_KEY="..."
titan config set provider openai
titan config set model gpt-5.4-mini
```

## Installer overrides

Use a fork or alternate Git remote:

```bash
TITAN_REPO_URL=https://github.com/OWNER/REPO.git ./scripts/install.sh
```

PowerShell:

```powershell
$env:TITAN_REPO_URL='https://github.com/OWNER/REPO.git'
.\scripts\install.ps1
```

Custom install root:

```bash
TITAN_INSTALL_ROOT="$HOME/.local/share/titan" ./scripts/install.sh
```

## Uninstall

Remove the launchers and isolated venv:

```bash
rm -f ~/.local/bin/titan ~/.local/bin/titan-tui
rm -rf ~/.titan/venv
```

Windows PowerShell:

```powershell
Remove-Item "$HOME\.local\bin\titan.cmd", "$HOME\.local\bin\titan-tui.cmd" -ErrorAction SilentlyContinue
Remove-Item "$HOME\.titan\venv" -Recurse -Force -ErrorAction SilentlyContinue
```

This leaves runtime state/config under `~/.titan` unless you remove it explicitly.
