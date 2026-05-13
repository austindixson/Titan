# AI Chat

Resilient local-first TUI agent harness with checkpointing and replay.

## Presentation Framework (Proven README Pattern)

### TL;DR
Resilient local-first TUI agent harness with checkpoint/replay and terminal-native execution.

### Why this project
- Solves a concrete workflow problem with reproducible command paths.
- Prioritizes operator reliability over demo-only output.
- Structured for practical use, not just conceptual documentation.

### Quick Start
```bash
titan
```

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

### Usage Examples
```bash
titan
python -m pytest -q
```

### Architecture at a glance
- src/ — runtime loop + command routing
- tests/ — behavioral correctness checks
- scripts/ — developer utilities and reproducibility tooling

### Troubleshooting
- If `titan` command is missing, reinstall editable package inside active venv.
- If terminal rendering glitches, verify UTF-8 locale and truecolor support.

### Project status
Active development; prioritize reliability and deterministic replay behavior.


## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## Quick Start

```bash
titan
```

## Usage Examples

- Run from source (dev loop)
```bash
python -m src.titan.cli
```

- Run tests
```bash
python -m pytest -q
```

- Run CI-equivalent checks locally
```bash
python -m pytest -q tests
```

## Implementation Overview

- `src/` contains the runtime loop, command routing, and terminal UI state transitions.
- `scripts/` contains utility tooling used for development and reproducibility.
- `tests/` contains behavior checks for checkpoint/replay and core chat loop reliability.
- `pyproject.toml` defines package metadata and runtime dependencies.

## Troubleshooting

- If `titan` is not found after install, run `python -m pip install -e .` again inside the active venv.
- If rendering looks broken in terminal, verify truecolor + UTF-8 locale and resize the terminal pane.
- If a run becomes inconsistent, clear local state/checkpoints and rerun from a clean session.

## Visual Overview

![Titan visual overview](docs/assets/visual-overview-titan.svg)


## Problem
Agent workflows need reliability and recoverability, not just one-shot outputs.

## Reproducibility
```bash
python -m pip install -e .
titan
```

## Limits
Behavior depends on model/provider and local runtime constraints.

## Contributing

Contributions are welcome. Open an issue first for significant changes, then submit a focused PR with reproducible validation steps.

## License

See `LICENSE` for terms.
