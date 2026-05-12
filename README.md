# AI Chat

Resilient local-first TUI agent harness with checkpointing and replay.

## Problem
Agent workflows need reliability and recoverability, not just one-shot outputs.

## Reproducibility
```bash
python -m pip install -e .
titan
```

## Limits
Behavior depends on model/provider and local runtime constraints.

## Visual Overview

![Install flow](docs/assets/install-flow.png)

![Setup flow](docs/assets/setup-flow.png)

To regenerate these visuals:

```bash
cd docs/remotion
npm install
npm run render:all
```
