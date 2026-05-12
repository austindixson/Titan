# Titan vs Hermes Parity Gap Matrix

Generated from live checks on this machine.

- Matrix JSON: `/Users/ghost/Desktop/Titan/.titan/parity-gap-matrix.json`
- Gate runner: `/Users/ghost/Desktop/Titan/scripts/parity_gate.py`
- Gate rule: completion is allowed only when every row is `green`.

## Current Matrix

| Area | Feature | Status | Evidence |
|---|---|---|---|
| setup/config | setup+config CLI surface | green | `titan report parity -> commands.setup/config` |
| capability | Hermes-core tool breadth | red | missing tools: `browser_navigate`, `cronjob`, `delegate_task`, `web_search` |
| accuracy | deterministic stop-contract eval | green | `accuracy_eval=4/4` |
| extensibility | skills lifecycle CLI | green | `titan report parity -> commands.skills` |
| ux | interactive TUI binary available | green | `titan doctor` |
| tested-e2e | deterministic e2e CLI | green | `tests/test_e2e_parity_cli.py` passed |
| tested-e2e | live-provider e2e (must run, no skip) | red | optional live test skipped due missing credentials |

## Gate Verdict

- gate status: `CLOSED`
- reason: not all rows are green

## How to re-run

```bash
cd /Users/ghost/Desktop/Titan
. .venv/bin/activate
python scripts/parity_gate.py
```

Exit code contract:
- `0` => all rows green (gate open)
- `1` => one or more rows red (gate closed)
