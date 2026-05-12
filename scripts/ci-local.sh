#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip >/dev/null
pip install . pytest >/dev/null

python -m pytest -q
python -m pytest -q tests/test_e2e_parity_cli.py
titan eval accuracy
echo "Titan CI smoke passed"
