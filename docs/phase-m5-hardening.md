# Titan M5 Hardening (Completed)

## Scope
Final hardening pass focused on local run reliability and operator diagnostics.

## Changes shipped
1. Added `titan doctor` command in `src/titan/titan_cli.py`
   - Verifies active venv path matches `<repo>/.venv`
   - Verifies `titan` and `titan-tui` are discoverable in PATH
   - Prints deterministic PASS/FAIL with suggested recovery steps

2. Added test coverage in `tests/test_cli_doctor.py`
   - PASS path when venv/path are healthy
   - FAIL path when `titan-tui` is missing

3. Updated run docs in `README.md`
   - Test workflow now uses `python -m pytest -q`
   - TUI startup now recommends `titan doctor` preflight

## Validation
- `titan doctor` => OK
- `python -m pytest -q` => 14 passed

## Outcome
M5 hardening now includes a preflight diagnostic to catch the exact venv/PATH drift class that previously caused `command not found: titan-tui`.
