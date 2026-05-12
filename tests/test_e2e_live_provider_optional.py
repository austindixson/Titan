from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from titan.auth import resolve_openai_credentials


def test_live_provider_run_smoke(tmp_path: Path):
    creds = resolve_openai_credentials()
    if not creds:
        pytest.skip("No live provider credentials in environment or auth files")

    env = os.environ.copy()
    env["TITAN_CONFIG_PATH"] = str(tmp_path / "live-config.json")

    # Keep this minimal to avoid long or expensive runs.
    setup = subprocess.run(
        ["titan", "setup", "--force"],
        cwd=str(tmp_path),
        env=env,
        text=True,
        capture_output=True,
    )
    assert setup.returncode == 0, setup.stderr

    # Prefer oauth codex path when available; otherwise OPENAI_API_KEY path.
    use_codex = bool(os.getenv("OPENAI_OAUTH_TOKEN")) or (creds.source.startswith("hermes:") or creds.source.startswith("env:OPENAI_OAUTH_TOKEN"))
    provider_name = "openai-codex" if use_codex else "openai"
    subprocess.run(["titan", "config", "set", "provider", provider_name], cwd=str(tmp_path), env=env, text=True, capture_output=True)

    run = subprocess.run(
        ["titan", "run", "Reply with exactly: LIVE_SMOKE_OK"],
        cwd=str(tmp_path),
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert run.returncode == 0, run.stderr
    assert "stop=AssistantFinal" in run.stdout
    assert "LIVE_SMOKE_OK" in run.stdout
