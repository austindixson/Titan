from __future__ import annotations

import os
import site
import subprocess
import sys
from pathlib import Path

import pytest
from titan.auth import resolve_openai_credentials


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(args: list[str], cwd: Path, env: dict[str, str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    subprocess_env = env.copy()
    pythonpath = [str(REPO_ROOT / "src"), site.getusersitepackages()]
    existing_pythonpath = subprocess_env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    subprocess_env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return subprocess.run(
        [sys.executable, "-m", "titan.titan_cli", *args],
        cwd=str(cwd),
        env=subprocess_env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def test_live_provider_run_smoke(tmp_path: Path):
    creds = resolve_openai_credentials()
    if not creds:
        pytest.skip("No live provider credentials in environment or auth files")

    env = os.environ.copy()
    env["TITAN_CONFIG_PATH"] = str(tmp_path / "live-config.json")

    # Keep this minimal to avoid long or expensive runs.
    setup = run_cli(["setup", "--force"], cwd=tmp_path, env=env)
    assert setup.returncode == 0, setup.stderr

    # Prefer oauth codex path when available; otherwise OPENAI_API_KEY path.
    use_codex = bool(os.getenv("OPENAI_OAUTH_TOKEN")) or (creds.source.startswith("hermes:") or creds.source.startswith("env:OPENAI_OAUTH_TOKEN"))
    provider_name = "openai-codex" if use_codex else "openai"
    run_cli(["config", "set", "provider", provider_name], cwd=tmp_path, env=env)

    run = run_cli(["run", "Reply with exactly: LIVE_SMOKE_OK"], cwd=tmp_path, env=env, timeout=180)

    assert run.returncode == 0, run.stderr
    assert "stop=AssistantFinal" in run.stdout
    assert "LIVE_SMOKE_OK" in run.stdout
