from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def run_cli(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    titan_bin = shutil.which("titan", path=env.get("PATH")) or "titan"
    cmd = [titan_bin, *args]
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)


def test_e2e_setup_config_skills_and_run(tmp_path: Path):
    env = os.environ.copy()
    cfg_path = tmp_path / "titan-config.json"
    env["TITAN_CONFIG_PATH"] = str(cfg_path)

    # isolate skill discovery to temp workspace only
    env["HOME"] = str(tmp_path)

    skills_dir = tmp_path / ".titan" / "skills" / "demo"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "SKILL.md").write_text("# Demo\nAlways use tools when possible.")

    r = run_cli(["setup", "--force"], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert cfg_path.exists()

    r = run_cli(["config", "set", "provider", "mock"], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr

    r = run_cli(["config", "get", "provider"], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "mock"

    r = run_cli(["skills", "list"], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert "demo" in r.stdout

    r = run_cli(["skills", "use", "demo"], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr

    r = run_cli(["skills", "active"], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert "demo" in r.stdout

    r = run_cli(["capability"], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert "provider=mock" in r.stdout
    assert "skills_active=demo" in r.stdout

    r = run_cli([
        "todo",
        "set",
        '[{"id":"t1","content":"ship parity","status":"in_progress"}]',
    ], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert '"saved": 1' in r.stdout

    r = run_cli(["todo", "get"], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert '"ship parity"' in r.stdout

    r = run_cli(["memory", "add", "User prefers concise updates"], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert '"saved": true' in r.stdout.lower()

    r = run_cli(["memory", "get", "--query", "concise"], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert "concise" in r.stdout.lower()

    r = run_cli(["eval", "accuracy"], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert "summary" in r.stdout.lower()
    assert "passed" in r.stdout.lower()

    r = run_cli(["run", "Say hi and finish."], cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert "Done. Tool executed." in r.stdout
    assert "stop=AssistantFinal" in r.stdout
