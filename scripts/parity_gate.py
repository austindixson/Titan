from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / ".titan" / "parity-report.json"
MATRIX_PATH = ROOT / ".titan" / "parity-gap-matrix.json"


def run(cmd: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    venv_bin = str(ROOT / ".venv" / "bin")
    src = str(ROOT / "src")
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=env)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out.strip()


def status(ok: bool) -> str:
    return "green" if ok else "red"


def main() -> int:
    rc, out = run([".venv/bin/titan", "report", "parity", "--out", str(REPORT_PATH)])
    if rc != 0:
        print(out)
        return rc

    report = json.loads(REPORT_PATH.read_text())
    tools = set(report.get("tools", []))
    commands = report.get("commands", {})
    accuracy = report.get("accuracy_eval", {})

    hermes_core_expected_tools = {
        "read_file",
        "write_file",
        "shell",
        "cd",
        "todo_get",
        "todo_set",
        "memory_get",
        "memory_add",
        "memory_remove",
        "session_recent",
        "session_search",
        # Hermes parity-critical surfaces not yet present in Titan:
        "web_search",
        "browser_navigate",
        "delegate_task",
        "cronjob",
    }

    missing_tools = sorted(hermes_core_expected_tools - tools)

    # deterministic e2e must pass
    rc_e2e, out_e2e = run([".venv/bin/python", "-m", "pytest", "-q", "tests/test_e2e_parity_cli.py"])
    e2e_det_ok = rc_e2e == 0

    # live e2e parity row is green only when actually executed and passed (not skipped)
    rc_live, out_live = run([".venv/bin/python", "-m", "pytest", "-q", "-r", "s", "tests/test_e2e_live_provider_optional.py"])
    low_live = out_live.lower()
    live_skipped = "skipped" in low_live or "skip" in low_live
    e2e_live_ok = rc_live == 0 and not live_skipped

    rows = [
        {
            "area": "setup/config",
            "feature": "setup+config CLI surface",
            "status": status(bool(commands.get("setup")) and bool(commands.get("config"))),
            "evidence": "titan report parity -> commands.setup/config",
        },
        {
            "area": "capability",
            "feature": "Hermes-core tool breadth",
            "status": status(len(missing_tools) == 0),
            "evidence": f"missing tools={missing_tools}",
        },
        {
            "area": "accuracy",
            "feature": "deterministic stop-contract eval",
            "status": status(int(accuracy.get("passed", 0)) == int(accuracy.get("total", -1)) and int(accuracy.get("total", 0)) > 0),
            "evidence": f"accuracy_eval={accuracy.get('passed')}/{accuracy.get('total')}",
        },
        {
            "area": "extensibility",
            "feature": "skills lifecycle CLI",
            "status": status(bool(commands.get("skills"))),
            "evidence": "titan report parity -> commands.skills",
        },
        {
            "area": "ux",
            "feature": "interactive TUI binary available",
            "status": status(run([".venv/bin/titan", "doctor"])[0] == 0),
            "evidence": "titan doctor",
        },
        {
            "area": "tested-e2e",
            "feature": "deterministic e2e CLI",
            "status": status(e2e_det_ok),
            "evidence": out_e2e.splitlines()[-1] if out_e2e else "",
        },
        {
            "area": "tested-e2e",
            "feature": "live-provider e2e (must run, no skip)",
            "status": status(e2e_live_ok),
            "evidence": out_live.splitlines()[-1] if out_live else "",
        },
    ]

    all_green = all(r["status"] == "green" for r in rows)
    matrix = {
        "goal": "Titan full parity with Hermes agent",
        "all_green": all_green,
        "rows": rows,
    }
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_PATH.write_text(json.dumps(matrix, indent=2) + "\n")

    print(f"wrote matrix: {MATRIX_PATH}")
    for r in rows:
        print(f"{r['status'].upper():5} | {r['area']:<11} | {r['feature']} | {r['evidence']}")
    print(f"gate={'OPEN' if all_green else 'CLOSED'}")
    return 0 if all_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
