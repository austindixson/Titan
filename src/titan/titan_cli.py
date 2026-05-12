from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console

from .config import (
    get_config_key,
    load_harness_config,
    resolve_config_path,
    unset_config_key,
    update_config_key,
    write_default_config,
)
from .titan import TitanHarness
from .provider import build_provider_from_config
from .session import SessionStore
from .tools import default_registry
from .types import Message, Role
from .skills import (
    build_skill_system_context,
    create_local_skill,
    delete_local_skill,
    discover_skills,
    get_active_skills,
    load_skill_text,
    unuse_skill,
    use_skill,
)
from .evals import run_accuracy_eval
from .tools import _todo_path as tool_todo_path


PARITY_REPORT_PATH = Path(".titan/parity-report.json")


def _build_harness(provider: str | None = None, model: str | None = None) -> TitanHarness:
    cfg = load_harness_config(provider_override=provider, model_override=model)
    provider_client = build_provider_from_config(cfg)
    return TitanHarness(provider=provider_client, tools=default_registry(), config=cfg, session_store=SessionStore(".titan/session.jsonl"))


def cmd_run(task: str, provider: str | None = None, model: str | None = None) -> int:
    c = Console()
    harness = _build_harness(provider=provider, model=model)
    skill_ctx = build_skill_system_context()
    sys_prompt = "You are Titan. Be resilient and tool-first."
    if skill_ctx:
        sys_prompt += "\n\nUse these loaded skills as guidance:\n" + skill_ctx
    history = [Message(role=Role.SYSTEM, content=sys_prompt)]
    out = harness.run_with_callback(task, history)
    c.print(out.text)
    c.print(f"trace_id={harness.session_store.trace_id} stop={out.stop.reason.value} iter={out.stop.iterations} tools={out.stop.tool_calls_total}")
    return 0


def cmd_replay(trace_id: str) -> int:
    c = Console()
    p = Path(".titan/trajectories") / f"{trace_id}.json"
    if not p.exists():
        c.print(f"trace not found: {trace_id}")
        return 1
    data = json.loads(p.read_text())
    c.print_json(data=json.dumps(data))
    return 0


def cmd_setup(force: bool = False) -> int:
    c = Console()
    path = resolve_config_path()
    created = write_default_config(path, force=force)
    if created:
        c.print(f"created config: {path}")
        return 0
    c.print(f"config already exists: {path} (use --force to overwrite)")
    return 0


def cmd_config_show() -> int:
    c = Console()
    path = resolve_config_path()
    if not path.exists():
        c.print(f"config not found: {path}")
        c.print("run: titan setup")
        return 1
    try:
        data = json.loads(path.read_text())
    except Exception:
        c.print(path.read_text().strip())
        return 0
    if isinstance(data, dict) and isinstance(data.get("api_keys"), dict):
        data["api_keys"] = {k: "********" for k in data["api_keys"]}
    c.print(json.dumps(data, indent=2))
    return 0


def cmd_config_set(key: str, value: str) -> int:
    c = Console()
    path = resolve_config_path()
    update_config_key(path, key, value)
    c.print(f"updated {key} in {path}")
    return 0


def cmd_config_get(key: str) -> int:
    c = Console()
    path = resolve_config_path()
    value = get_config_key(path, key)
    if value is None:
        c.print(f"key not found: {key}")
        return 1
    if key == "api_keys" or key.startswith("api_keys."):
        c.print("********" if value else "")
        return 0
    c.print(json.dumps(value) if isinstance(value, (dict, list, bool, int)) else str(value))
    return 0


def cmd_config_unset(key: str) -> int:
    c = Console()
    path = resolve_config_path()
    ok = unset_config_key(path, key)
    if not ok:
        c.print(f"key not found: {key}")
        return 1
    c.print(f"removed {key} from {path}")
    return 0


def cmd_capability() -> int:
    c = Console()
    cfg = load_harness_config()
    tool_names = ", ".join(t["function"]["name"] for t in default_registry().definitions())
    active = get_active_skills()
    c.print("Titan capability")
    c.print(f"provider={cfg.provider}")
    c.print(f"model={cfg.model}")
    c.print(f"tools={tool_names}")
    c.print(f"skills_active={','.join(active) if active else '(none)'}")
    c.print("interfaces=cli,tui")
    return 0


def cmd_skills_list() -> int:
    c = Console()
    skills = discover_skills()
    if not skills:
        c.print("no skills discovered")
        return 0
    for s in skills:
        c.print(f"{s.slug}\t{s.path}")
    return 0


def cmd_skills_active() -> int:
    c = Console()
    active = get_active_skills()
    if not active:
        c.print("(none)")
        return 0
    for s in active:
        c.print(s)
    return 0


def cmd_skills_use(slug: str) -> int:
    c = Console()
    ok = use_skill(slug)
    if not ok:
        c.print(f"skill not found: {slug}")
        return 1
    c.print(f"enabled skill: {slug}")
    return 0


def cmd_skills_unuse(slug: str) -> int:
    c = Console()
    ok = unuse_skill(slug)
    if not ok:
        c.print(f"skill not active: {slug}")
        return 1
    c.print(f"disabled skill: {slug}")
    return 0


def cmd_skills_view(slug: str) -> int:
    c = Console()
    txt = load_skill_text(slug)
    if txt is None:
        c.print(f"skill not found: {slug}")
        return 1
    c.print(txt)
    return 0


def cmd_skills_create(slug: str, content: str) -> int:
    c = Console()
    try:
        entry = create_local_skill(slug, content)
    except Exception as e:
        c.print(f"failed to create skill: {e}")
        return 1
    c.print(f"created skill: {entry.slug}\t{entry.path}")
    return 0


def cmd_skills_delete(slug: str) -> int:
    c = Console()
    ok = delete_local_skill(slug)
    if not ok:
        c.print(f"skill not found: {slug}")
        return 1
    c.print(f"deleted skill: {slug}")
    return 0


def _read_session_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def cmd_sessions_recent(limit: int = 5) -> int:
    c = Console()
    rows = _read_session_rows(Path(".titan/session.jsonl"))
    if not rows:
        c.print("no sessions yet")
        return 0

    by_trace: dict[str, dict] = {}
    for r in rows:
        tid = str(r.get("trace_id", "")).strip()
        if not tid:
            continue
        ts = int(r.get("ts", 0) or 0)
        role = str(r.get("role", ""))
        content = str(r.get("content", "")).strip()
        entry = by_trace.setdefault(tid, {"trace_id": tid, "ts": ts, "preview": ""})
        if ts >= entry["ts"]:
            entry["ts"] = ts
        if role == "user" and content and not entry["preview"]:
            entry["preview"] = content[:120]

    recent = sorted(by_trace.values(), key=lambda x: x["ts"], reverse=True)[: max(1, limit)]
    for item in recent:
        c.print(f"{item['trace_id']}\t{item['ts']}\t{item['preview']}")
    return 0


def cmd_sessions_search(query: str, limit: int = 5) -> int:
    c = Console()
    q = query.strip().lower()
    if not q:
        c.print("query cannot be empty")
        return 1

    rows = _read_session_rows(Path(".titan/session.jsonl"))
    if not rows:
        c.print("no sessions yet")
        return 0

    scores: dict[str, int] = defaultdict(int)
    previews: dict[str, str] = {}
    ts_by_trace: dict[str, int] = {}

    for r in rows:
        tid = str(r.get("trace_id", "")).strip()
        if not tid:
            continue
        content = str(r.get("content", ""))
        role = str(r.get("role", ""))
        ts = int(r.get("ts", 0) or 0)
        ts_by_trace[tid] = max(ts_by_trace.get(tid, 0), ts)
        if q in content.lower():
            scores[tid] += 1
            if tid not in previews and role == "user":
                previews[tid] = content.strip()[:120]

    ranked = sorted(scores.items(), key=lambda kv: (kv[1], ts_by_trace.get(kv[0], 0)), reverse=True)[: max(1, limit)]
    if not ranked:
        c.print("no matches")
        return 0

    for tid, score in ranked:
        c.print(f"{tid}\tscore={score}\tts={ts_by_trace.get(tid, 0)}\t{previews.get(tid, '')}")
    return 0


def cmd_eval_accuracy() -> int:
    c = Console()
    results = run_accuracy_eval()
    passed = 0
    for r in results:
        ok = "PASS" if r.passed else "FAIL"
        c.print(f"{ok}\t{r.name}\texpected={r.expected.value}\tactual={r.actual.value}")
        if r.passed:
            passed += 1
    c.print(f"summary\t{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


def cmd_parity_report(write_path: Path = PARITY_REPORT_PATH) -> int:
    c = Console()
    cfg = load_harness_config()
    tools = [t["function"]["name"] for t in default_registry().definitions()]
    skills_total = len(discover_skills())
    skills_active = get_active_skills()
    evals = run_accuracy_eval()
    eval_passed = sum(1 for e in evals if e.passed)

    report = {
        "provider": cfg.provider,
        "model": cfg.model,
        "tools": tools,
        "skills": {
            "total": skills_total,
            "active": skills_active,
        },
        "accuracy_eval": {
            "passed": eval_passed,
            "total": len(evals),
            "cases": [
                {
                    "name": e.name,
                    "expected": e.expected.value,
                    "actual": e.actual.value,
                    "passed": e.passed,
                }
                for e in evals
            ],
        },
        "commands": {
            "setup": True,
            "config": True,
            "skills": True,
            "sessions": True,
            "todo": True,
            "memory": True,
            "eval": True,
            "doctor": True,
        },
    }

    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(json.dumps(report, indent=2) + "\n")

    c.print(f"wrote parity report: {write_path}")
    c.print(f"accuracy_eval: {eval_passed}/{len(evals)} passed")
    return 0 if eval_passed == len(evals) else 1


def _read_json_obj(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def cmd_todo_get() -> int:
    c = Console()
    p = tool_todo_path(default_registry())
    data = _read_json_obj(p)
    c.print(json.dumps({"todos": data.get("todos", [])}, indent=2))
    return 0


def cmd_todo_set(todos_json: str) -> int:
    c = Console()
    try:
        raw = json.loads(todos_json)
    except Exception as e:
        c.print(f"invalid todos json: {e}")
        return 1
    if not isinstance(raw, list):
        c.print("todos json must be a list")
        return 1
    reg = default_registry()
    tr = reg.execute("todo_cli", "todo_set", {"todos": raw})
    if tr.is_error:
        c.print(tr.content)
        return 1
    c.print(tr.content)
    return 0


def cmd_memory_get(query: str | None = None) -> int:
    c = Console()
    reg = default_registry()
    args = {"query": query} if query else {}
    tr = reg.execute("mem_get", "memory_get", args)
    if tr.is_error:
        c.print(tr.content)
        return 1
    c.print(tr.content)
    return 0


def cmd_memory_add(content: str) -> int:
    c = Console()
    reg = default_registry()
    tr = reg.execute("mem_add", "memory_add", {"content": content})
    if tr.is_error:
        c.print(tr.content)
        return 1
    c.print(tr.content)
    return 0


def cmd_memory_remove(contains: str) -> int:
    c = Console()
    reg = default_registry()
    tr = reg.execute("mem_rm", "memory_remove", {"contains": contains})
    if tr.is_error:
        c.print(tr.content)
        return 1
    c.print(tr.content)
    return 0


def cmd_doctor() -> int:
    c = Console()
    problems: list[str] = []

    expected_venv = str((Path.cwd() / ".venv").resolve())
    active_venv = os.environ.get("VIRTUAL_ENV")

    c.print("Titan doctor")
    c.print(f"cwd={Path.cwd()}")
    c.print(f"python={sys.executable}")
    c.print(f"active_venv={active_venv or '(none)'}")

    if active_venv:
        active_resolved = str(Path(active_venv).resolve())
        if active_resolved != expected_venv:
            problems.append(
                f"VIRTUAL_ENV mismatch: expected {expected_venv}, got {active_resolved}"
            )

    titan_bin = shutil.which("titan")
    titan_tui_bin = shutil.which("titan-tui")
    c.print(f"which titan={titan_bin or '(not found)'}")
    c.print(f"which titan-tui={titan_tui_bin or '(optional; not found)'}")

    if not titan_bin:
        problems.append("titan not found in PATH")

    if problems:
        c.print("\nDoctor status: FAIL")
        for p in problems:
            c.print(f"- {p}")
        c.print("\nSuggested fix:")
        c.print("1) deactivate")
        c.print("2) cd /Users/ghost/Desktop/Titan")
        c.print("3) source .venv/bin/activate")
        c.print("4) hash -r")
        return 1

    c.print("\nDoctor status: OK")
    return 0


def main() -> None:
    if len(sys.argv) == 1:
        from .titan_tui import run as run_tui

        run_tui()
        return

    parser = argparse.ArgumentParser(prog="titan")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("task")
    run_p.add_argument("--provider", default=None)
    run_p.add_argument("--model", default=None)

    replay_p = sub.add_parser("replay")
    replay_p.add_argument("trace_id")

    setup_p = sub.add_parser("setup")
    setup_p.add_argument("--force", action="store_true")

    config_p = sub.add_parser("config")
    config_sub = config_p.add_subparsers(dest="config_cmd", required=True)
    config_sub.add_parser("path")
    config_sub.add_parser("show")
    set_p = config_sub.add_parser("set")
    set_p.add_argument("key")
    set_p.add_argument("value")
    get_p = config_sub.add_parser("get")
    get_p.add_argument("key")
    unset_p = config_sub.add_parser("unset")
    unset_p.add_argument("key")

    skills_p = sub.add_parser("skills")
    skills_sub = skills_p.add_subparsers(dest="skills_cmd", required=True)
    skills_sub.add_parser("list")
    skills_sub.add_parser("active")
    use_p = skills_sub.add_parser("use")
    use_p.add_argument("slug")
    unuse_p = skills_sub.add_parser("unuse")
    unuse_p.add_argument("slug")
    view_p = skills_sub.add_parser("view")
    view_p.add_argument("slug")
    create_p = skills_sub.add_parser("create")
    create_p.add_argument("slug")
    create_p.add_argument("content")
    delete_p = skills_sub.add_parser("delete")
    delete_p.add_argument("slug")

    sessions_p = sub.add_parser("sessions")
    sessions_sub = sessions_p.add_subparsers(dest="sessions_cmd", required=True)
    recent_p = sessions_sub.add_parser("recent")
    recent_p.add_argument("--limit", type=int, default=5)
    search_p = sessions_sub.add_parser("search")
    search_p.add_argument("query")
    search_p.add_argument("--limit", type=int, default=5)

    eval_p = sub.add_parser("eval")
    eval_sub = eval_p.add_subparsers(dest="eval_cmd", required=True)
    eval_sub.add_parser("accuracy")

    report_p = sub.add_parser("report")
    report_sub = report_p.add_subparsers(dest="report_cmd", required=True)
    parity_p = report_sub.add_parser("parity")
    parity_p.add_argument("--out", default=str(PARITY_REPORT_PATH))

    todo_p = sub.add_parser("todo")
    todo_sub = todo_p.add_subparsers(dest="todo_cmd", required=True)
    todo_sub.add_parser("get")
    todo_set_p = todo_sub.add_parser("set")
    todo_set_p.add_argument("todos_json")

    memory_p = sub.add_parser("memory")
    memory_sub = memory_p.add_subparsers(dest="memory_cmd", required=True)
    memory_get_p = memory_sub.add_parser("get")
    memory_get_p.add_argument("--query", default=None)
    memory_add_p = memory_sub.add_parser("add")
    memory_add_p.add_argument("content")
    memory_rm_p = memory_sub.add_parser("remove")
    memory_rm_p.add_argument("contains")

    sub.add_parser("capability")
    sub.add_parser("doctor")

    args = parser.parse_args()
    if args.cmd == "run":
        raise SystemExit(cmd_run(args.task, provider=args.provider, model=args.model))
    if args.cmd == "replay":
        raise SystemExit(cmd_replay(args.trace_id))
    if args.cmd == "setup":
        raise SystemExit(cmd_setup(force=bool(args.force)))
    if args.cmd == "config":
        if args.config_cmd == "path":
            Console().print(str(resolve_config_path()))
            raise SystemExit(0)
        if args.config_cmd == "show":
            raise SystemExit(cmd_config_show())
        if args.config_cmd == "set":
            raise SystemExit(cmd_config_set(args.key, args.value))
        if args.config_cmd == "get":
            raise SystemExit(cmd_config_get(args.key))
        if args.config_cmd == "unset":
            raise SystemExit(cmd_config_unset(args.key))
    if args.cmd == "skills":
        if args.skills_cmd == "list":
            raise SystemExit(cmd_skills_list())
        if args.skills_cmd == "active":
            raise SystemExit(cmd_skills_active())
        if args.skills_cmd == "use":
            raise SystemExit(cmd_skills_use(args.slug))
        if args.skills_cmd == "unuse":
            raise SystemExit(cmd_skills_unuse(args.slug))
        if args.skills_cmd == "view":
            raise SystemExit(cmd_skills_view(args.slug))
        if args.skills_cmd == "create":
            raise SystemExit(cmd_skills_create(args.slug, args.content))
        if args.skills_cmd == "delete":
            raise SystemExit(cmd_skills_delete(args.slug))
    if args.cmd == "sessions":
        if args.sessions_cmd == "recent":
            raise SystemExit(cmd_sessions_recent(limit=args.limit))
        if args.sessions_cmd == "search":
            raise SystemExit(cmd_sessions_search(args.query, limit=args.limit))
    if args.cmd == "eval":
        if args.eval_cmd == "accuracy":
            raise SystemExit(cmd_eval_accuracy())
    if args.cmd == "report":
        if args.report_cmd == "parity":
            raise SystemExit(cmd_parity_report(Path(args.out).expanduser().resolve()))
    if args.cmd == "todo":
        if args.todo_cmd == "get":
            raise SystemExit(cmd_todo_get())
        if args.todo_cmd == "set":
            raise SystemExit(cmd_todo_set(args.todos_json))
    if args.cmd == "memory":
        if args.memory_cmd == "get":
            raise SystemExit(cmd_memory_get(args.query))
        if args.memory_cmd == "add":
            raise SystemExit(cmd_memory_add(args.content))
        if args.memory_cmd == "remove":
            raise SystemExit(cmd_memory_remove(args.contains))
    if args.cmd == "capability":
        raise SystemExit(cmd_capability())
    if args.cmd == "doctor":
        raise SystemExit(cmd_doctor())


if __name__ == "__main__":
    main()
