from __future__ import annotations

import json
from dataclasses import dataclass

from .config import get_config_key, resolve_config_path, unset_config_key, update_config_key
from .skills import discover_skills, get_active_skills, unuse_skill, use_skill
from .tools import default_registry


@dataclass
class SlashResult:
    handled: bool
    message: str
    is_error: bool = False


def execute_slash_command(text: str) -> SlashResult:
    raw = text.strip()
    if not raw.startswith("/"):
        return SlashResult(handled=False, message="")

    parts = raw[1:].split()
    cmd = (parts[0] if parts else "").lower()
    args = parts[1:]

    if cmd in {"help", "h"}:
        return SlashResult(
            handled=True,
            message=(
                "commands: /help, /skills, /active, /use <slug>, /unuse <slug>, "
                "/todo, /memory [query], /config [get|set|unset] [key] [value], /trace"
            ),
        )

    if cmd == "skills":
        skills = discover_skills()
        if not skills:
            return SlashResult(handled=True, message="no skills discovered")
        preview = ", ".join(s.slug for s in skills[:20])
        extra = "" if len(skills) <= 20 else f" (+{len(skills)-20} more)"
        return SlashResult(handled=True, message=f"skills: {preview}{extra}")

    if cmd == "active":
        active = get_active_skills()
        return SlashResult(handled=True, message=(", ".join(active) if active else "(none)"))

    if cmd == "use":
        if not args:
            return SlashResult(handled=True, message="usage: /use <slug>", is_error=True)
        slug = args[0]
        ok = use_skill(slug)
        if not ok:
            return SlashResult(handled=True, message=f"skill not found: {slug}", is_error=True)
        return SlashResult(handled=True, message=f"enabled skill: {slug}")

    if cmd == "unuse":
        if not args:
            return SlashResult(handled=True, message="usage: /unuse <slug>", is_error=True)
        slug = args[0]
        ok = unuse_skill(slug)
        if not ok:
            return SlashResult(handled=True, message=f"skill not active: {slug}", is_error=True)
        return SlashResult(handled=True, message=f"disabled skill: {slug}")

    if cmd == "todo":
        reg = default_registry()
        tr = reg.execute("slash_todo", "todo_get", {})
        if tr.is_error:
            return SlashResult(handled=True, message=tr.content, is_error=True)
        data = json.loads(tr.content)
        todos = data.get("todos", []) if isinstance(data, dict) else []
        if not todos:
            return SlashResult(handled=True, message="todos: (none)")
        lines = [f"- [{t.get('status','pending')}] {t.get('id','?')}: {t.get('content','')}" for t in todos[:10]]
        if len(todos) > 10:
            lines.append(f"... +{len(todos)-10} more")
        return SlashResult(handled=True, message="todos:\n" + "\n".join(lines))

    if cmd == "memory":
        q = " ".join(args).strip()
        reg = default_registry()
        tr = reg.execute("slash_mem", "memory_get", ({"query": q} if q else {}))
        if tr.is_error:
            return SlashResult(handled=True, message=tr.content, is_error=True)
        data = json.loads(tr.content)
        entries = data.get("entries", []) if isinstance(data, dict) else []
        if not entries:
            return SlashResult(handled=True, message="memory: (none)")
        lines = [f"- {e}" for e in entries[:10]]
        if len(entries) > 10:
            lines.append(f"... +{len(entries)-10} more")
        return SlashResult(handled=True, message="memory:\n" + "\n".join(lines))

    if cmd == "config":
        path = resolve_config_path()
        subcmd = (args[0].lower() if args else "show")
        if subcmd == "show":
            recap = get_config_key(path, "chat_recaps_enabled")
            if recap is None:
                recap = False
            learning = get_config_key(path, "learning_enabled")
            if learning is None:
                learning = False
            return SlashResult(
                handled=True,
                message=(
                    f"config: {path}\n"
                    f"chat_recaps_enabled={str(recap).lower()}\n"
                    f"learning_enabled={str(learning).lower()}"
                ),
            )
        if subcmd == "get":
            if len(args) < 2:
                return SlashResult(handled=True, message="usage: /config get <key>", is_error=True)
            value = get_config_key(path, args[1])
            return SlashResult(handled=True, message=("null" if value is None else str(value)))
        if subcmd == "set":
            if len(args) < 3:
                return SlashResult(handled=True, message="usage: /config set <key> <value>", is_error=True)
            update_config_key(path, args[1], " ".join(args[2:]))
            return SlashResult(handled=True, message=f"config set: {args[1]}")
        if subcmd == "unset":
            if len(args) < 2:
                return SlashResult(handled=True, message="usage: /config unset <key>", is_error=True)
            ok = unset_config_key(path, args[1])
            if not ok:
                return SlashResult(handled=True, message=f"config key not found: {args[1]}", is_error=True)
            return SlashResult(handled=True, message=f"config unset: {args[1]}")
        return SlashResult(handled=True, message="usage: /config [show|get|set|unset] [key] [value]", is_error=True)

    if cmd == "trace":
        return SlashResult(handled=True, message="trace-toggle")

    return SlashResult(handled=True, message=f"unknown command: /{cmd}", is_error=True)
