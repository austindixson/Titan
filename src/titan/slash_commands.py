from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .config import get_config_key, resolve_config_path, unset_config_key, update_config_key
from .git_checkpoint import GitCheckpointError, format_checkpoint_list, list_checkpoints, restore_checkpoint
from .skills import discover_skills, get_active_skills, unuse_skill, use_skill
from .tools import default_registry


@dataclass
class SlashResult:
    handled: bool
    message: str
    is_error: bool = False


def _slash_help(_args: list[str], **_kwargs: object) -> SlashResult:
    return SlashResult(
        handled=True,
        message=(
            "commands: /help, /skills, /active, /use <slug>, /unuse <slug>, "
            "/todo, /memory [query], /config [get|set|unset] [key] [value], /trace, "
            "/undo [checkpoint_id]"
        ),
    )


def _slash_skills(_args: list[str], **_kwargs: object) -> SlashResult:
    skills = discover_skills()
    if not skills:
        return SlashResult(handled=True, message="no skills discovered")
    preview = ", ".join(s.slug for s in skills[:20])
    extra = "" if len(skills) <= 20 else f" (+{len(skills)-20} more)"
    return SlashResult(handled=True, message=f"skills: {preview}{extra}")


def _slash_active(_args: list[str], **_kwargs: object) -> SlashResult:
    active = get_active_skills()
    return SlashResult(handled=True, message=(", ".join(active) if active else "(none)"))


def _slash_use(args: list[str], **_kwargs: object) -> SlashResult:
    if not args:
        return SlashResult(handled=True, message="usage: /use <slug>", is_error=True)
    slug = args[0]
    ok = use_skill(slug)
    if not ok:
        return SlashResult(handled=True, message=f"skill not found: {slug}", is_error=True)
    return SlashResult(handled=True, message=f"enabled skill: {slug}")


def _slash_unuse(args: list[str], **_kwargs: object) -> SlashResult:
    if not args:
        return SlashResult(handled=True, message="usage: /unuse <slug>", is_error=True)
    slug = args[0]
    ok = unuse_skill(slug)
    if not ok:
        return SlashResult(handled=True, message=f"skill not active: {slug}", is_error=True)
    return SlashResult(handled=True, message=f"disabled skill: {slug}")


def _slash_todo(_args: list[str], **_kwargs: object) -> SlashResult:
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


def _slash_memory(args: list[str], **_kwargs: object) -> SlashResult:
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


def _slash_config_show() -> SlashResult:
    path = resolve_config_path()
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


def _slash_config(args: list[str], **_kwargs: object) -> SlashResult:
    path = resolve_config_path()
    subcmd = (args[0].lower() if args else "show")
    if subcmd == "show":
        return _slash_config_show()
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


def _slash_trace(_args: list[str], **_kwargs: object) -> SlashResult:
    return SlashResult(handled=True, message="trace-toggle")


def _slash_undo(args: list[str], *, run_pending: bool = False, **_kwargs: object) -> SlashResult:
    if run_pending:
        return SlashResult(
            handled=True,
            message="undo refused while a run is pending",
            is_error=True,
        )
    checkpoint_id = args[0] if args else None
    try:
        result = restore_checkpoint(checkpoint_id=checkpoint_id)
    except GitCheckpointError as exc:
        return SlashResult(handled=True, message=str(exc), is_error=True)
    return SlashResult(handled=True, message=result.message, is_error=not result.ok)


def _slash_checkpoints(_args: list[str], **_kwargs: object) -> SlashResult:
    return SlashResult(handled=True, message=format_checkpoint_list(list_checkpoints()))


_SLASH_HANDLERS: dict[str, Callable[..., SlashResult]] = {
    "help": _slash_help,
    "h": _slash_help,
    "skills": _slash_skills,
    "active": _slash_active,
    "use": _slash_use,
    "unuse": _slash_unuse,
    "todo": _slash_todo,
    "memory": _slash_memory,
    "config": _slash_config,
    "trace": _slash_trace,
    "undo": _slash_undo,
    "checkpoints": _slash_checkpoints,
}


def _parse_slash(text: str) -> tuple[str, list[str]] | None:
    raw = text.strip()
    if not raw.startswith("/"):
        return None
    parts = raw[1:].split()
    cmd = (parts[0] if parts else "").lower()
    return cmd, parts[1:]


def execute_slash_command(text: str, *, run_pending: bool = False) -> SlashResult:
    parsed = _parse_slash(text)
    if parsed is None:
        return SlashResult(handled=False, message="")
    cmd, args = parsed
    handler = _SLASH_HANDLERS.get(cmd)
    if handler is None:
        return SlashResult(handled=True, message=f"unknown command: /{cmd}", is_error=True)
    return handler(args, run_pending=run_pending)
