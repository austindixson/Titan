"""Harness-owned verify-after-edit.

After a mutating tool batch (write_file / edit_file succeeded), Titan detects
and runs the project's tests. Failures are the agent's problem; do not ask.

Evidence is a provider-legal assistant+tool pair (K11/K19). A bare role=tool
row 400s on Grok and Codex. Execution goes through ToolRegistry.execute so
blocked_shell_reason still applies.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from .types import Message, Role, ToolCall, ToolResult


MUTATING_TOOLS = frozenset({"write_file", "edit_file"})
VERIFY_TIMEOUT_S = 120
VERIFY_CALL_PREFIX = "verify_"

_SKIP = object()

_PROJECT_MARKERS: tuple[tuple[str, str], ...] = (
    ("pyproject.toml", "pytest"),
    ("pytest.ini", "pytest"),
    ("setup.cfg", "pytest"),
    ("setup.py", "pytest"),
    ("tox.ini", "pytest"),
    ("package.json", "npm test"),
    ("Cargo.toml", "cargo test"),
    ("go.mod", "go test ./..."),
    ("Gemfile", "bundle exec rake test"),
    ("pom.xml", "mvn test"),
    ("build.gradle", "gradle test"),
    ("build.gradle.kts", "gradle test"),
    ("Makefile", "make test"),
    ("GNUmakefile", "make test"),
)


def normalize_shell_command(command: str) -> str:
    return " ".join((command or "").split())


def verify_call_id() -> str:
    return f"{VERIFY_CALL_PREFIX}{uuid.uuid4().hex}"


def _resolve_mutated_path(raw: str, cwd: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path


def _mutator_path(call: ToolCall) -> str:
    args = call.arguments or {}
    raw = args.get("path") or args.get("file_path") or ""
    return str(raw).strip()


def succeeded_mutator_paths(executed: Iterable[tuple[ToolCall, ToolResult]]) -> list[str]:
    paths: list[str] = []
    for call, result in executed:
        if call.name not in MUTATING_TOOLS or result.is_error:
            continue
        path = _mutator_path(call)
        if path:
            paths.append(path)
    return paths


def last_shell_command(executed: Iterable[tuple[ToolCall, ToolResult]]) -> str | None:
    last: str | None = None
    for call, _result in executed:
        if call.name != "shell":
            continue
        last = str((call.arguments or {}).get("command", "") or "")
    return last


def _load_verify_json(directory: Path) -> dict[str, Any] | None:
    path = directory / ".titan" / "verify.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _marker_command(directory: Path) -> str | None:
    for name, command in _PROJECT_MARKERS:
        if (directory / name).exists():
            return command
    return None


def _command_or_skip_at(directory: Path) -> object | None:
    cfg = _load_verify_json(directory)
    if cfg is not None:
        if cfg.get("skip") is True:
            return _SKIP
        command = cfg.get("command")
        if isinstance(command, str) and command.strip():
            return command.strip()
    return _marker_command(directory)


def _walk_parents(start: Path) -> list[Path]:
    current = start
    seen: set[Path] = set()
    out: list[Path] = []
    while True:
        resolved = current.resolve()
        if resolved in seen:
            break
        seen.add(resolved)
        out.append(resolved)
        parent = resolved.parent
        if parent == resolved:
            break
        current = parent
    return out


def detect_verify_command(mutated_paths: list[str], cwd: Path) -> str | None:
    """Parent-walk from mutated files. `.titan/verify.json` command wins; skip:true skips."""
    root = cwd.resolve()
    for raw in mutated_paths:
        path = _resolve_mutated_path(raw, root)
        start = path if path.is_dir() else path.parent
        for directory in _walk_parents(start):
            found = _command_or_skip_at(directory)
            if found is _SKIP:
                return None
            if isinstance(found, str) and found:
                return found
    return None


def should_skip_verify(
    *,
    reserved_finalization: bool,
    last_iteration: bool,
    command: str | None,
    last_shell: str | None,
) -> bool:
    if reserved_finalization or last_iteration:
        return True
    if not command:
        return True
    if last_shell is None:
        return False
    return normalize_shell_command(last_shell) == normalize_shell_command(command)


def _verify_assistant_message(call_id: str, command: str) -> Message:
    return Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="shell",
                arguments={"command": command, "timeout": VERIFY_TIMEOUT_S},
            )
        ],
    )


def _verify_tool_message(result: ToolResult) -> Message:
    return Message(
        role=Role.TOOL,
        content=result.content,
        tool_call_id=result.call_id,
        tool_name=result.tool_name,
        is_error=result.is_error,
    )


def append_verify_evidence(
    append: Callable[[Message], None],
    call_id: str,
    command: str,
    result: ToolResult,
) -> None:
    append(_verify_assistant_message(call_id, command))
    append(_verify_tool_message(result))


def run_verify_command(execute: Callable[[str, str, dict[str, Any]], ToolResult], command: str) -> tuple[str, ToolResult]:
    call_id = verify_call_id()
    result = execute(call_id, "shell", {"command": command, "timeout": VERIFY_TIMEOUT_S})
    return call_id, result


def maybe_verify_after_edit(
    *,
    reserved_finalization: bool,
    last_iteration: bool,
    cwd: Path,
    executed: list[tuple[ToolCall, ToolResult]],
    execute: Callable[[str, str, dict[str, Any]], ToolResult],
    append: Callable[[Message], None],
    emit: Callable[..., Any],
) -> None:
    paths = succeeded_mutator_paths(executed)
    if not paths:
        return
    command = detect_verify_command(paths, cwd)
    last_shell = last_shell_command(executed)
    if should_skip_verify(
        reserved_finalization=reserved_finalization,
        last_iteration=last_iteration,
        command=command,
        last_shell=last_shell,
    ):
        return
    assert command is not None
    call_id, result = run_verify_command(execute, command)
    append_verify_evidence(append, call_id, command, result)
    emit(
        "verify",
        call_id=call_id,
        command=command,
        is_error=result.is_error,
        content=result.content,
    )
