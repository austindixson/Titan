"""Git checkpoints for titan undo.

Writer stores HEAD plus the dirty tree under `.titan/checkpoints/{stamp}.json`.
Restore uses `git restore` + `git apply`. It never runs `git reset --hard`,
never force-pushes, and never runs `git clean`. HEAD does not move.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


CHECKPOINTS_DIRNAME = ".titan/checkpoints"


class GitCheckpointError(RuntimeError):
    """Undo/checkpoint failure that should be reported to the user."""


@dataclass
class UndoResult:
    ok: bool
    checkpoint_id: str = ""
    message: str = ""
    missing_untracked: list[str] = field(default_factory=list)
    head: str = ""
    commands: list[list[str]] = field(default_factory=list)


def checkpoints_dir(cwd: Path | None = None) -> Path:
    root = Path(cwd) if cwd is not None else Path.cwd()
    return (root / CHECKPOINTS_DIRNAME).resolve()


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _reject_forbidden(args: list[str]) -> None:
    joined = " ".join(args)
    if args[:2] == ["reset", "--hard"] or "reset --hard" in joined:
        raise GitCheckpointError("git reset --hard is forbidden")
    if args and args[0] == "clean":
        raise GitCheckpointError("git clean is forbidden")
    if args and args[0] == "push" and ("--force" in args or "-f" in args):
        raise GitCheckpointError("force-push is forbidden")


def _run_git(
    cwd: Path,
    args: list[str],
    *,
    input_text: str | None = None,
    recorder: list[list[str]] | None = None,
) -> subprocess.CompletedProcess[str]:
    _reject_forbidden(args)
    if recorder is not None:
        recorder.append(list(args))
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        input=input_text,
        capture_output=True,
        check=False,
    )


def _git_ok(proc: subprocess.CompletedProcess[str], action: str) -> str:
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise GitCheckpointError(f"{action} failed: {err}")
    return proc.stdout


def current_head(cwd: Path, recorder: list[list[str]] | None = None) -> str:
    proc = _run_git(cwd, ["rev-parse", "HEAD"], recorder=recorder)
    sha = _git_ok(proc, "read HEAD").strip()
    if not sha:
        raise GitCheckpointError("could not read HEAD")
    return sha


def _capture_diff(cwd: Path, recorder: list[list[str]] | None = None) -> str:
    proc = _run_git(cwd, ["diff", "HEAD"], recorder=recorder)
    return _git_ok(proc, "capture dirty diff")


def _untracked_paths(cwd: Path, recorder: list[list[str]] | None = None) -> list[str]:
    proc = _run_git(cwd, ["ls-files", "--others", "--exclude-standard"], recorder=recorder)
    text = _git_ok(proc, "list untracked")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _read_untracked_entry(cwd: Path, rel: str) -> dict[str, Any]:
    path = cwd / rel
    if rel.startswith(".titan/checkpoints/") or "/.titan/checkpoints/" in rel:
        return {"path": rel, "content": None}
    try:
        return {"path": rel, "content": path.read_text()}
    except (OSError, UnicodeDecodeError):
        return {"path": rel, "content": None}


def _checkpoint_payload(cwd: Path, stamp: str, recorder: list[list[str]] | None = None) -> dict[str, Any]:
    untracked = [_read_untracked_entry(cwd, rel) for rel in _untracked_paths(cwd, recorder)]
    return {
        "id": stamp,
        "head": current_head(cwd, recorder),
        "diff": _capture_diff(cwd, recorder),
        "untracked": untracked,
    }


def git_checkpoint(cwd: Path | None = None, *, force: bool = False) -> str | None:
    """Write HEAD + dirty tree to `.titan/checkpoints/{stamp}.json`.

    Returns None under PYTEST_CURRENT_TEST unless force=True.
    """
    if not force and os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    root = Path(cwd) if cwd is not None else Path.cwd()
    stamp = _now_stamp()
    payload = _checkpoint_payload(root, stamp)
    dest_dir = checkpoints_dir(root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stamp}.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return stamp


def write_checkpoint_for_tests(cwd: Path | None = None) -> str:
    """Test helper: write a checkpoint even under PYTEST_CURRENT_TEST."""
    stamp = git_checkpoint(cwd, force=True)
    if stamp is None:
        raise GitCheckpointError("test checkpoint writer returned None")
    return stamp


def _stem_matches(path: Path, checkpoint_id: str) -> bool:
    wanted = checkpoint_id.removesuffix(".json")
    return path.stem == wanted or path.name == checkpoint_id


def list_checkpoint_paths(cwd: Path | None = None) -> list[Path]:
    dest = checkpoints_dir(cwd)
    if not dest.is_dir():
        return []
    return sorted(p for p in dest.glob("*.json") if p.is_file())


def list_checkpoints(cwd: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in list_checkpoint_paths(cwd):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        rows.append(
            {
                "id": str(data.get("id") or path.stem),
                "head": str(data.get("head") or ""),
                "path": str(path),
            }
        )
    return rows


def load_checkpoint(cwd: Path | None = None, checkpoint_id: str | None = None) -> dict[str, Any]:
    paths = list_checkpoint_paths(cwd)
    if not paths:
        raise GitCheckpointError("no git checkpoints in .titan/checkpoints/")
    if checkpoint_id:
        matches = [p for p in paths if _stem_matches(p, checkpoint_id)]
        if not matches:
            raise GitCheckpointError(f"checkpoint not found: {checkpoint_id}")
        chosen = matches[-1]
    else:
        chosen = paths[-1]
    try:
        data = json.loads(chosen.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GitCheckpointError(f"invalid checkpoint file: {chosen.name}") from exc
    if not isinstance(data, dict):
        raise GitCheckpointError(f"invalid checkpoint file: {chosen.name}")
    data.setdefault("id", chosen.stem)
    data.setdefault("head", "")
    data.setdefault("diff", "")
    data.setdefault("untracked", [])
    return data


def _restore_tracked_from_head(cwd: Path, recorder: list[list[str]]) -> None:
    proc = _run_git(
        cwd,
        ["restore", "--source=HEAD", "--staged", "--worktree", "--", "."],
        recorder=recorder,
    )
    _git_ok(proc, "git restore --source=HEAD")


def _apply_stored_diff(cwd: Path, diff: str, recorder: list[list[str]]) -> None:
    if not (diff or "").strip():
        return
    proc = _run_git(cwd, ["apply", "--whitespace=nowarn"], input_text=diff, recorder=recorder)
    _git_ok(proc, "git apply")


def _untracked_relpaths(raw: Any) -> list[tuple[str, str | None]]:
    items: list[tuple[str, str | None]] = []
    if not isinstance(raw, list):
        return items
    for entry in raw:
        if isinstance(entry, str):
            items.append((entry, None))
            continue
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path") or "").strip()
        if not path:
            continue
        content = entry.get("content")
        items.append((path, content if isinstance(content, str) else None))
    return items


def _restore_untracked(cwd: Path, raw: Any) -> list[str]:
    missing: list[str] = []
    for rel, content in _untracked_relpaths(raw):
        path = cwd / rel
        if path.exists():
            continue
        if content is None:
            missing.append(rel)
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        except OSError:
            missing.append(rel)
    return missing


def restore_checkpoint(
    cwd: Path | None = None,
    checkpoint_id: str | None = None,
    emit: Callable[..., Any] | None = None,
) -> UndoResult:
    """Restore latest or named checkpoint. HEAD does not move."""
    root = Path(cwd) if cwd is not None else Path.cwd()
    recorder: list[list[str]] = []
    data = load_checkpoint(root, checkpoint_id)
    cid = str(data.get("id") or "")
    stored_head = str(data.get("head") or "")
    live_head = current_head(root, recorder)
    if live_head != stored_head:
        msg = f"HEAD diverged from checkpoint {cid}: current={live_head} checkpoint={stored_head}"
        if emit is not None:
            emit("undo", ok=False, id=cid, error="head_mismatch")
        return UndoResult(ok=False, checkpoint_id=cid, message=msg, head=live_head, commands=recorder)

    _restore_tracked_from_head(root, recorder)
    _apply_stored_diff(root, str(data.get("diff") or ""), recorder)
    missing = _restore_untracked(root, data.get("untracked"))
    after = current_head(root, recorder)
    parts = [f"restored checkpoint {cid}"]
    if missing:
        parts.append("missing untracked: " + ", ".join(missing))
    message = "; ".join(parts)
    if emit is not None:
        emit("undo", ok=True, id=cid, missing_untracked=missing, head=after)
    return UndoResult(
        ok=True,
        checkpoint_id=cid,
        message=message,
        missing_untracked=missing,
        head=after,
        commands=recorder,
    )


def format_checkpoint_list(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no git checkpoints"
    lines = [f"{row['id']}\thead={row['head']}" for row in rows]
    return "\n".join(lines)
