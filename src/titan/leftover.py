"""Harness-owned leftover-stop predicate.

Before AssistantFinal, Titan cheaply probes for named leftover paths/packages
extracted from the task (rename/move/delete sources). If any still exist in the
workspace, stop is blocked and a user note is injected so the loop continues.

Allowlisted legacy names (legacy_*) and explicit keep-paths must remain and
never block stop. Rename targets are not leftovers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

INTERNAL_NOTE_PREFIX = "Titan internal note:\n"

# Historical artifacts that MUST keep the old name and must not block stop.
ALLOWLIST_RELATIVE_PATHS = frozenset(
    {
        "tests/data/legacy_blueledger.json",
        "migrations/legacy_blueledger.sql",
    }
)

# Generic / keep-path tokens that are never treated as leftovers.
_SKIP_NAMES = frozenset(
    {
        "a",
        "an",
        "app",
        "bin",
        "data",
        "dir",
        "directory",
        "docs",
        "file",
        "files",
        "folder",
        "it",
        "lib",
        "module",
        "modules",
        "name",
        "package",
        "packages",
        "path",
        "python",
        "src",
        "test",
        "tests",
        "that",
        "the",
        "them",
        "this",
    }
)

_SKIP_WALK_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".titan",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)

_NAME = r"[\w][\w./-]*"

_RENAME = re.compile(
    rf"\brename(?:\s+the)?(?:\s+python)?(?:\s+(?:package|module|directory|dir|folder|path))?"
    rf"(?:\s+from)?\s+"
    rf"[`'\"]?(?P<source>{_NAME})[`'\"]?\s+to\s+[`'\"]?(?P<target>{_NAME})[`'\"]?",
    re.IGNORECASE,
)

_MOVE = re.compile(
    rf"\b(?:move|mv)\s+[`'\"]?(?P<source>{_NAME})[`'\"]?\s+(?:to|into)\s+[`'\"]?(?P<target>{_NAME})[`'\"]?",
    re.IGNORECASE,
)

_DELETE = re.compile(
    rf"\b(?:delete|remove|rm)\s+(?:the\s+)?(?:old\s+)?(?:leftover\s+)?"
    rf"(?:(?:python\s+)?(?:package|module|directory|dir|folder|path|file)\s+)?"
    rf"[`'\"]?(?P<source>{_NAME})[`'\"]?",
    re.IGNORECASE,
)

_KEEP_PATH = re.compile(
    r"(?:tests|migrations|docs)/[A-Za-z0-9_./-]+",
    re.IGNORECASE,
)


def _normalize_name(raw: str) -> str:
    return raw.strip("`'\" \t\n").rstrip("/")


def _basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def is_allowlisted_name(name: str) -> bool:
    normalized = _normalize_name(name)
    if not normalized:
        return False
    if normalized in ALLOWLIST_RELATIVE_PATHS:
        return True
    base = _basename(normalized)
    if base.startswith("legacy_"):
        return True
    return False


def is_allowlisted_path(workspace: Path, path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        rel = path.name
    if rel in ALLOWLIST_RELATIVE_PATHS:
        return True
    if path.name.startswith("legacy_"):
        return True
    return False


def extract_keep_paths(task: str) -> set[str]:
    """Paths the task says to leave alone (tests/, legacy files, etc.)."""
    found = {_normalize_name(m.group(0)) for m in _KEEP_PATH.finditer(task or "")}
    found.update(ALLOWLIST_RELATIVE_PATHS)
    return {p for p in found if p}


def extract_leftover_names(task: str) -> list[str]:
    """Rename/move/delete sources from the task. Excludes targets, keep-paths, allowlist."""
    text = task or ""
    keep = extract_keep_paths(text)
    sources: list[str] = []
    targets: set[str] = set()

    for pattern in (_RENAME, _MOVE):
        for match in pattern.finditer(text):
            source = _normalize_name(match.group("source"))
            target = _normalize_name(match.group("target"))
            if source:
                sources.append(source)
            if target:
                targets.add(target)
                targets.add(_basename(target))

    for match in _DELETE.finditer(text):
        source = _normalize_name(match.group("source"))
        if source:
            sources.append(source)

    leftovers: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if not source or source.lower() in _SKIP_NAMES:
            continue
        if source in keep or source in targets:
            continue
        if _basename(source).lower() in {t.lower() for t in targets}:
            continue
        if is_allowlisted_name(source):
            continue
        if source in seen:
            continue
        seen.add(source)
        leftovers.append(source)
    return leftovers


def _hit_relpath(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path)


def _candidate_locations(workspace: Path, name: str) -> list[Path]:
    locations = [
        workspace / name,
        workspace / "src" / name,
    ]
    base = _basename(name)
    if "." not in base:
        locations.extend(
            [
                workspace / f"{name}.py",
                workspace / "src" / f"{name}.py",
                workspace / f"{base}.py",
                workspace / "src" / f"{base}.py",
            ]
        )
    return locations


def _walk_exact_names(workspace: Path, names: set[str], limit: int = 400) -> list[Path]:
    """Cheap walk for exact basename matches; skips heavy/irrelevant trees."""
    hits: list[Path] = []
    scanned = 0
    root = workspace.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_WALK_DIRS]
        parent = Path(dirpath)
        for entry in dirnames + filenames:
            scanned += 1
            if scanned > limit:
                return hits
            if entry not in names:
                continue
            hits.append(parent / entry)
        if scanned > limit:
            break
    return hits


def find_leftovers(task: str, workspace: Path | None = None) -> list[str]:
    """Return leftover relative paths still present. Empty means stop is allowed."""
    names = extract_leftover_names(task)
    if not names:
        return []

    root = (workspace or Path.cwd()).resolve()
    if not root.is_dir():
        return []

    keep = extract_keep_paths(task)
    wanted_basenames = {_basename(n) for n in names}
    wanted_basenames -= {n for n in wanted_basenames if n.startswith("legacy_")}

    found: list[str] = []
    seen: set[str] = set()

    def record(path: Path) -> None:
        if not path.exists():
            return
        if is_allowlisted_path(root, path):
            return
        rel = _hit_relpath(root, path)
        if rel in keep or rel in ALLOWLIST_RELATIVE_PATHS:
            return
        if path.name.startswith("legacy_"):
            return
        if rel in seen:
            return
        seen.add(rel)
        found.append(rel)

    for name in names:
        for candidate in _candidate_locations(root, name):
            record(candidate)

    if wanted_basenames:
        for path in _walk_exact_names(root, wanted_basenames):
            record(path)

    return found


def leftover_user_note(leftovers: list[str]) -> str:
    listed = "\n".join(f"- {item}" for item in leftovers)
    return (
        f"{INTERNAL_NOTE_PREFIX}"
        "The following leftover path(s)/package(s) from the task still exist. "
        "Do not stop; rename, move, or delete them first:\n"
        f"{listed}"
    )
