from __future__ import annotations

import re
import shlex
from pathlib import Path
from urllib.parse import unquote, urlparse
from .path_resolver import resolve_existing_read_path


def _normalize_wrapped_path_text(text: str) -> str:
    """Heal accidental hard-wraps inside absolute/file:// paths.

    Terminal transcripts can insert newlines mid-path (e.g. after `_` in a long
    temp directory). Join newlines that sit between path-like characters so
    image extraction still works, while leaving normal paragraph newlines alone.
    """
    return re.sub(r"(?<=[\w./~%:+\-])\n(?=[\w./~%:+\-])", "", text)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def local_image_references_from_text(text: str) -> list[str]:
    """Extract local image-looking path references, regardless of file existence."""
    normalized_text = _normalize_wrapped_path_text(text)
    refs: list[str] = []
    seen: set[str] = set()

    def add_ref(raw: str) -> None:
        cleaned = raw.strip().strip("`\"'“”‘’()[]{}<>,")
        if cleaned.startswith("file://"):
            parsed = urlparse(cleaned)
            cleaned = unquote(parsed.path)
        else:
            cleaned = unquote(cleaned)
        if not cleaned:
            return
        path = Path(cleaned).expanduser()
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return
        key = str(path)
        if key not in seen:
            seen.add(key)
            refs.append(key)

    try:
        tokens = shlex.split(normalized_text)
    except ValueError:
        tokens = normalized_text.split()
    for token in tokens:
        add_ref(token)

    extension_pattern = "|".join(re.escape(ext.lstrip(".")) for ext in sorted(IMAGE_EXTENSIONS, key=len, reverse=True))
    path_pattern = re.compile(
        rf"(?:file://)?(?:~|/)[^\n`\"'<>]*?\.(?:{extension_pattern})",
        flags=re.IGNORECASE,
    )
    for match in path_pattern.finditer(normalized_text):
        add_ref(match.group(0))
    return refs


def candidate_image_paths_from_text(text: str) -> list[Path]:
    """Extract existing local image paths from natural-language text.

    Terminal drag/drop commonly produces absolute macOS screenshot paths with
    spaces, sometimes wrapped in backticks/quotes or encoded as file:// URLs.
    Token-only parsing misses unquoted paths with spaces, so this combines
    shlex tokens with a path-through-extension regex and verifies existence.
    """
    normalized_text = _normalize_wrapped_path_text(text)
    paths: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(raw: str) -> None:
        cleaned = raw.strip().strip("`\"'“”‘’()[]{}<>,")
        if cleaned.startswith("file://"):
            parsed = urlparse(cleaned)
            cleaned = unquote(parsed.path)
        else:
            cleaned = unquote(cleaned)
        if not cleaned:
            return
        path = Path(cleaned).expanduser()
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return
        resolved = resolve_existing_read_path(str(path))
        if resolved is None:
            return
        if resolved not in seen:
            seen.add(resolved)
            paths.append(resolved)

    try:
        tokens = shlex.split(normalized_text)
    except ValueError:
        tokens = normalized_text.split()
    for token in tokens:
        add_candidate(token)

    extension_pattern = "|".join(re.escape(ext.lstrip(".")) for ext in sorted(IMAGE_EXTENSIONS, key=len, reverse=True))
    path_pattern = re.compile(
        rf"(?:file://)?(?:~|/)[^\n`\"'<>]*?\.(?:{extension_pattern})",
        flags=re.IGNORECASE,
    )
    for match in path_pattern.finditer(normalized_text):
        add_candidate(match.group(0))
    return paths
