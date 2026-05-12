from __future__ import annotations

import re
import shlex
from pathlib import Path
from urllib.parse import unquote, urlparse

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def candidate_image_paths_from_text(text: str) -> list[Path]:
    """Extract existing local image paths from natural-language text.

    Terminal drag/drop commonly produces absolute macOS screenshot paths with
    spaces, sometimes wrapped in backticks/quotes or encoded as file:// URLs.
    Token-only parsing misses unquoted paths with spaces, so this combines
    shlex tokens with a path-through-extension regex and verifies existence.
    """
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
        if not path.exists() or not path.is_file():
            return
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            paths.append(resolved)

    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    for token in tokens:
        add_candidate(token)

    extension_pattern = "|".join(re.escape(ext.lstrip(".")) for ext in sorted(IMAGE_EXTENSIONS, key=len, reverse=True))
    path_pattern = re.compile(
        rf"(?:file://)?(?:~|/)[^\n`\"'<>]*?\.(?:{extension_pattern})",
        flags=re.IGNORECASE,
    )
    for match in path_pattern.finditer(text):
        add_candidate(match.group(0))
    return paths
