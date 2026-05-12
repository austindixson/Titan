from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse

UNICODE_SPACES_RE = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")


def _path_variants(raw_path: str) -> list[str]:
    variants: list[str] = [raw_path]

    normalized_spaces = UNICODE_SPACES_RE.sub(" ", raw_path)
    variants.append(normalized_spaces)

    ampm_spaced = re.sub(r"(\d)(AM|PM)(\.)", r"\1 \2\3", normalized_spaces, flags=re.IGNORECASE)
    variants.append(ampm_spaced)

    variants.append(re.sub(r" (AM|PM)(\.)", "\u202f\\1\\2", normalized_spaces, flags=re.IGNORECASE))
    variants.append(re.sub(r" (AM|PM)(\.)", "\u202f\\1\\2", ampm_spaced, flags=re.IGNORECASE))

    variants.append(unicodedata.normalize("NFD", raw_path))
    variants.append(unicodedata.normalize("NFC", raw_path))

    out: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        out.append(variant)
    return out


def _normalize_input_path(raw_path: str) -> str:
    text = raw_path.strip().strip("`\"'“”‘’()[]{}<>,")
    if text.startswith("file://"):
        parsed = urlparse(text)
        text = unquote(parsed.path)
    else:
        text = unquote(text)
    if text.startswith("@"):
        text = text[1:]
    return text


def resolve_existing_read_path(raw_path: str) -> Path | None:
    normalized = _normalize_input_path(raw_path)
    for variant in _path_variants(normalized):
        candidate = Path(variant).expanduser()
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None
