from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _default_roots() -> list[Path]:
    return [
        Path.cwd() / ".titan" / "skills",
        Path.home() / ".hermes" / "skills",
        Path.home() / ".claude" / "skills",
    ]


def _iter_skill_files(roots: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            low = p.name.lower()
            if low == "skill.md" or low.endswith(".md"):
                out.append(p)
    return sorted(set(out))


def _slug_for(path: Path) -> str:
    stem = path.stem.lower().replace(" ", "-")
    if stem == "skill":
        stem = path.parent.name.lower().replace(" ", "-")
    return stem


@dataclass
class SkillEntry:
    slug: str
    path: Path


def discover_skills(roots: list[Path] | None = None) -> list[SkillEntry]:
    files = _iter_skill_files(roots or _default_roots())
    items: dict[str, SkillEntry] = {}
    for p in files:
        slug = _slug_for(p)
        if slug not in items:
            items[slug] = SkillEntry(slug=slug, path=p)
    return sorted(items.values(), key=lambda x: x.slug)


def active_skills_path() -> Path:
    return Path.cwd() / ".titan" / "active-skills.txt"


def get_active_skills() -> list[str]:
    p = active_skills_path()
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def set_active_skills(slugs: list[str]) -> None:
    p = active_skills_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(slugs) + ("\n" if slugs else ""))


def use_skill(slug: str) -> bool:
    known = {s.slug for s in discover_skills()}
    if slug not in known:
        return False
    active = get_active_skills()
    if slug not in active:
        active.append(slug)
        set_active_skills(active)
    return True


def unuse_skill(slug: str) -> bool:
    active = get_active_skills()
    if slug not in active:
        return False
    set_active_skills([s for s in active if s != slug])
    return True


def resolve_active_skill_entries() -> list[SkillEntry]:
    by_slug = {s.slug: s for s in discover_skills()}
    out: list[SkillEntry] = []
    for slug in get_active_skills():
        if slug in by_slug:
            out.append(by_slug[slug])
    return out


def create_local_skill(slug: str, content: str) -> SkillEntry:
    safe = slug.strip().lower().replace(" ", "-")
    if not safe:
        raise ValueError("slug is required")
    root = Path.cwd() / ".titan" / "skills" / safe
    root.mkdir(parents=True, exist_ok=True)
    p = root / "SKILL.md"
    p.write_text(content)
    return SkillEntry(slug=safe, path=p)


def delete_local_skill(slug: str) -> bool:
    safe = slug.strip().lower().replace(" ", "-")
    if not safe:
        return False
    root = Path.cwd() / ".titan" / "skills" / safe
    p = root / "SKILL.md"
    if not p.exists():
        return False
    p.unlink()
    try:
        root.rmdir()
    except Exception:
        pass

    # also unuse if active
    active = [s for s in get_active_skills() if s != safe]
    set_active_skills(active)
    return True


def load_skill_text(slug: str) -> str | None:
    by_slug = {s.slug: s for s in discover_skills()}
    entry = by_slug.get(slug)
    if not entry:
        return None
    try:
        return entry.path.read_text()
    except Exception:
        return None


def build_skill_system_context(limit_chars: int = 12000) -> str:
    chunks: list[str] = []
    total = 0
    for entry in resolve_active_skill_entries():
        text = load_skill_text(entry.slug)
        if not text:
            continue
        block = f"\n\n[SKILL:{entry.slug}]\n{text.strip()}\n"
        if total + len(block) > limit_chars:
            break
        chunks.append(block)
        total += len(block)
    return "".join(chunks).strip()
