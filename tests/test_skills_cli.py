from pathlib import Path

from titan.skills import (
    build_skill_system_context,
    discover_skills,
    get_active_skills,
    set_active_skills,
    unuse_skill,
    use_skill,
)


def test_discover_and_use_skill(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    skills_root = tmp_path / ".titan" / "skills" / "demo"
    skills_root.mkdir(parents=True, exist_ok=True)
    (skills_root / "SKILL.md").write_text("# Demo skill")

    discovered = discover_skills([tmp_path / ".titan" / "skills"])
    assert any(s.slug == "demo" for s in discovered)

    assert use_skill("demo") is True
    assert "demo" in get_active_skills()


def test_unuse_skill(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    set_active_skills(["a", "b"])
    assert unuse_skill("a") is True
    assert get_active_skills() == ["b"]


def test_skill_context_includes_active(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    skills_root = tmp_path / ".titan" / "skills" / "ops"
    skills_root.mkdir(parents=True, exist_ok=True)
    (skills_root / "SKILL.md").write_text("# Ops\nUse checklists")
    assert use_skill("ops") is True

    ctx = build_skill_system_context(limit_chars=10000)
    assert "[SKILL:ops]" in ctx
    assert "Use checklists" in ctx
