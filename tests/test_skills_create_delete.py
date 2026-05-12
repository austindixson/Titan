from pathlib import Path

from titan.skills import create_local_skill, delete_local_skill, load_skill_text, use_skill, get_active_skills


def test_create_and_delete_local_skill(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    entry = create_local_skill("my-new-skill", "# My Skill\nDo X")
    assert entry.path.exists()
    assert load_skill_text("my-new-skill") is not None

    assert use_skill("my-new-skill") is True
    assert "my-new-skill" in get_active_skills()

    assert delete_local_skill("my-new-skill") is True
    assert load_skill_text("my-new-skill") is None
    assert "my-new-skill" not in get_active_skills()
