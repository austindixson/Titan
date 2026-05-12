from pathlib import Path

from titan.slash_commands import execute_slash_command
from titan.tools import default_registry


def test_slash_help(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    r = execute_slash_command("/help")
    assert r.handled is True
    assert "commands:" in r.message


def test_slash_use_unuse(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    skill_dir = tmp_path / ".titan" / "skills" / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text("# demo")

    r1 = execute_slash_command("/use demo")
    assert r1.handled and not r1.is_error

    r2 = execute_slash_command("/active")
    assert "demo" in r2.message

    r3 = execute_slash_command("/unuse demo")
    assert r3.handled and not r3.is_error


def test_slash_todo_and_memory(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    reg = default_registry()
    reg.execute("t1", "todo_set", {"todos": [{"id": "x", "content": "ship", "status": "pending"}]})
    reg.execute("m1", "memory_add", {"content": "User likes concise output"})

    t = execute_slash_command("/todo")
    assert t.handled and "ship" in t.message

    m = execute_slash_command("/memory concise")
    assert m.handled and "concise" in m.message.lower()


def test_slash_trace_token(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    r = execute_slash_command("/trace")
    assert r.handled
    assert r.message == "trace-toggle"


def test_slash_config_recaps_default_off_and_settable(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(cfg_path))
    monkeypatch.chdir(tmp_path)

    shown = execute_slash_command("/config")
    assert shown.handled and "chat_recaps_enabled=false" in shown.message
    assert "learning_enabled=false" in shown.message

    set_result = execute_slash_command("/config set chat_recaps_enabled true")
    assert set_result.handled and not set_result.is_error

    got = execute_slash_command("/config get chat_recaps_enabled")
    assert got.handled and got.message == "True"
