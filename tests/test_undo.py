from __future__ import annotations

import asyncio
import io
import json
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

from titan.git_checkpoint import (
    git_checkpoint,
    list_checkpoints,
    restore_checkpoint,
    write_checkpoint_for_tests,
)
from titan.loop import AgentEvent
from titan.slash_commands import execute_slash_command
from titan.titan_cli import cmd_checkpoints, cmd_undo
from titan.titan_tui import TitanTui
import titan.titan_tui as titan_tui_module


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=True)


def _init_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "titan@example.com")
    _git(tmp_path, "config", "user.name", "Titan Test")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "init")
    return tracked


def _head(cwd: Path) -> str:
    return _git(cwd, "rev-parse", "HEAD").stdout.strip()


def test_writer_returns_none_under_pytest(tmp_path: Path):
    _init_repo(tmp_path)
    assert git_checkpoint(tmp_path) is None
    assert list_checkpoints(tmp_path) == []


def test_test_helper_bypasses_pytest_guard(tmp_path: Path):
    _init_repo(tmp_path)
    stamp = write_checkpoint_for_tests(tmp_path)
    assert stamp
    rows = list_checkpoints(tmp_path)
    assert len(rows) == 1
    assert rows[0]["id"] == stamp


def test_restore_latest_dirty_tree(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    tracked = _init_repo(tmp_path)
    tracked.write_text("checkpointed\n")
    (tmp_path / "notes.txt").write_text("keep me\n")
    stamp = write_checkpoint_for_tests(tmp_path)

    tracked.write_text("later dirty\n")
    (tmp_path / "extra.txt").write_text("leave this\n")
    head_before = _head(tmp_path)

    result = restore_checkpoint(tmp_path)
    assert result.ok
    assert result.checkpoint_id == stamp
    assert tracked.read_text() == "checkpointed\n"
    assert (tmp_path / "notes.txt").read_text() == "keep me\n"
    assert (tmp_path / "extra.txt").exists()
    assert _head(tmp_path) == head_before == result.head


def test_restore_named_id(tmp_path: Path):
    tracked = _init_repo(tmp_path)
    tracked.write_text("first\n")
    first = write_checkpoint_for_tests(tmp_path)
    tracked.write_text("second\n")
    second = write_checkpoint_for_tests(tmp_path)
    assert first != second

    tracked.write_text("third\n")
    result = restore_checkpoint(tmp_path, first)
    assert result.ok
    assert result.checkpoint_id == first
    assert tracked.read_text() == "first\n"


def test_head_mismatch_aborts_without_restore(tmp_path: Path):
    tracked = _init_repo(tmp_path)
    tracked.write_text("saved\n")
    stamp = write_checkpoint_for_tests(tmp_path)

    tracked.write_text("committed later\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "move HEAD")
    head_after_commit = _head(tmp_path)
    tracked.write_text("do not lose me\n")

    result = restore_checkpoint(tmp_path, stamp)
    assert result.ok is False
    assert "diverged" in result.message
    assert tracked.read_text() == "do not lose me\n"
    assert _head(tmp_path) == head_after_commit


def test_restore_does_not_use_reset_hard(tmp_path: Path):
    tracked = _init_repo(tmp_path)
    tracked.write_text("snap\n")
    write_checkpoint_for_tests(tmp_path)
    tracked.write_text("changed\n")

    result = restore_checkpoint(tmp_path)
    assert result.ok
    flat = [" ".join(cmd) for cmd in result.commands]
    assert not any("reset --hard" in item for item in flat)
    assert not any(cmd and cmd[0] == "reset" for cmd in result.commands)
    assert not any(cmd and cmd[0] == "clean" for cmd in result.commands)
    assert not any("push" in cmd for cmd in result.commands)
    assert any(cmd[:1] == ["restore"] for cmd in result.commands)


def test_head_unchanged_after_restore(tmp_path: Path):
    tracked = _init_repo(tmp_path)
    tracked.write_text("one\n")
    write_checkpoint_for_tests(tmp_path)
    tracked.write_text("two\n")
    before = _head(tmp_path)
    result = restore_checkpoint(tmp_path)
    assert result.ok
    assert _head(tmp_path) == before


def test_missing_untracked_is_reported_not_deleted(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "present.txt").write_text("here\n")
    stamp = write_checkpoint_for_tests(tmp_path)

    payload_path = tmp_path / ".titan" / "checkpoints" / f"{stamp}.json"
    data = json.loads(payload_path.read_text())
    data["untracked"].append({"path": "vanished.txt", "content": None})
    payload_path.write_text(json.dumps(data) + "\n")

    extra = tmp_path / "extra-after.txt"
    extra.write_text("do not delete\n")

    result = restore_checkpoint(tmp_path, stamp)
    assert result.ok
    assert "vanished.txt" in result.missing_untracked
    assert extra.exists()
    assert extra.read_text() == "do not delete\n"
    assert (tmp_path / "present.txt").exists()


def test_cli_undo_and_checkpoints(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    tracked = _init_repo(tmp_path)
    tracked.write_text("cli-saved\n")
    stamp = write_checkpoint_for_tests(tmp_path)
    tracked.write_text("cli-dirty\n")

    listed = io.StringIO()
    with redirect_stdout(listed):
        assert cmd_checkpoints() == 0
    assert stamp in listed.getvalue()

    out = io.StringIO()
    with redirect_stdout(out):
        assert cmd_undo() == 0
    assert stamp in out.getvalue()
    assert tracked.read_text() == "cli-saved\n"

    tracked.write_text("named-dirty\n")
    named = io.StringIO()
    with redirect_stdout(named):
        assert cmd_undo(stamp) == 0
    assert tracked.read_text() == "cli-saved\n"


def test_slash_undo_latest_and_named(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    tracked = _init_repo(tmp_path)
    tracked.write_text("slash-a\n")
    first = write_checkpoint_for_tests(tmp_path)
    tracked.write_text("slash-b\n")
    write_checkpoint_for_tests(tmp_path)
    tracked.write_text("slash-now\n")

    latest = execute_slash_command("/undo")
    assert latest.handled and not latest.is_error
    assert tracked.read_text() == "slash-b\n"

    named = execute_slash_command(f"/undo {first}")
    assert named.handled and not named.is_error
    assert tracked.read_text() == "slash-a\n"


def test_slash_undo_refuses_while_run_pending(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    tracked = _init_repo(tmp_path)
    tracked.write_text("pending-saved\n")
    write_checkpoint_for_tests(tmp_path)
    tracked.write_text("pending-dirty\n")

    refused = execute_slash_command("/undo", run_pending=True)
    assert refused.handled
    assert refused.is_error
    assert "pending" in refused.message
    assert tracked.read_text() == "pending-dirty\n"


def test_slash_help_lists_undo():
    r = execute_slash_command("/help")
    assert "/undo" in r.message


def _patch_tui(monkeypatch):
    from titan.config import HarnessConfig
    from titan.mock_provider import MockProvider

    monkeypatch.setattr("titan.titan_tui.load_harness_config", lambda: HarnessConfig(provider="openai", model="mock"))
    monkeypatch.setattr("titan.titan_tui.build_provider_from_config", lambda cfg: MockProvider(script=[]))
    monkeypatch.setattr("titan.titan_tui.supported_openai_compat_providers", lambda: ["openai"])


def test_tui_trace_shows_git_checkpoint_and_undo_events(monkeypatch):
    _patch_tui(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app._trace_git_event(AgentEvent("git_checkpoint", {"id": "20260101T000000000000Z"}))
            app._trace_git_event(AgentEvent("undo", {"id": "20260101T000000000000Z", "ok": True}))
            assert any("git-checkpoint 20260101T000000000000Z" in line for line in app.trace_lines)
            assert any(line.startswith("undo 20260101T000000000000Z") for line in app.trace_lines)

    asyncio.run(_run())


def test_tui_slash_undo_refuses_while_pending(monkeypatch, tmp_path: Path):
    _patch_tui(monkeypatch)
    monkeypatch.chdir(tmp_path)
    tracked = _init_repo(tmp_path)
    tracked.write_text("tui-saved\n")
    write_checkpoint_for_tests(tmp_path)
    tracked.write_text("tui-dirty\n")

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.ui.pending = True
            handled = app._handle_slash_task("/undo")
            assert handled is True
            assert any("pending" in line for line in app.chat_lines)
            assert tracked.read_text() == "tui-dirty\n"

    asyncio.run(_run())
