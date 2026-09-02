from __future__ import annotations

import shutil
from pathlib import Path

from titan.config import HarnessConfig
from titan.leftover import (
    ALLOWLIST_RELATIVE_PATHS,
    INTERNAL_NOTE_PREFIX,
    extract_leftover_names,
    find_leftovers,
)
from titan.loop import AgentLoop
from titan.provider import Provider
from titan.session import SessionStore
from titan.titan import TitanHarness
from titan.tools import default_registry
from titan.types import AssistantResponse, Message, Role, RunStopReason

PACKAGE_RENAME_TASK = (
    "Rename the Python package `blueledger` to `aurorabooks`. "
    "Update imports and references so the old package directory no longer remains. "
    "Keep tests/data/legacy_blueledger.json and migrations/legacy_blueledger.sql "
    "with the original names; do not rename those historical artifacts."
)


def _seed_legacy_keep_files(root: Path) -> None:
    (root / "tests" / "data").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "data" / "legacy_blueledger.json").write_text("{\"legacy\": true}\n")
    (root / "migrations").mkdir(parents=True, exist_ok=True)
    (root / "migrations" / "legacy_blueledger.sql").write_text("-- legacy blueledger schema\n")


def _seed_old_package(root: Path) -> Path:
    pkg = root / "blueledger"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("# old package\n")
    return pkg


class HistoryProvider(Provider):
    def __init__(self, script: list[AssistantResponse]):
        self.script = script
        self.idx = 0
        self.calls: list[list[Message]] = []

    def generate(self, model: str, messages: list[Message], tools: list[dict]) -> AssistantResponse:
        self.calls.append(list(messages))
        if self.idx >= len(self.script):
            return AssistantResponse(text="")
        resp = self.script[self.idx]
        self.idx += 1
        return resp


class CleanupOnSecondCallProvider(Provider):
    """First call claims done while leftover exists; second call after note removes it."""

    def __init__(self, leftover_dir: Path):
        self.leftover_dir = leftover_dir
        self.idx = 0
        self.calls: list[list[Message]] = []

    def generate(self, model: str, messages: list[Message], tools: list[dict]) -> AssistantResponse:
        self.calls.append(list(messages))
        self.idx += 1
        if self.idx == 1:
            return AssistantResponse(text="Done. Renamed blueledger to aurorabooks.", tool_calls=[])
        if self.leftover_dir.exists():
            shutil.rmtree(self.leftover_dir)
        return AssistantResponse(text="Removed leftover blueledger. Done.", tool_calls=[])


def test_extractor_uses_rename_source_not_target_or_keep_paths():
    names = extract_leftover_names(PACKAGE_RENAME_TASK)
    assert names == ["blueledger"]
    assert "aurorabooks" not in names
    assert "tests" not in names
    assert all("legacy_blueledger" not in n for n in names)


def test_extractor_ignores_generic_tasks_and_keep_paths():
    assert extract_leftover_names("say hi") == []
    assert extract_leftover_names("build a website") == []
    assert extract_leftover_names("fix and test the harness loop behavior") == []
    assert extract_leftover_names("Keep tests/data/legacy_blueledger.json in place.") == []
    assert extract_leftover_names("review and improve planning/decompose behavior") == []


def test_probe_finds_old_package_but_not_allowlisted_legacy_files(tmp_path: Path):
    _seed_old_package(tmp_path)
    _seed_legacy_keep_files(tmp_path)
    leftovers = find_leftovers(PACKAGE_RENAME_TASK, tmp_path)
    assert leftovers
    assert any(item == "blueledger" or item.startswith("blueledger/") for item in leftovers)
    assert "tests/data/legacy_blueledger.json" not in leftovers
    assert "migrations/legacy_blueledger.sql" not in leftovers
    for allowed in ALLOWLIST_RELATIVE_PATHS:
        assert allowed not in leftovers


def test_probe_does_not_treat_tests_dir_or_random_files_as_leftovers(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "README.md").write_text("hello")
    (tmp_path / "aurorabooks").mkdir()
    (tmp_path / "aurorabooks" / "__init__.py").write_text("")
    _seed_legacy_keep_files(tmp_path)
    assert find_leftovers(PACKAGE_RENAME_TASK, tmp_path) == []
    assert find_leftovers("say hi and finish", tmp_path) == []


def test_blueledger_leftover_blocks_assistant_final_and_injects_internal_note(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    leftover_dir = _seed_old_package(tmp_path)
    _seed_legacy_keep_files(tmp_path)
    (tmp_path / "aurorabooks").mkdir()
    (tmp_path / "aurorabooks" / "__init__.py").write_text("# new name\n")

    provider = CleanupOnSecondCallProvider(leftover_dir)
    events: list[tuple[str, dict]] = []
    loop = AgentLoop(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow", max_iterations=6),
        session=SessionStore(str(tmp_path / "session.jsonl")),
    )
    history = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run_with_callback(
        PACKAGE_RENAME_TASK,
        history,
        on_event=lambda e: events.append((e.type, e.payload)),
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    assert out.stop.iterations >= 2
    assert provider.idx >= 2
    note_messages = [m for m in history if m.role == Role.USER and m.content.startswith(INTERNAL_NOTE_PREFIX)]
    assert note_messages
    assert "blueledger" in note_messages[0].content
    blocked = [payload for event_type, payload in events if event_type == "leftover_stop_blocked"]
    assert blocked
    assert any("blueledger" in item for item in blocked[0]["leftovers"])
    assert not leftover_dir.exists()


def test_allowlisted_legacy_files_do_not_block_stop(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _seed_legacy_keep_files(tmp_path)
    (tmp_path / "aurorabooks").mkdir()
    (tmp_path / "aurorabooks" / "__init__.py").write_text("# new name\n")

    provider = HistoryProvider([AssistantResponse(text="Done. Package renamed.", tool_calls=[])])
    loop = AgentLoop(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow", max_iterations=4),
        session=SessionStore(str(tmp_path / "session.jsonl")),
    )
    history = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run(PACKAGE_RENAME_TASK, history)

    assert out.stop.reason == RunStopReason.AssistantFinal
    assert out.stop.iterations == 1
    assert provider.idx == 1
    assert not any(m.role == Role.USER and m.content.startswith(INTERNAL_NOTE_PREFIX) for m in history)
    assert (tmp_path / "tests" / "data" / "legacy_blueledger.json").exists()
    assert (tmp_path / "migrations" / "legacy_blueledger.sql").exists()


def test_text_only_done_stops_immediately_when_leftovers_are_gone(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _seed_legacy_keep_files(tmp_path)
    (tmp_path / "aurorabooks").mkdir()
    (tmp_path / "aurorabooks" / "__init__.py").write_text("# new name\n")
    assert not (tmp_path / "blueledger").exists()

    provider = HistoryProvider([AssistantResponse(text="All done.", tool_calls=[])])
    loop = AgentLoop(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow"),
        session=SessionStore(str(tmp_path / "session.jsonl")),
    )
    history = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run(PACKAGE_RENAME_TASK, history)

    assert out.stop.reason == RunStopReason.AssistantFinal
    assert out.text == "All done."
    assert out.stop.iterations == 1
    assert provider.idx == 1


def test_titan_harness_leftover_blocks_finalize_and_continues(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    leftover_dir = _seed_old_package(tmp_path)
    _seed_legacy_keep_files(tmp_path)

    provider = CleanupOnSecondCallProvider(leftover_dir)
    events: list[tuple[str, dict]] = []
    harness = TitanHarness(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow", max_iterations=6),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    history = [Message(role=Role.SYSTEM, content="sys")]
    out = harness.run_with_callback(
        PACKAGE_RENAME_TASK,
        history,
        on_event=lambda e: events.append((e.type, e.payload)),
    )

    assert out.stop.reason == RunStopReason.AssistantFinal
    assert out.stop.iterations >= 2
    assert provider.idx >= 2
    note_messages = [m for m in history if m.role == Role.USER and m.content.startswith(INTERNAL_NOTE_PREFIX)]
    assert note_messages
    assert "blueledger" in note_messages[0].content
    assert any(event_type == "leftover_stop_blocked" for event_type, _payload in events)
    assert not leftover_dir.exists()


def test_titan_harness_allowlisted_legacy_files_do_not_block_stop(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _seed_legacy_keep_files(tmp_path)

    provider = HistoryProvider([AssistantResponse(text="Done.", tool_calls=[])])
    harness = TitanHarness(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow", max_iterations=4),
        session_store=SessionStore(str(tmp_path / "session.jsonl")),
    )
    history = [Message(role=Role.SYSTEM, content="sys")]
    out = harness.run_with_callback(PACKAGE_RENAME_TASK, history)

    assert out.stop.reason == RunStopReason.AssistantFinal
    assert out.stop.iterations == 1
    assert provider.idx == 1
    assert not any(m.role == Role.USER and m.content.startswith(INTERNAL_NOTE_PREFIX) for m in history)


def test_leftover_block_continues_loop_without_assistant_final_while_present(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _seed_old_package(tmp_path)
    _seed_legacy_keep_files(tmp_path)

    provider = HistoryProvider(
        [
            AssistantResponse(text="Done.", tool_calls=[]),
            AssistantResponse(text="Still done.", tool_calls=[]),
            AssistantResponse(text="Really done.", tool_calls=[]),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        tools=default_registry(),
        config=HarnessConfig(permission_mode="allow", max_iterations=3),
        session=SessionStore(str(tmp_path / "session.jsonl")),
    )
    history = [Message(role=Role.SYSTEM, content="sys")]
    out = loop.run(PACKAGE_RENAME_TASK, history)

    assert out.stop.reason != RunStopReason.AssistantFinal
    assert any(m.role == Role.USER and m.content.startswith(INTERNAL_NOTE_PREFIX) for m in history)
    assert provider.idx >= 3
    assert (tmp_path / "blueledger").exists()
