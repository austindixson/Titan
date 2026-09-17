"""Behavioral UI checks: real layout/scroll positions, worker lifecycle, and input."""

import asyncio
import threading

import pytest
from textual import events

from titan.config import HarnessConfig
from titan.mock_provider import MockProvider
from titan.titan_tui import ComposerTextArea, ConversationLog, LoopDoneMsg, LoopEventMsg, TitanTui
from titan.loop import AgentEvent
from titan.types import RunOutcome, RunStopContract, RunStopReason


@pytest.fixture(autouse=True)
def isolated_app(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setattr("titan.titan_tui.load_harness_config", lambda: HarnessConfig(provider="mock", model="mock"))
    monkeypatch.setattr("titan.titan_tui.build_provider_from_config", lambda cfg: MockProvider(script=[]))


def final(text="Finished.", reason=RunStopReason.AssistantFinal):
    return RunOutcome(text=text, stop=RunStopContract(reason=reason, iterations=1, tool_calls_total=0, elapsed_ms=10, notes=""))


def emit(app, event_type, **payload):
    app.on_loop_event_msg(LoopEventMsg(AgentEvent(event_type, payload)))


def assert_bottom(output):
    assert output.max_scroll_y > 0
    assert abs(output.scroll_y - output.max_scroll_y) <= 1, (output.scroll_y, output.max_scroll_y)


def test_bottom_follow_survives_streaming_markdown_reflow_and_final_replacement():
    async def run():
        app = TitanTui()
        async with app.run_test(size=(80, 24)) as pilot:
            app._write_chat_box("You", "build it", "")
            app.ui.pending = True
            for batch in range(5):
                for index in range(8):
                    emit(app, "provider_stream_delta", text=f"\n\n### Step {batch}-{index}\n\n" + "wrapped content " * 15, kind="text")
                app._tick()
                await pilot.pause()
                assert_bottom(app.query_one("#output", ConversationLog))
            app.on_loop_done_msg(LoopDoneMsg(final(app.stream_text + "\n\nFINAL SENTINEL")))
            await pilot.pause()
            assert_bottom(app.query_one("#output", ConversationLog))
    asyncio.run(run())


def test_manual_scroll_pauses_follow_and_bottom_resumes_it():
    async def run():
        app = TitanTui()
        async with app.run_test(size=(80, 24)) as pilot:
            for i in range(15):
                app._write_chat_box("Titan", f"Message {i}\n\n" + "content " * 60, "")
            await pilot.pause()
            output = app.query_one("#output", ConversationLog)
            output.scroll_home(animate=False, immediate=True)
            await pilot.pause()
            original = output.scroll_y
            for i in range(3):
                app._write_chat_box("Titan", f"new message {i}", "")
            await pilot.pause()
            assert output.scroll_y == original
            output.scroll_end(animate=False, immediate=True)
            await pilot.pause()
            app._write_chat_box("Titan", "## More\n\n" + "new content\n\n" * 40, "")
            await pilot.pause()
            assert_bottom(output)
    asyncio.run(run())


def test_follow_survives_resize_composer_growth_and_detail_views():
    async def run():
        app = TitanTui()
        async with app.run_test(size=(120, 36)) as pilot:
            app._write_chat_box("Titan", "\n\n".join(f"Paragraph {i} " + "long text " * 20 for i in range(25)), "")
            await pilot.pause()
            output = app.query_one("#output", ConversationLog)
            output.scroll_end(animate=False, immediate=True)
            await pilot.resize_terminal(60, 22)
            await pilot.pause()
            assert_bottom(output)
            app.query_one("#input", ComposerTextArea).load_text("draft\n" * 6)
            await pilot.pause()
            assert_bottom(output)
            await pilot.click("#tab-trace")
            app._write_chat_box("Titan", "\n\n".join(f"Tool update {i}" for i in range(30)), "")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert_bottom(output)
    asyncio.run(run())


def test_latest_shortcut_and_new_prompt_reanchor_after_reading_history(monkeypatch):
    async def run():
        app = TitanTui()
        async with app.run_test(size=(80, 24)) as pilot:
            app._write_chat_box("Titan", "\n\n".join(f"Paragraph {i}" for i in range(50)), "")
            await pilot.pause()
            output = app.query_one("#output", ConversationLog)
            output.scroll_home(animate=False, immediate=True)
            await pilot.press("ctrl+end")
            await pilot.pause()
            assert_bottom(output)
            app._write_chat_box("Titan", "\n\n".join(f"new paragraph {i}" for i in range(20)), "")
            await pilot.pause()
            assert_bottom(output)
            output.scroll_home(animate=False, immediate=True)
            composer = app.query_one("#input", ComposerTextArea)
            composer.focus()
            composer.load_text("/help")
            await pilot.press("enter")
            await pilot.pause()
            assert_bottom(output)
    asyncio.run(run())


def test_worker_exception_keeps_app_alive_and_accepts_next_task(monkeypatch):
    async def run():
        app = TitanTui()
        calls = []

        def fail_then_succeed(task, history, on_event):
            calls.append(task)
            if len(calls) == 1:
                on_event(AgentEvent("provider_stream_delta", {"text": "Partial work", "kind": "text"}))
                raise OSError("session disk unavailable")
            return final("Recovered.")

        monkeypatch.setattr(app.harness, "run_with_callback", fail_then_succeed)
        async with app.run_test() as pilot:
            composer = app.query_one("#input", ComposerTextArea)
            composer.load_text("first task")
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not app.ui.pending
            assert "session disk unavailable" in app.query_one("#output").text
            assert "Partial work" in app.query_one("#output").text
            assert not app.query_one("#btn-provider").disabled
            composer.load_text("second task")
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert calls == ["first task", "second task"]
            assert "Recovered." in app.query_one("#output").text
    asyncio.run(run())


def test_stop_waits_for_worker_and_does_not_discard_next_draft(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    async def run():
        app = TitanTui()
        calls = []

        def slow_run(task, history, on_event):
            calls.append(task)
            started.set()
            assert release.wait(5)
            return final("Stopped.", RunStopReason.Interrupted)

        monkeypatch.setattr(app.harness, "run_with_callback", slow_run)
        async with app.run_test() as pilot:
            try:
                composer = app.query_one("#input", ComposerTextArea)
                composer.load_text("slow task")
                await pilot.press("enter")
                assert await asyncio.to_thread(started.wait, 2)
                await pilot.press("escape")
                assert app.ui.pending
                composer.load_text("next draft")
                await pilot.press("enter")
                assert composer.text == "next draft"
                assert calls == ["slow task"]
            finally:
                release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not app.ui.pending
            assert composer.text == "next draft"
    asyncio.run(run())


def test_slow_git_diff_does_not_block_typing_or_escape(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    async def run():
        app = TitanTui()

        def slow_diff():
            started.set()
            assert release.wait(5)
            return "+finished"

        monkeypatch.setattr(app, "_collect_git_diff", slow_diff)
        async with app.run_test() as pilot:
            try:
                await pilot.click("#tab-diff")
                assert await asyncio.to_thread(started.wait, 2)
                await pilot.press("escape", "h", "i")
                assert not app.top_tab_expanded
                assert app.query_one("#input", ComposerTextArea).text == "hi"
                assert not release.is_set()
            finally:
                release.set()
            await app.workers.wait_for_complete()
            assert app.diff_lines == ["+finished"]
    asyncio.run(run())


def test_slow_compaction_keeps_ui_responsive_and_serializes_workspace_work(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def compact(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return None

    monkeypatch.setattr("titan.titan_tui.CompactionController.maybe_compact", compact)

    async def run():
        app = TitanTui()
        async with app.run_test() as pilot:
            composer = app.query_one("#input", ComposerTextArea)
            try:
                composer.load_text("/compact")
                await pilot.press("enter")
                assert await asyncio.to_thread(started.wait, 2)
                await pilot.press("h", "i", "enter")
                assert composer.text == "hi"
                assert app.ui.pending
                assert app.query_one("#btn-provider").disabled
            finally:
                release.set()
            await app.workers.wait_for_complete()
            assert not app.ui.pending
            assert composer.text == "hi"
    asyncio.run(run())


def test_mouse_wheel_releases_anchor_during_streaming():
    async def run():
        app = TitanTui()
        async with app.run_test(size=(80, 24)) as pilot:
            app._write_chat_box("Titan", "\n\n".join(f"Existing message {i}" for i in range(60)), "")
            await pilot.pause()
            output = app.query_one("#output", ConversationLog)
            output.post_message(events.MouseScrollUp(output, 3, 3, 0, -1, 0, False, False, False))
            await pilot.pause(0.3)
            position = output.scroll_y
            assert position < output.max_scroll_y
            emit(app, "provider_stream_delta", text="\n\n".join(f"New message {i}" for i in range(60)), kind="text")
            app._tick()
            await pilot.pause()
            assert output.scroll_y == position
            await pilot.press("ctrl+end")
            await pilot.pause()
            assert_bottom(output)
    asyncio.run(run())


def test_restored_session_starts_at_latest_message(tmp_path):
    from titan.session import SessionStore
    from titan.types import Message, Role

    store = SessionStore(str(tmp_path / ".titan" / "session.jsonl"))
    for i in range(30):
        store.append(Message(role=Role.USER if i % 2 == 0 else Role.ASSISTANT, content=f"Restored {i}\n\n" + "content " * 30))

    async def run():
        app = TitanTui()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            output = app.query_one("#output", ConversationLog)
            assert "Restored 29" in output.text
            assert_bottom(output)
    asyncio.run(run())


def test_paste_file_failure_keeps_exact_text_in_composer(monkeypatch):
    async def run():
        app = TitanTui()
        async with app.run_test() as pilot:
            composer = app.query_one("#input", ComposerTextArea)
            text = "paste content\n" * 100
            def fail(*args, **kwargs):
                raise OSError("disk full")
            monkeypatch.setattr("titan.titan_tui.tempfile.NamedTemporaryFile", fail)
            await composer._on_paste(events.Paste(text))
            assert composer.text == text
            await pilot.pause()
            assert composer.region.bottom <= app.size.height
    asyncio.run(run())


def test_slow_key_validation_is_responsive_and_does_not_replace_working_key(monkeypatch):
    from textual.widgets import Input

    started = threading.Event()
    release = threading.Event()

    async def run():
        app = TitanTui()
        def validate(*args):
            started.set()
            assert release.wait(5)
            return False, "http 401"
        monkeypatch.setattr(app, "_validate_provider_key", validate)
        async with app.run_test() as pilot:
            app.harness.config.api_keys["openai"] = "existing-working-key"
            app._prompt_for_provider_key("openai")
            key_input = app.query_one("#api_key_input", Input)
            try:
                key_input.value = "invalid-new-key"
                app.on_input_submitted(Input.Submitted(key_input, key_input.value))
                assert await asyncio.to_thread(started.wait, 2)
                await pilot.press("ctrl+f", "h", "i")
                assert app.query_one("#input", ComposerTextArea).text == "hi"
                assert app.harness.config.api_keys["openai"] == "existing-working-key"
            finally:
                release.set()
            await app.workers.wait_for_complete()
            assert not app.ui.pending
            assert not key_input.disabled
            assert app.harness.config.api_keys["openai"] == "existing-working-key"
            assert "invalid-new-key" not in app.query_one("#output").text
    asyncio.run(run())


def test_completion_time_includes_provider_wait(monkeypatch):
    async def run():
        app = TitanTui()
        async with app.run_test():
            app.ui.pending = True
            app.ui.started_at = 100.0
            monkeypatch.setattr("titan.titan_tui.time.time", lambda: 112.5)
            app.on_loop_done_msg(LoopDoneMsg(final()))
            assert "Completed · 12.5s" in str(app.query_one("#assistant_line").render())
    asyncio.run(run())
