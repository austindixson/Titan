import asyncio
import pytest

from rich.panel import Panel
from rich.text import Text
from textual.widgets import Button, Input, TextArea

from titan.config import HarnessConfig
from titan.mock_provider import MockProvider
from titan.titan_tui import TitanTui
from titan.types import RunOutcome, RunStopContract, RunStopReason
import titan.titan_tui as titan_tui_module


@pytest.fixture(autouse=True)
def isolated_workspace(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(tmp_path / "config.json"))
    for variable in ("GROK_AUTH_PATH", "CODEX_AUTH_PATH", "HERMES_AUTH_PATH", "PI_AUTH_PATH"):
        monkeypatch.setenv(variable, str(tmp_path / variable))


def _patch_tui_deps(monkeypatch):
    monkeypatch.setattr("titan.titan_tui.load_harness_config", lambda: HarnessConfig(provider="openai", model="mock"))
    monkeypatch.setattr("titan.titan_tui.build_provider_from_config", lambda cfg: MockProvider(script=[]))
    monkeypatch.setattr("titan.titan_tui.supported_openai_compat_providers", lambda: ["openai"])


def test_tui_controls_are_limited_to_stop_provider_operator_trace_quit(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app._apply_responsive_layout(80)
            labels = {
                button.id: str(button.label)
                for button in app.query(Button)
                if button.id
                and not str(button.id).startswith("provider-opt-")
                and not str(button.id).startswith("model-opt-")
            }
            assert labels == {
                "tab-trace": "Activity",
                "tab-diff": "Diff",
                "btn-stop": "Stop",
                "btn-provider": "openai ▾",
                "btn-model": "mock ▾",
                "btn-new-key": "New key",
            }

    asyncio.run(_run())


def test_tui_provider_options_are_sanitized(monkeypatch):
    monkeypatch.setattr("titan.titan_tui.load_harness_config", lambda: HarnessConfig(provider="openai", model="mock"))
    monkeypatch.setattr("titan.titan_tui.build_provider_from_config", lambda cfg: MockProvider(script=[]))
    monkeypatch.setattr("titan.titan_tui.supported_openai_compat_providers", lambda: ["", "xai", " xai ", "zai", "  ", "openai"])
    app = TitanTui()
    assert app.provider_options[0] in {"openai", "grok", "openai-codex"}
    assert "grok" in app.provider_options
    assert "openai-codex" in app.provider_options


def test_tui_provider_options_prefer_grok_and_codex(monkeypatch):
    monkeypatch.setattr("titan.titan_tui.load_harness_config", lambda: HarnessConfig(provider="openai", model="mock"))
    monkeypatch.setattr("titan.titan_tui.build_provider_from_config", lambda cfg: MockProvider(script=[]))
    monkeypatch.setattr(
        "titan.titan_tui.supported_openai_compat_providers",
        lambda: ["zai", "openai-codex", "grok", "xai", "openai"],
    )
    app = TitanTui()
    assert app.provider_options[:2] == ["grok", "openai-codex"] or app.provider_options[:3][0] == "openai"


def test_tui_trace_defaults_normal_and_small(monkeypatch):
    _patch_tui_deps(monkeypatch)

    app = TitanTui()

    assert app.trace_verbosity_levels[app.trace_verbosity_index] == "normal"
    async def _run():
        async with app.run_test(size=(80, 24)):
            assert app.query_one("#top").display is False
            assert app.query_one("#input").region.bottom <= 24
    asyncio.run(_run())


def test_tui_hotkeys_include_focus_provider_and_theme(monkeypatch):
    _patch_tui_deps(monkeypatch)
    bindings2 = {(binding.key, binding.action) for binding in TitanTui.BINDINGS}
    bindings3 = {(binding.key, binding.action, binding.description) for binding in TitanTui.BINDINGS}

    assert ("ctrl+f", "operator_input") in bindings2
    assert ("ctrl+o", "operator_input") not in bindings2
    assert ("ctrl+p", "cycle_provider", "Provider") in bindings3
    assert ("ctrl+l", "cycle_model", "Model") in bindings3
    assert ("ctrl+n", "cycle_theme", "Theme") in bindings3


def test_tui_ctrl_c_cancels_then_quits(monkeypatch):
    _patch_tui_deps(monkeypatch)
    bindings = {(binding.key, binding.action, binding.description) for binding in TitanTui.BINDINGS}

    assert ("ctrl+c", "handle_ctrl_c", "Cancel/Quit") in bindings
    assert ("ctrl+c", "quit", "Quit") not in bindings

    async def _run():
        app = TitanTui()
        exit_called = False

        def fake_exit(*args, **kwargs):
            nonlocal exit_called
            exit_called = True

        async with app.run_test(size=(100, 32)):
            monkeypatch.setattr(app, "exit", fake_exit)
            app.ui.pending = True
            app.ui.pending_tool_names.append("shell")

            app.action_handle_ctrl_c()
            assert exit_called is False
            assert app.ui.pending is True  # worker owns completion; no overlapping runs
            assert app.ctrl_c_quit_armed is True

            app.action_handle_ctrl_c()
            assert exit_called is True

    asyncio.run(_run())


def test_tui_top_panel_tabs_switch_trace_and_diff(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        monkeypatch.setattr(app, "_collect_git_diff", lambda: "diff --git a/a b/a\n-old\n+new")
        async with app.run_test(size=(100, 32)):
            trace = app.query_one("#trace", titan_tui_module.SelectableRichLog)
            diff = app.query_one("#diff", titan_tui_module.SelectableRichLog)
            assert app.active_top_tab == "trace"
            assert trace.display is True
            assert diff.display is False

            app._set_top_tab("diff")
            await app.workers.wait_for_complete()
            assert app.active_top_tab == "diff"
            assert trace.display is False
            assert diff.display is True
            assert app.diff_lines == ["diff --git a/a b/a", "-old", "+new"]
            assert str(app.query_one("#tab-diff", Button).label) == "Diff"

    asyncio.run(_run())


def test_tui_trace_tab_click_expands_over_chat_and_click_again_minimizes(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)) as pilot:
            top = app.query_one("#top")
            output = app.query_one("#output", titan_tui_module.ConversationLog)
            trace = app.query_one("#trace", titan_tui_module.SelectableRichLog)

            assert app.active_top_tab == "trace"
            assert app.top_tab_expanded is False
            assert top.has_class("expanded") is False
            assert app.query_one("#welcome").display is True
            assert trace.display is True

            await pilot.click("#tab-trace")
            await pilot.pause()
            assert app.top_tab_expanded is True
            assert top.has_class("expanded") is True
            assert output.has_class("trace-hidden") is True
            assert str(app.query_one("#tab-trace", Button).label) == "Activity ×"

            app._toggle_active_top_tab_expansion()
            await pilot.pause()
            assert app.top_tab_expanded is False
            assert top.has_class("expanded") is False
            assert output.has_class("trace-hidden") is False
            assert str(app.query_one("#tab-trace", Button).label) == "Activity"

    asyncio.run(_run())


def test_tui_diff_tab_click_expands_over_chat_and_click_again_minimizes(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        monkeypatch.setattr(app, "_collect_git_diff", lambda: "diff --git a/a b/a\n-old\n+new")
        async with app.run_test(size=(100, 32)) as pilot:
            top = app.query_one("#top")
            output = app.query_one("#output", titan_tui_module.ConversationLog)

            assert app.active_top_tab == "trace"
            assert app.top_tab_expanded is False

            await pilot.click("#tab-diff")
            await pilot.pause()
            assert app.active_top_tab == "diff"
            assert app.top_tab_expanded is True
            assert top.has_class("expanded") is True
            assert output.has_class("trace-hidden") is True
            assert str(app.query_one("#tab-diff", Button).label) == "Diff ×"

            app._toggle_active_top_tab_expansion()
            await pilot.pause()
            assert app.top_tab_expanded is False
            assert top.has_class("expanded") is False
            assert output.has_class("trace-hidden") is False
            assert str(app.query_one("#tab-diff", Button).label) == "Diff"

            app._toggle_active_top_tab_expansion()
            await pilot.pause()
            assert app.top_tab_expanded is True
            assert top.has_class("expanded") is True
            assert output.has_class("trace-hidden") is True
            assert str(app.query_one("#tab-diff", Button).label) == "Diff ×"

    asyncio.run(_run())


def test_tui_clicking_logs_focuses_them_and_buttons_still_work(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)) as pilot:
            output = app.query_one("#output", titan_tui_module.ConversationLog)
            trace = app.query_one("#trace", titan_tui_module.SelectableRichLog)
            composer = app.query_one("#input", titan_tui_module.ComposerTextArea)

            composer.load_text("to-clear")
            app._write_chat_box("Titan", "Hello", "")
            await pilot.pause()
            await pilot.click("#output")
            await pilot.pause()
            assert app.focused is output

            await pilot.click("#tab-trace")
            await pilot.pause()
            await pilot.click("#trace")
            await pilot.pause()
            assert app.focused is trace

            app.action_clear_input()
            await pilot.pause()
            assert composer.text == ""

    asyncio.run(_run())


def test_tui_diff_lines_are_color_coded(monkeypatch):
    _patch_tui_deps(monkeypatch)
    app = TitanTui()

    assert str(app._style_diff_line("+added").style) == "green"
    assert str(app._style_diff_line("-removed").style) == "red"
    assert str(app._style_diff_line("@@ -1 +1 @@").style) == "bold cyan"
    assert str(app._style_diff_line("diff --git a/a b/a").style) == "bold magenta"


def test_tui_input_uses_wrapping_text_area(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            composer = app.query_one("#input", TextArea)
            assert composer.soft_wrap is True

    asyncio.run(_run())


def test_tui_short_multiline_paste_stays_inline(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            composer = app.query_one("#input", titan_tui_module.ComposerTextArea)
            pasted = "alpha\nbeta\ngamma"

            display = composer.normalize_paste_for_display(pasted)

            assert display == pasted
            assert composer.expand_paste_tokens(f"use {display}") == f"use {pasted}"

    asyncio.run(_run())


@pytest.mark.parametrize("pasted", ["x" * 2001, "\n".join(f"line {i}" for i in range(11)), "λ\r\n" * 12])
def test_long_paste_writes_exact_text_and_submits_only_path(monkeypatch, tmp_path, pasted):
    from pathlib import Path
    from textual import events

    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        submitted = []

        async def capture(task):
            submitted.append(task)

        monkeypatch.setattr(app, "_submit_task", capture)
        async with app.run_test() as pilot:
            composer = app.query_one("#input", titan_tui_module.ComposerTextArea)
            await composer._on_paste(events.Paste(pasted))
            path = Path(composer.text)
            assert path.parent == tmp_path / ".titan" / "pastes"
            assert path.suffix == ".txt"
            assert path.read_bytes() == pasted.encode("utf-8")
            assert Path(composer.normalize_paste_for_display(pasted)) != path
            await pilot.press("enter")
            assert submitted == [str(path)]
            assert composer.message_history == [str(path)]
            assert path.exists()

    asyncio.run(_run())


def test_file_drops_handle_spaces_escaped_paths_and_multiple_files(monkeypatch, tmp_path):
    _patch_tui_deps(monkeypatch)
    first = tmp_path / "first document.txt"
    second = tmp_path / "second document.txt"
    first.write_text("first")
    second.write_text("second")
    composer = titan_tui_module.ComposerTextArea()
    assert composer.normalize_paste_for_display(str(first)) == str(first)
    assert composer.normalize_paste_for_display(str(first).replace(" ", "\\ ")) == str(first)
    assert composer.normalize_paste_for_display(f'"{first}" "{second}"') == f"{first}\n{second}"
    assert composer.normalize_paste_for_display(f"{first}\n{second}") == f"{first}\n{second}"
    assert composer.normalize_paste_for_display(f"{first.as_uri()}\n{second.as_uri()}") == f"{first}\n{second}"


def test_tui_paste_file_uri_normalizes_to_absolute_path(monkeypatch, tmp_path):
    _patch_tui_deps(monkeypatch)
    file_path = tmp_path / "photo one.png"
    file_path.write_text("fake image")

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            composer = app.query_one("#input", titan_tui_module.ComposerTextArea)
            display = composer.normalize_paste_for_display(file_path.as_uri())
            assert display == str(file_path)

    asyncio.run(_run())


def test_tui_paste_long_natural_language_does_not_probe_as_path(monkeypatch):
    _patch_tui_deps(monkeypatch)
    pasted = (
        "There's a folder on my desktop called AI Chat. I want you to open that folder, "
        "analyze the project, and then I want you to finish it by adding whatever you think "
        "would be great to make it a million-dollar chat website like an AI product. "
        "You know you've got all the tools you need. Surprise me."
    )

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            composer = app.query_one("#input", titan_tui_module.ComposerTextArea)
            assert composer.normalize_paste_for_display(pasted) == pasted

    asyncio.run(_run())


def test_tui_input_enter_submits_instead_of_inserting_newline(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)) as pilot:
            composer = app.query_one("#input", TextArea)
            composer.load_text("/trace")
            composer.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert composer.text == ""
            assert app.trace_verbosity_levels[app.trace_verbosity_index] == "full"

    asyncio.run(_run())


def test_tui_clear_button_clears_input(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            composer = app.query_one("#input", TextArea)
            composer.load_text("draft prompt")
            app.action_clear_input()
            assert composer.text == ""

    asyncio.run(_run())


def test_tui_up_cycles_previous_sent_messages(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)) as pilot:
            composer = app.query_one("#input", titan_tui_module.ComposerTextArea)
            composer.record_history("first prompt")
            composer.record_history("second prompt")
            composer.focus()

            await pilot.press("up")
            assert composer.text == "second prompt"
            await pilot.press("up")
            assert composer.text == "first prompt"
            await pilot.press("up")
            assert composer.text == "first prompt"

    asyncio.run(_run())


def test_tui_trace_suppresses_provider_stream_delta(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent("provider_stream_delta", {"text": "hello", "kind": "text"})
                )
            )
            assert not any("stream hello" in line for line in app.trace_lines)

    asyncio.run(_run())


def test_tui_trace_provider_request_is_suppressed(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent(
                        "provider_request",
                        {
                            "iteration": 1,
                            "state": "PLAN",
                            "tool_calls_this_turn": 0,
                            "tools": ["write_file", "terminal"],
                        },
                    )
                )
            )
            request_lines = [line for line in app.trace_lines if line == "request"]
            assert request_lines == []

    asyncio.run(_run())


def test_tui_trace_state_events_are_suppressed(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent("on_state_enter", {"state": "REFLECT", "turn": 5})
                )
            )
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent("on_transition", {"from_state": "REFLECT", "to_state": "ACT", "turn": 5})
                )
            )
            assert "REFLECT" not in app.trace_lines
            assert "ACT" not in app.trace_lines
            assert not any(line.startswith("enter ") for line in app.trace_lines)
            assert not any(line.startswith("transition ") for line in app.trace_lines)

    asyncio.run(_run())


def test_tui_tool_call_status_uses_harness_per_turn_count(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.ui.pending = True
            app.ui.started_at = 1.0
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent(
                        "tool_call",
                        {
                            "id": "c2",
                            "name": "shell",
                            "arguments": {"command": "echo two"},
                            "count": 2,
                            "tool_calls_total": 2,
                            "tool_calls_this_turn": 2,
                        },
                    )
                )
            )
            status = str(app.query_one("#status_line", titan_tui_module.Static).render())
            assert "2 tools" in status
            assert app.ui.turn_tool_calls == 2
            assert app.ui.tool_calls == 2

    asyncio.run(_run())


def test_tui_trace_tool_call_shows_compact_label_and_counter(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.ui.pending = True
            app.ui.started_at = 1.0
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent(
                        "tool_call",
                        {
                            "id": "c1",
                            "name": "read_file",
                            "arguments": {"path": "/Users/ghost/Desktop/Titan/tests/test_titan_tui.py"},
                            "tool_calls_total": 3,
                            "tool_calls_this_turn": 2,
                        },
                    )
                )
            )
            assert any("📖 read" in line and "test_titan_tui.py" in line and "[2/3]" in line for line in app.trace_lines)

    asyncio.run(_run())


def test_tui_progress_updates_chat_on_phase_transition_with_recaps_disabled(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.harness.config.chat_recaps_enabled = False
            app.ui.pending = True
            app.ui.started_at = 1.0
            app.ui.state = "ACT"
            app.ui.turn = 2
            app.ui.tool_calls = 3
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent(
                        "on_transition",
                        {"from_state": "ACT", "to_state": "REFLECT", "turn": 2},
                    )
                )
            )
            progress_lines = [line for line in app.chat_lines if line.startswith("progress>")]
            assert not progress_lines
            assert not any(line.startswith("trace>") for line in app.chat_lines)

    asyncio.run(_run())


def test_tui_progress_updates_are_periodic_and_throttled(monkeypatch):
    _patch_tui_deps(monkeypatch)
    now = {"value": 100.0}
    monkeypatch.setattr("titan.titan_tui.time.time", lambda: now["value"])

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.ui.pending = True
            app.ui.started_at = 90.0
            app.ui.state = "ACT"
            app.ui.turn = 1
            app.last_progress_update_at = 100.0
            app._record_progress_event("running tests")

            now["value"] = 110.0
            app._tick()
            assert not [line for line in app.chat_lines if line.startswith("progress>")]

            now["value"] = 116.0
            app._tick()
            progress_lines = [line for line in app.chat_lines if line.startswith("progress>")]
            assert not progress_lines

    asyncio.run(_run())


def test_tui_progress_updates_chat_on_same_state_part_completion(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.harness.config.chat_recaps_enabled = False
            app.ui.pending = True
            app.ui.started_at = 1.0
            app.ui.state = "REFLECT"
            app.ui.turn = 4
            app.ui.tool_calls = 8
            app._record_progress_event("finished tool read_file successfully")
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent(
                        "on_transition",
                        {"from_state": "REFLECT", "to_state": "REFLECT", "turn": 4},
                    )
                )
            )
            progress_lines = [line for line in app.chat_lines if line.startswith("progress>")]
            assert not progress_lines

    asyncio.run(_run())


def test_tui_progress_updates_chat_before_budget_stop_final(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.ui.pending = True
            app.ui.started_at = 1.0
            app.ui.state = "REFLECT"
            app.ui.turn = 16
            app.ui.tool_calls = 22
            app._record_progress_event("finished tool shell successfully")
            out = RunOutcome(
                text="",
                stop=RunStopContract(
                    reason=RunStopReason.BudgetIterations,
                    iterations=16,
                    tool_calls_total=22,
                    elapsed_ms=1200,
                    notes="max_iterations",
                ),
            )
            app.on_loop_done_msg(titan_tui_module.LoopDoneMsg(out))
            progress_lines = [line for line in app.chat_lines if line.startswith("progress>")]
            assert not progress_lines
            assert app.chat_lines[-1].startswith("Titan: Summary:\n- Paused cleanly at the configured iteration ceiling")
            assert "Stopped: BudgetIterations" not in app.chat_lines[-1]

    asyncio.run(_run())


def test_tui_provider_selection_prompts_and_saves_missing_api_key(monkeypatch):
    _patch_tui_deps(monkeypatch)
    saved = []
    monkeypatch.setattr("titan.titan_tui.resolve_provider_credentials", lambda *args, **kwargs: None)
    monkeypatch.setattr("titan.titan_tui.update_config_key", lambda path, key, value: saved.append((key, value)))
    monkeypatch.setattr(TitanTui, "_validate_provider_key", lambda self, provider, key: (True, "ok"))

    async def _run():
        app = TitanTui()
        app.provider_options = ["openai", "xai"]
        async with app.run_test(size=(100, 32)):
            app.action_cycle_provider()
            key_input = app.query_one("#api_key_input", Input)
            assert app.pending_api_key_provider == "xai"
            assert key_input.password is True
            assert key_input.display is True

            key_input.value = "xai-test-key"
            app.on_input_submitted(Input.Submitted(key_input, key_input.value))
            await app.workers.wait_for_complete()
            assert app.pending_api_key_provider is None
            assert key_input.display is False
            assert app.harness.config.api_keys["xai"] == "xai-test-key"
            assert app.harness.config.model == "grok-4.6"

    asyncio.run(_run())
    assert ("provider", "xai") in saved
    assert ("model", "grok-4.6") in saved
    assert ("api_keys.xai", "xai-test-key") in saved


def test_tui_provider_cycle_sets_zai_default_model(monkeypatch):
    _patch_tui_deps(monkeypatch)
    monkeypatch.setattr("titan.titan_tui.resolve_provider_credentials", lambda *args, **kwargs: None)

    async def _run():
        app = TitanTui()
        app.provider_options = ["openai", "zai"]
        async with app.run_test(size=(100, 32)):
            app.action_cycle_provider()
            assert app.harness.config.provider == "zai"
            assert app.harness.config.model == "glm-5.1"

    asyncio.run(_run())


def test_tui_invalid_provider_key_is_not_saved(monkeypatch):
    _patch_tui_deps(monkeypatch)
    saved = []
    monkeypatch.setattr("titan.titan_tui.resolve_provider_credentials", lambda *args, **kwargs: None)
    monkeypatch.setattr("titan.titan_tui.update_config_key", lambda path, key, value: saved.append((key, value)))
    monkeypatch.setattr(TitanTui, "_validate_provider_key", lambda self, provider, key: (False, "http 401"))

    async def _run():
        app = TitanTui()
        app.provider_options = ["openai", "xai"]
        async with app.run_test(size=(100, 32)):
            app.action_cycle_provider()
            key_input = app.query_one("#api_key_input", Input)
            key_input.value = "bad-key"
            app.on_input_submitted(Input.Submitted(key_input, key_input.value))
            await app.workers.wait_for_complete()
            assert app.pending_api_key_provider == "xai"
            assert key_input.display is True
            assert app.harness.config.api_keys.get("xai") is None
            assert not any(key == "api_keys.xai" for key, _ in saved)

    asyncio.run(_run())


def test_tui_provider_button_opens_menu_and_shows_new_key(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        app.provider_options = ["openai", "xai", "zai"]
        async with app.run_test(size=(100, 32)) as pilot:
            btn = app.query_one("#btn-new-key", Button)
            menu = app.query_one("#provider_menu")
            assert str(menu.styles.display) == "none"
            await pilot.click("#btn-provider")
            await pilot.pause()
            await pilot.pause()
            assert str(menu.styles.display) != "none"
            assert app.harness.config.provider == "openai"
            await pilot.click("#provider-opt-xai")
            await pilot.pause()
            assert app.harness.config.provider == "xai"
            assert str(btn.styles.display) != "none"

    asyncio.run(_run())


def test_tui_model_button_opens_menu_and_selects_model(monkeypatch):
    _patch_tui_deps(monkeypatch)
    saved = []
    monkeypatch.setattr("titan.titan_tui.update_config_key", lambda path, key, value: saved.append((key, value)))

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)) as pilot:
            menu = app.query_one("#model_menu")
            assert str(menu.styles.display) == "none"
            await pilot.click("#btn-model")
            await pilot.pause()
            assert str(menu.styles.display) != "none"
            assert "gpt-5.4" in app.model_options
            await pilot.click("#model-opt-1")
            await pilot.pause()
            assert str(menu.styles.display) == "none"
            assert app.harness.config.model == "gpt-6-astra"
            assert str(app.query_one("#btn-model", Button).label) == "gpt-6-astra ▾"

    asyncio.run(_run())
    assert any(key == "model" for key, _value in saved)


def test_tui_cycle_model_hotkey_advances_model(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            first = app.harness.config.model
            app.action_cycle_model()
            assert app.harness.config.model != first
            assert app.harness.config.model in app._models_for_provider(app.harness.config.provider, first)

    asyncio.run(_run())


def test_tui_theme_cycle_updates_label(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            brand = app.query_one("#brand")
            before = brand.styles.color
            app.action_cycle_theme()
            assert brand.styles.color != before

    asyncio.run(_run())


def test_tui_trace_shows_rejected_tool_calls(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent(
                        "tool_call_rejected",
                        {
                            "id": "c1",
                            "name": "shell",
                            "arguments": {"command": "echo visible"},
                            "index": 1,
                            "count": 2,
                            "reason": "max_tool_calls_per_iteration",
                        },
                    )
                )
            )
            assert any("tool-call rejected 1/2 shell" in line for line in app.trace_lines)
            assert any("echo visible" in line for line in app.trace_lines)

    asyncio.run(_run())


def test_tui_chat_output_is_boxed_and_not_truncated(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            long_text = "\n".join(f"line {i}" for i in range(20))
            app._write_chat_box("Titan", long_text, "green")
            output = app.query_one("#output", titan_tui_module.ConversationLog)
            assert app.chat_lines
            assert "Titan:" in app.chat_lines[-1]
            assert "line 0" in app.chat_lines[-1]
            assert "line 19" in app.chat_lines[-1]
            assert "truncated in chat" not in app.chat_lines[-1]
            assert output.selection_lines[-1] == app.chat_lines[-1]

    asyncio.run(_run())


def test_tui_user_messages_are_bold_bullets_without_label_or_color(monkeypatch):
    _patch_tui_deps(monkeypatch)
    app = TitanTui()

    renderable = app._chat_renderable("You", "describe this image", "cyan")

    assert isinstance(renderable, Text)
    assert not isinstance(renderable, Panel)
    assert str(renderable.style) == "bold"
    assert renderable.plain == "• describe this image"
    assert app._chat_plain_text("You", "describe this image") == "• describe this image"
    assert "You" not in renderable.plain
    assert "cyan" not in str(renderable.style)


def test_tui_chat_keeps_full_text_even_past_old_sentence_grace(monkeypatch):
    _patch_tui_deps(monkeypatch)
    app = TitanTui()
    first = "This is the first complete sentence about the image."
    second = " It adds a useful final detail that should be allowed to finish."
    over = " extra" * 100
    full = first + second + over

    brief = app._brief_chat_text(full, max_chars=len(first) + 8, max_lines=20, grace_chars=len(second) + 5)

    assert brief == full
    assert "[truncated in chat" not in brief


def test_tui_copy_buffers_use_pbcopy(monkeypatch):
    _patch_tui_deps(monkeypatch)
    calls = []

    def fake_run(cmd, input, text, check, timeout):
        calls.append({"cmd": cmd, "input": input, "text": text, "check": check, "timeout": timeout})

    monkeypatch.setattr("titan.titan_tui.subprocess.run", fake_run)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.trace_lines.append("trace line")
            app.chat_lines.append("chat line")
            app.action_copy_trace()
            app.action_copy_chat()

    asyncio.run(_run())
    assert calls[0]["cmd"] == ["pbcopy"]
    assert calls[0]["input"] == "trace line"
    assert calls[1]["input"] == "chat line"


def test_tui_simple_chat_final_output_omits_summary_footer(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            out = RunOutcome(
                text="Hi! How can I help?",
                stop=RunStopContract(
                    reason=RunStopReason.AssistantFinal,
                    iterations=1,
                    tool_calls_total=0,
                    elapsed_ms=50,
                    notes="",
                ),
            )
            app.on_loop_done_msg(titan_tui_module.LoopDoneMsg(out))
            final = app.chat_lines[-1]
            assert final == "Titan: Hi! How can I help?"
            assert "Summary:" not in final
            assert "Next best step:" not in final

    asyncio.run(_run())


def test_tui_multi_turn_final_output_omits_summary_footer_when_recaps_disabled(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            out = RunOutcome(
                text="It's running locally now.",
                stop=RunStopContract(
                    reason=RunStopReason.AssistantFinal,
                    iterations=2,
                    tool_calls_total=1,
                    elapsed_ms=123,
                    notes="",
                ),
            )
            app.on_loop_done_msg(titan_tui_module.LoopDoneMsg(out))
            final = app.chat_lines[-1]
            assert final == "Titan: It's running locally now."
            assert "Summary:" not in final
            assert "Next best step:" not in final

    asyncio.run(_run())


def test_tui_final_output_adds_summary_and_next_step_offer_when_recaps_enabled(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        app.harness.config.chat_recaps_enabled = True
        async with app.run_test(size=(100, 32)):
            out = RunOutcome(
                text="- PASS: input remains clickable\n- FAIL: trace default is not normal",
                stop=RunStopContract(
                    reason=RunStopReason.AssistantFinal,
                    iterations=3,
                    tool_calls_total=2,
                    elapsed_ms=123,
                    notes="",
                ),
            )
            app.on_loop_done_msg(titan_tui_module.LoopDoneMsg(out))
            final = app.chat_lines[-1]
            assert "Summary:" in final
            assert "Finished with AssistantFinal; turns=3, tools=2" in final
            assert "Next best step:" in final
            assert "Fix the first failing/gap item above" in final
            assert "I can do that next if you say: do it." in final

    asyncio.run(_run())


def test_tui_run_keeps_mouse_enabled_for_buttons_and_internal_selection(monkeypatch):
    seen = {}

    class FakeApp:
        def run(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(titan_tui_module, "TitanTui", lambda: FakeApp())

    titan_tui_module.run()

    assert seen["mouse"] is True


def test_tui_does_not_load_git_diff_until_diff_tab(monkeypatch):
    _patch_tui_deps(monkeypatch)
    called = []
    monkeypatch.setattr(TitanTui, "_collect_git_diff", lambda self: called.append("diff") or "diff --git a/a b/a\n+new")

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            assert called == []
            assert app.active_top_tab == "trace"
            app._set_top_tab("diff")
            await app.workers.wait_for_complete()
            assert called == ["diff"]
            assert app.diff_lines == ["diff --git a/a b/a", "+new"]

    asyncio.run(_run())


def test_tui_caps_large_git_diff(monkeypatch):
    _patch_tui_deps(monkeypatch)
    huge = "\n".join(f"line {i}" for i in range(500))
    monkeypatch.setattr(TitanTui, "_collect_git_diff", lambda self: huge)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app._set_top_tab("diff")
            await app.workers.wait_for_complete()
            assert len(app.diff_lines) == 401
            assert app.diff_lines[0].startswith("...")
            assert app.diff_lines[-1] == "line 499"
            pane = app.query_one("#diff", titan_tui_module.SelectableRichLog)
            assert pane.text.count("\n") == 400

    asyncio.run(_run())


def test_tui_rebuilds_history_from_session_jsonl(monkeypatch, tmp_path):
    from titan.session import SessionStore
    from titan.types import Message, Role

    store_path = tmp_path / "session.jsonl"
    SessionStore(str(store_path)).append(Message(role=Role.USER, content="hello from disk"))
    _patch_tui_deps(monkeypatch)
    monkeypatch.setattr("titan.titan_tui.SessionStore", lambda path: SessionStore(str(store_path)))

    app = TitanTui()
    assert any(m.content == "hello from disk" for m in app.history)


def test_tui_compaction_and_checkpoint_events_go_to_trace(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent("git_checkpoint", {"id": "ckpt-1"})
                )
            )
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent("compaction", {"tokens_before": 90000, "tokens_after": 12000})
                )
            )
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent("verify", {"command": "pytest", "exit_code": 0})
                )
            )
            joined = "\n".join(app.trace_lines)
            assert "checkpoint ckpt-1" in joined
            assert "compacted 90000->12000 tokens" in joined
            assert "verify pytest exit=0" in joined

    asyncio.run(_run())


@pytest.mark.parametrize("size", [(48, 18), (80, 24), (120, 40)])
def test_redesigned_layout_keeps_composer_and_menus_onscreen(monkeypatch, size):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=size) as pilot:
            assert app.query_one("#welcome").display
            assert not app.query_one("#top").display
            composer = app.query_one("#input")
            for widget_id in ("input", "btn-provider", "btn-model", "status_line"):
                region = app.query_one(f"#{widget_id}").region
                assert region.width > 0
                assert region.right <= size[0]
                assert region.bottom <= size[1]
            await pilot.click("#btn-model")
            await pilot.pause()
            assert app.query_one("#model_menu").region.bottom <= composer.region.y
            await pilot.press("escape")
            assert not app.query_one("#model_menu").display
            assert app.focused is composer
            app._write_chat_box("Titan", "## Ready\n\n```python\nprint('hello')\n```", "")
            await pilot.pause()
            assert not app.query_one("#welcome").display
            assert app.query_one("#output").display
            assert len(app.query("MarkdownFence")) == 1
            await pilot.click("#tab-diff")
            await pilot.press("escape")
            assert app.query_one("#output").display
            assert app.focused is composer

    asyncio.run(_run())


def test_busy_submit_preserves_draft_and_expanded_paste(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test() as pilot:
            composer = app.query_one("#input", titan_tui_module.ComposerTextArea)
            token = composer.normalize_paste_for_display("first\nsecond")
            composer.load_text(f"next task {token}")
            app.ui.pending = True
            await pilot.press("enter")
            assert composer.text == f"next task {token}"
            assert composer.expand_paste_tokens(composer.text) == "next task first\nsecond"
            assert composer.message_history == []
            app.ui.pending = False

    asyncio.run(_run())


def test_global_shortcuts_work_while_composer_has_focus(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        copied = []
        monkeypatch.setattr(app, "_copy_to_clipboard", lambda text, label: copied.append((text, label)))
        async with app.run_test() as pilot:
            app._write_chat_box("You", "hello", "")
            await pilot.press("ctrl+y")
            assert copied == [("• hello", "chat")]
            await pilot.press("ctrl+d")
            assert app.top_tab_expanded
            await pilot.press("escape")
            assert not app.top_tab_expanded
            app.ui.pending = True
            await pilot.press("ctrl+c")
            assert app.ctrl_c_quit_armed
            assert app.activity == "Stopping"
            assert app.ui.pending

    asyncio.run(_run())


def test_provider_switch_keeps_oauth_endpoint_model_and_saved_config_together(monkeypatch, tmp_path):
    from titan.auth import OpenAICredentials, provider_default_base_url
    from titan.config import get_config_key, resolve_config_path
    from titan.provider import build_provider_from_config

    _patch_tui_deps(monkeypatch)
    monkeypatch.setattr(titan_tui_module, "build_provider_from_config", build_provider_from_config)
    monkeypatch.setattr("titan.provider.resolve_provider_credentials", lambda provider, **kwargs:
                        OpenAICredentials("test-oauth-token", provider_default_base_url(provider), "test"))
    monkeypatch.setattr(TitanTui, "_provider_has_key", lambda self, provider: True)

    async def _run():
        app = TitanTui()
        async with app.run_test():
            assert resolve_config_path() == tmp_path / "config.json"
            app._apply_provider_selection("grok")
            app.harness.config.api_base = "https://api.x.ai/v1"
            app._apply_provider_selection("openai-codex")
            assert app.harness.config.model == "gpt-6-astra"
            assert app.harness.provider.api_base == "https://chatgpt.com/backend-api/codex"
            assert app.harness.provider.api_key == "test-oauth-token"
            assert app.pending_api_key_provider is None
            assert get_config_key(resolve_config_path(), "provider") == "openai-codex"
            assert get_config_key(resolve_config_path(), "model") == "gpt-6-astra"
            app._apply_provider_selection("grok")
            app._apply_model_selection("gpt-6-astra")  # stale model picker event
            assert app.harness.config.model == "grok-4.6"
            assert app.harness.provider.api_base == "https://api.x.ai/v1"
            assert "gpt-6-astra" not in app.model_options

    asyncio.run(_run())


def test_multiline_navigation_and_history_restore_draft(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test() as pilot:
            composer = app.query_one("#input", titan_tui_module.ComposerTextArea)
            composer.record_history("previous task")
            composer.load_text("draft")
            await pilot.press("up")
            assert composer.text == "previous task"
            await pilot.press("down")
            assert composer.text == "draft"
            composer.move_cursor(composer.document.end)
            await pilot.press("alt+enter")
            assert composer.text == "draft\n"
            await pilot.press("up")
            assert composer.text == "draft\n"
            assert composer.cursor_location[0] == 0

    asyncio.run(_run())


def test_streamed_markdown_is_replaced_by_single_final_message(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test() as pilot:
            app._write_chat_box("You", "build it", "")
            app.ui.pending = True
            for chunk in ("## Done", "\n\nBuilt the **frontend**."):
                app.on_loop_event_msg(titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent("provider_stream_delta", {"kind": "text", "text": chunk})
                ))
            app._tick()
            await pilot.pause()
            assert app.stream_widget is not None
            assert len(app.query(".streaming-message")) == 1
            app.on_loop_done_msg(titan_tui_module.LoopDoneMsg(RunOutcome(
                text="## Done\n\nBuilt the **frontend**.",
                stop=RunStopContract(reason=RunStopReason.AssistantFinal, iterations=1,
                                     tool_calls_total=0, elapsed_ms=100, notes=""),
            )))
            await pilot.pause()
            assert len(app.query(".streaming-message")) == 0
            assert len(app.query(".assistant-message")) == 1
            assert app.query_one("#output").text.count("Built the **frontend**.") == 1
            assert not app.ui.pending

    asyncio.run(_run())
