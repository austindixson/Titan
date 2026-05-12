import asyncio

from rich.panel import Panel
from rich.text import Text
from textual.widgets import Button, Input, TextArea

from titan.config import HarnessConfig
from titan.mock_provider import MockProvider
from titan.titan_tui import TitanTui
from titan.types import RunOutcome, RunStopContract, RunStopReason
import titan.titan_tui as titan_tui_module


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
            }
            assert labels == {
                "tab-trace": "Trace ●",
                "tab-diff": "Diff",
                "btn-stop": "Stop",
                "btn-provider": "Provider: openai",
                "btn-clear": "Clear",
                "btn-trace": "Trace: normal",
                "btn-quit": "Quit",
            }

    asyncio.run(_run())


def test_tui_trace_defaults_normal_and_small(monkeypatch):
    _patch_tui_deps(monkeypatch)

    app = TitanTui()

    assert app.trace_verbosity_levels[app.trace_verbosity_index] == "normal"
    assert "#top {\n        height: 6;\n        min-height: 6;" in app.CSS


def test_tui_focus_input_hotkey_moves_off_ctrl_o(monkeypatch):
    _patch_tui_deps(monkeypatch)
    bindings = {(binding[0], binding[1]) for binding in TitanTui.BINDINGS}

    assert ("ctrl+f", "operator_input") in bindings
    assert ("ctrl+o", "operator_input") not in bindings


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
            assert app.active_top_tab == "diff"
            assert trace.display is False
            assert diff.display is True
            assert app.diff_lines == ["diff --git a/a b/a", "-old", "+new"]
            assert str(app.query_one("#tab-diff", Button).label) == "Diff ●"

    asyncio.run(_run())


def test_tui_trace_tab_click_expands_over_chat_and_click_again_minimizes(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)) as pilot:
            top = app.query_one("#top")
            output = app.query_one("#output", titan_tui_module.SelectableRichLog)
            trace = app.query_one("#trace", titan_tui_module.SelectableRichLog)

            assert app.active_top_tab == "trace"
            assert app.trace_expanded is False
            assert top.has_class("expanded") is False
            assert output.display is True
            assert trace.display is True

            await pilot.click("#tab-trace")
            await pilot.pause()
            assert app.trace_expanded is True
            assert top.has_class("expanded") is True
            assert output.has_class("trace-hidden") is True
            assert str(app.query_one("#tab-trace", Button).label) == "Trace ▾"

            await pilot.click("#tab-trace")
            await pilot.pause()
            assert app.trace_expanded is False
            assert top.has_class("expanded") is False
            assert output.has_class("trace-hidden") is False
            assert str(app.query_one("#tab-trace", Button).label) == "Trace ●"

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


def test_tui_paste_multiline_shows_line_count_but_expands_on_submit(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            composer = app.query_one("#input", titan_tui_module.ComposerTextArea)
            pasted = "alpha\nbeta\ngamma"

            display = composer.normalize_paste_for_display(pasted)

            assert display == "[pasted 3 lines #1]"
            assert composer.expand_paste_tokens(f"use {display}") == f"use {pasted}"

    asyncio.run(_run())


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
            assert composer.text == "second prompt"

    asyncio.run(_run())


def test_tui_trace_renders_provider_stream_delta(monkeypatch):
    _patch_tui_deps(monkeypatch)

    async def _run():
        app = TitanTui()
        async with app.run_test(size=(100, 32)):
            app.on_loop_event_msg(
                titan_tui_module.LoopEventMsg(
                    titan_tui_module.AgentEvent("provider_stream_delta", {"text": "hello", "kind": "text"})
                )
            )
            assert any("stream hello" in line for line in app.trace_lines)

    asyncio.run(_run())


def test_tui_trace_provider_request_shows_tools_used_not_available(monkeypatch):
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
            request_lines = [line for line in app.trace_lines if line.startswith("provider request")]
            assert request_lines == ["provider request iteration=1 state=PLAN tools_used_this_turn=0"]
            assert "tools_available" not in request_lines[0]
            assert "write_file" not in request_lines[0]
            assert "terminal" not in request_lines[0]

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
            assert "tools_used_this_turn=2" in status
            assert app.ui.turn_tool_calls == 2
            assert app.ui.tool_calls == 2

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
            assert progress_lines
            assert "finished ACT and moved to REFLECT" in progress_lines[-1]
            assert "total tools 3" in progress_lines[-1]
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
            assert len(progress_lines) == 1
            assert "running tests" in progress_lines[0]

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
            assert progress_lines
            assert "finished REFLECT and moved to REFLECT" in progress_lines[-1]
            assert "total tools 8" in progress_lines[-1]

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
            assert progress_lines
            assert "run stopped at BudgetIterations" in progress_lines[-1]
            assert app.chat_lines[-1] == "Titan: Stopped: BudgetIterations (max_iterations)"

    asyncio.run(_run())


def test_tui_provider_selection_prompts_and_saves_missing_api_key(monkeypatch):
    _patch_tui_deps(monkeypatch)
    saved = []
    monkeypatch.setattr("titan.titan_tui.resolve_provider_credentials", lambda *args, **kwargs: None)
    monkeypatch.setattr("titan.titan_tui.update_config_key", lambda path, key, value: saved.append((key, value)))

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
            assert app.pending_api_key_provider is None
            assert key_input.display is False
            assert app.harness.config.api_keys["xai"] == "xai-test-key"

    asyncio.run(_run())
    assert saved == [("api_keys.xai", "xai-test-key")]


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
            output = app.query_one("#output", titan_tui_module.SelectableRichLog)
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
