import asyncio

from textual.widgets import Button, TextArea

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
                "btn-operator": "Operator",
                "btn-trace": "Trace: compact",
                "btn-quit": "Quit",
            }

    asyncio.run(_run())


def test_tui_trace_defaults_compact_and_small(monkeypatch):
    _patch_tui_deps(monkeypatch)

    app = TitanTui()

    assert app.trace_verbosity_levels[app.trace_verbosity_index] == "compact"
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
            assert app.trace_verbosity_levels[app.trace_verbosity_index] == "normal"

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


def test_tui_chat_output_is_boxed_and_brevity_limited(monkeypatch):
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
            assert "line 12" not in app.chat_lines[-1]
            assert "truncated in chat" in app.chat_lines[-1]
            assert output.selection_lines[-1] == app.chat_lines[-1]

    asyncio.run(_run())


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
                text="- PASS: input remains clickable\n- FAIL: trace default is not compact",
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
