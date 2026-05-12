Hermes-grade TUI paced-logic extraction (from original Ferroclaw)

Source:
- /Users/ghost/Desktop/Projects/ferroclaw/src/tui/hermes_tui.rs

Why the original felt good
1) Alternate-screen isolation + raw mode + mouse capture
   - Prevents shell scrollback bleed and frame corruption.
   - Keeps scroll behavior deterministic in-app.

2) Explicit transcript scroll model (not terminal scrollback)
   - PageUp/PageDown jump by larger chunks.
   - Shift+Up/Shift+Down fine-grain by 1.
   - Ctrl+Home/Ctrl+End hard jumps.
   - Wheel maps to app scroll_up/scroll_down.

3) Pacing tied to run lifecycle
   - run_started_at captured at dispatch.
   - elapsed_ms_since(run_started_at) drives status verb updates.
   - Distinct pending/working states so user sees progress, not lockups.

4) Event-driven streaming updates
   - AgentEvent channel feeds UI incrementally.
   - apply_agent_event() updates transcript/status as events arrive.
   - Tool lifecycle appears in real time, not only at end.

5) Careful slash/menu state machine
   - Separate mode/query/menu selection state.
   - Input composer and slash/model query state are isolated.
   - Predictable accept/cancel semantics avoid accidental input pollution.

6) Forced bottom anchoring after meaningful state transitions
   - scroll_to_bottom() after sends, completion, and key transitions.
   - Avoids "where did my reply go?" syndrome.

Port target for Titan
- Keep Python backend loop.
- Rebuild frontend as explicit state machine (not simplistic line log).
- Implement deterministic scroll model + event callback bridge.
- Mirror run_started_at-driven pacing and status verbs.
- Preserve slash/model mode separation (if/when slash commands added).

Acceptance criteria for parity
- Long sessions remain scrollable without frame drift.
- Wheel/PageUp/PageDown/Home/End all work predictably.
- Tool calls stream into transcript as they happen.
- Status line shows elapsed-time-based pacing states.
- No shell bleed into chat frame.
