import sys

import titan.titan_cli as titan_cli
import titan.titan_tui as titan_tui


def test_titan_without_subcommand_launches_tui(monkeypatch):
    called = []

    monkeypatch.setattr(sys, "argv", ["titan"])
    monkeypatch.setattr(titan_tui, "run", lambda: called.append("run"))

    titan_cli.main()

    assert called == ["run"]
