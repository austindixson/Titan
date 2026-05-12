import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from titan.titan_cli import cmd_sessions_recent, cmd_sessions_search


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_sessions_recent_and_search(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    rows = [
        {"ts": 1000, "trace_id": "aaa", "role": "user", "content": "build titan parity"},
        {"ts": 1001, "trace_id": "aaa", "role": "assistant", "content": "working"},
        {"ts": 2000, "trace_id": "bbb", "role": "user", "content": "fix terminal layout"},
        {"ts": 2001, "trace_id": "bbb", "role": "assistant", "content": "done"},
    ]
    _write_rows(tmp_path / ".titan" / "session.jsonl", rows)

    b1 = io.StringIO()
    with redirect_stdout(b1):
        rc = cmd_sessions_recent(limit=2)
    out1 = b1.getvalue()
    assert rc == 0
    assert "bbb" in out1
    assert "aaa" in out1

    b2 = io.StringIO()
    with redirect_stdout(b2):
        rc = cmd_sessions_search("layout", limit=5)
    out2 = b2.getvalue()
    assert rc == 0
    assert "bbb" in out2
    assert "score=" in out2


def test_sessions_search_no_match(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _write_rows(
        tmp_path / ".titan" / "session.jsonl",
        [{"ts": 1, "trace_id": "x", "role": "user", "content": "hello"}],
    )

    b = io.StringIO()
    with redirect_stdout(b):
        rc = cmd_sessions_search("unmatched", limit=3)
    assert rc == 0
    assert "no matches" in b.getvalue()
