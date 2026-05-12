import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from titan.titan_cli import (
    cmd_memory_add,
    cmd_memory_get,
    cmd_memory_remove,
    cmd_todo_get,
    cmd_todo_set,
)


def test_todo_cli_roundtrip(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    todos = [
        {"id": "a", "content": "first", "status": "pending"},
        {"id": "b", "content": "second", "status": "completed"},
    ]

    out = io.StringIO()
    with redirect_stdout(out):
        rc = cmd_todo_set(json.dumps(todos))
    assert rc == 0

    out = io.StringIO()
    with redirect_stdout(out):
        rc = cmd_todo_get()
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert len(payload["todos"]) == 2


def test_memory_cli_roundtrip(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    with redirect_stdout(io.StringIO()):
        assert cmd_memory_add("User prefers concise updates") == 0
    with redirect_stdout(io.StringIO()):
        assert cmd_memory_add("Titan project root is Desktop/Titan") == 0

    out = io.StringIO()
    with redirect_stdout(out):
        assert cmd_memory_get("concise") == 0
    data = json.loads(out.getvalue())
    assert len(data["entries"]) == 1

    with redirect_stdout(io.StringIO()):
        assert cmd_memory_remove("project root") == 0

    out = io.StringIO()
    with redirect_stdout(out):
        assert cmd_memory_get() == 0
    data = json.loads(out.getvalue())
    assert len(data["entries"]) == 1
