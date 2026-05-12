import json

from titan.tools import default_registry


def test_todo_roundtrip(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reg = default_registry()

    r1 = reg.execute("c1", "todo_get", {})
    assert r1.is_error is False
    assert json.loads(r1.content) == {"todos": []}

    payload = {
        "todos": [
            {"id": "t1", "content": "build parity", "status": "in_progress"},
            {"id": "t2", "content": "ship e2e", "status": "pending"},
        ]
    }
    r2 = reg.execute("c2", "todo_set", payload)
    assert r2.is_error is False
    assert json.loads(r2.content)["saved"] == 2

    r3 = reg.execute("c3", "todo_get", {})
    data = json.loads(r3.content)
    assert len(data["todos"]) == 2
    assert data["todos"][0]["id"] == "t1"
