import json

from titan.tools import default_registry


def test_memory_add_get_remove(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reg = default_registry()

    r0 = reg.execute("m0", "memory_get", {})
    assert json.loads(r0.content) == {"entries": []}

    r1 = reg.execute("m1", "memory_add", {"content": "User prefers concise updates."})
    assert r1.is_error is False

    r2 = reg.execute("m2", "memory_add", {"content": "Project path is /Users/ghost/Desktop/Titan."})
    assert r2.is_error is False

    r3 = reg.execute("m3", "memory_get", {"query": "concise"})
    data = json.loads(r3.content)
    assert len(data["entries"]) == 1
    assert "concise" in data["entries"][0].lower()

    r4 = reg.execute("m4", "memory_remove", {"contains": "project path"})
    out = json.loads(r4.content)
    assert out["removed"] == 1

    r5 = reg.execute("m5", "memory_get", {})
    entries = json.loads(r5.content)["entries"]
    assert len(entries) == 1
    assert "concise" in entries[0].lower()
