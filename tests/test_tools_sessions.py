import json
from pathlib import Path

from titan.tools import default_registry


def test_session_recent_and_search(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    p = Path('.titan/session.jsonl')
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"ts": 100, "trace_id": "aaa111", "role": "user", "content": "Plan parity milestone"},
        {"ts": 110, "trace_id": "aaa111", "role": "assistant", "content": "Done"},
        {"ts": 200, "trace_id": "bbb222", "role": "user", "content": "Run accuracy eval"},
        {"ts": 210, "trace_id": "bbb222", "role": "assistant", "content": "4/4 passed"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    reg = default_registry()

    recent = reg.execute("s1", "session_recent", {"limit": 1})
    assert recent.is_error is False
    recent_items = json.loads(recent.content)["items"]
    assert len(recent_items) == 1
    assert recent_items[0]["trace_id"] == "bbb222"

    found = reg.execute("s2", "session_search", {"query": "parity"})
    assert found.is_error is False
    found_items = json.loads(found.content)["items"]
    assert len(found_items) == 1
    assert found_items[0]["trace_id"] == "aaa111"


def test_session_search_requires_query(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reg = default_registry()
    r = reg.execute("s3", "session_search", {"query": ""})
    assert r.is_error is True
    assert "query is required" in r.content
