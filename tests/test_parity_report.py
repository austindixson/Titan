import json
from pathlib import Path

from titan.titan_cli import cmd_parity_report


def test_parity_report_writes_json(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "parity.json"
    rc = cmd_parity_report(out)
    assert rc == 0
    assert out.exists()

    data = json.loads(out.read_text())
    assert "provider" in data
    assert "model" in data
    assert "tools" in data and isinstance(data["tools"], list)
    assert "accuracy_eval" in data
    assert data["accuracy_eval"]["passed"] == data["accuracy_eval"]["total"]
