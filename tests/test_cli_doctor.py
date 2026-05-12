from titan.titan_cli import cmd_doctor


def test_doctor_ok_when_tui_present_and_venv_matches(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str((tmp_path / ".venv").resolve()))

    def _which(name: str):
        if name == "titan":
            return "/tmp/.venv/bin/titan"
        if name == "titan-tui":
            return "/tmp/.venv/bin/titan-tui"
        return None

    monkeypatch.setattr("shutil.which", _which)
    assert cmd_doctor() == 0


def test_doctor_fails_when_tui_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str((tmp_path / ".venv").resolve()))

    def _which(name: str):
        if name == "titan":
            return "/tmp/.venv/bin/titan"
        return None

    monkeypatch.setattr("shutil.which", _which)
    assert cmd_doctor() == 1
