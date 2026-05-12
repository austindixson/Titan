from __future__ import annotations

import json

from titan.tools import default_registry


def test_read_file_uses_registry_cwd_for_relative_paths(tmp_path):
    reg = default_registry()
    reg.cwd = tmp_path
    p = tmp_path / "notes.txt"
    p.write_text("a\nb\nc\n")

    out = reg.execute("c1", "read_file", {"path": "notes.txt", "offset": 2, "limit": 2})

    assert not out.is_error
    assert out.content.splitlines() == ["2|b", "3|c"]


def test_read_file_resolves_macos_screenshot_ampm_variant(tmp_path):
    reg = default_registry()
    reg.cwd = tmp_path

    # actual filename contains narrow no-break space before AM
    real = tmp_path / "Screenshot 2026-05-12 at 5.47.12\u202fAM.txt"
    real.write_text("ok")

    pasted = str(real).replace("\u202fAM", "AM")
    out = reg.execute("c1", "read_file", {"path": pasted})

    assert not out.is_error
    assert out.content.strip() == "1|ok"


def test_read_file_image_returns_structured_descriptor(tmp_path):
    reg = default_registry()
    reg.cwd = tmp_path

    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    out = reg.execute("c1", "read_file", {"path": str(img)})

    assert not out.is_error
    payload = json.loads(out.content)
    assert payload["type"] == "image_file"
    assert payload["mime"] == "image/png"
    assert payload["path"] == str(img.resolve())
