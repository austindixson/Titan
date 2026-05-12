from __future__ import annotations

from titan.path_resolver import resolve_existing_read_path


def test_resolve_existing_read_path_handles_macos_ampm_spacing_variant(tmp_path):
    real = tmp_path / "Screenshot 2026-05-12 at 5.47.12\u202fAM.png"
    real.write_bytes(b"x")

    pasted = str(real).replace("\u202fAM", "AM")
    resolved = resolve_existing_read_path(pasted)

    assert resolved is not None
    assert resolved == real.resolve()


def test_resolve_existing_read_path_supports_at_prefixed_path(tmp_path):
    real = tmp_path / "shot.png"
    real.write_bytes(b"x")

    resolved = resolve_existing_read_path(f"@{real}")
    assert resolved is not None
    assert resolved == real.resolve()


def test_resolve_existing_read_path_supports_file_uri(tmp_path):
    real = tmp_path / "shot with spaces.png"
    real.write_bytes(b"x")
    uri = f"file://{str(real).replace(' ', '%20')}"

    resolved = resolve_existing_read_path(uri)
    assert resolved is not None
    assert resolved == real.resolve()


def test_resolve_existing_read_path_returns_none_for_missing_file(tmp_path):
    missing = tmp_path / "Screenshot 2026-05-12 at 5.47.12AM.png"
    assert resolve_existing_read_path(str(missing)) is None
