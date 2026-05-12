from __future__ import annotations

from pathlib import Path

from titan.image_preprocess import preprocess_image_for_attachment


def test_preprocess_image_falls_back_for_invalid_bytes(tmp_path: Path):
    image = tmp_path / "bad.png"
    image.write_bytes(b"not-an-image")

    mime, data = preprocess_image_for_attachment(image)

    assert mime == "image/png"
    assert data == b"not-an-image"


def test_preprocess_image_supports_jpeg_mime_fallback(tmp_path: Path):
    image = tmp_path / "bad.jpg"
    image.write_bytes(b"not-an-image")

    mime, data = preprocess_image_for_attachment(image)

    assert mime == "image/jpeg"
    assert data == b"not-an-image"
