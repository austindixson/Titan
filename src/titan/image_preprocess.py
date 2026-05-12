from __future__ import annotations

import io
import mimetypes
from pathlib import Path

MAX_WIDTH = 2000
MAX_HEIGHT = 2000
MAX_BASE64_BYTES = int(4.5 * 1024 * 1024)
JPEG_QUALITY_STEPS = (80, 85, 70, 55, 40)


def _encoded_size(data: bytes) -> int:
    # base64 expands ~4/3
    return ((len(data) + 2) // 3) * 4


def _guess_mime(path: Path, fallback: str = "image/png") -> str:
    return mimetypes.guess_type(str(path))[0] or fallback


def preprocess_image_for_attachment(path: Path) -> tuple[str, bytes]:
    """Best-effort Pi-like image preprocessing for provider attachments.

    - EXIF orientation normalization (if Pillow available)
    - Resize toward <= 2000x2000
    - Prefer candidate encoding under ~4.5MB base64 payload budget

    Falls back to original bytes if Pillow is unavailable or processing fails.
    """
    original_bytes = path.read_bytes()
    original_mime = _guess_mime(path)

    if _encoded_size(original_bytes) <= MAX_BASE64_BYTES:
        return original_mime, original_bytes

    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception:
        return original_mime, original_bytes

    try:
        img = Image.open(io.BytesIO(original_bytes))
        img = ImageOps.exif_transpose(img)

        w, h = img.size
        scale = min(1.0, MAX_WIDTH / max(1, w), MAX_HEIGHT / max(1, h))
        current = img
        if scale < 1.0:
            current = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

        best_mime = original_mime
        best_bytes = original_bytes
        best_size = _encoded_size(original_bytes)

        def try_candidates(im: Image.Image) -> tuple[str, bytes, int] | None:
            candidates: list[tuple[str, bytes, int]] = []

            png_buf = io.BytesIO()
            im.save(png_buf, format="PNG")
            png_bytes = png_buf.getvalue()
            candidates.append(("image/png", png_bytes, _encoded_size(png_bytes)))

            rgb = im.convert("RGB")
            for q in JPEG_QUALITY_STEPS:
                jpg_buf = io.BytesIO()
                rgb.save(jpg_buf, format="JPEG", quality=q, optimize=True)
                jpg_bytes = jpg_buf.getvalue()
                candidates.append(("image/jpeg", jpg_bytes, _encoded_size(jpg_bytes)))

            under = [c for c in candidates if c[2] <= MAX_BASE64_BYTES]
            if under:
                return min(under, key=lambda c: c[2])
            return min(candidates, key=lambda c: c[2]) if candidates else None

        while True:
            picked = try_candidates(current)
            if picked is not None:
                pmime, pbytes, psz = picked
                if psz < best_size:
                    best_mime, best_bytes, best_size = pmime, pbytes, psz
                if psz <= MAX_BASE64_BYTES:
                    break

            cw, ch = current.size
            if cw == 1 and ch == 1:
                break
            nw = max(1, int(cw * 0.75))
            nh = max(1, int(ch * 0.75))
            if nw == cw and nh == ch:
                break
            current = current.resize((nw, nh), Image.Resampling.LANCZOS)

        return best_mime, best_bytes
    except Exception:
        return original_mime, original_bytes
