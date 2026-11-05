"""Portable rendering contract and strict PNG validation."""
from __future__ import annotations

import io
import zlib
from typing import Any, Protocol
from PIL import Image

class RenderBackend(Protocol):
    async def render(self, code: str, seed: int) -> dict[str, Any]: ...

RUNTIME_VERSION = "pycairo-v1"
OUTPUT_PATH_GUIDANCE = (
    'Save to the relative path output.png in the current working directory. '
    'Python previews and final validation rerun your program in a fresh directory; '
    'an absolute output path or a PNG copied beforehand will not satisfy this requirement.'
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PNG_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 8192


def validate_png(png: bytes, expected: dict[str, int] | None = None) -> dict[str, int]:
    """Validate PNG framing, chunk CRCs, dimensions, and compressed image data."""
    if len(png) < 24 or len(png) > MAX_PNG_BYTES or not png.startswith(PNG_SIGNATURE):
        raise ValueError("Invalid or oversized PNG")
    offset = len(PNG_SIGNATURE)
    first = True
    ended = False
    while offset + 12 <= len(png):
        size = int.from_bytes(png[offset:offset + 4], "big")
        end = offset + 12 + size
        if end > len(png):
            raise ValueError("Invalid PNG chunk length")
        kind = png[offset + 4:offset + 8]
        chunk = png[offset + 4:end - 4]
        if zlib.crc32(chunk) & 0xffffffff != int.from_bytes(png[end - 4:end], "big"):
            raise ValueError("Invalid PNG checksum")
        if first and (kind != b"IHDR" or size != 13):
            raise ValueError("Invalid PNG header")
        first = False
        offset = end
        if kind == b"IEND":
            if size != 0 or offset != len(png):
                raise ValueError("Invalid PNG ending")
            ended = True
            break
    if not ended:
        raise ValueError("Invalid PNG ending")
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    if not (1 <= width <= 4096 and 1 <= height <= 4096):
        raise ValueError("PNG dimensions exceed 4096×4096")
    if expected and (width != expected["width"] or height != expected["height"]):
        raise ValueError(f"Expected {expected['width']}×{expected['height']} PNG")

    try:
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or image.size != (width, height):
                raise ValueError("Invalid PNG dimensions or format")
            image.verify()
        with Image.open(io.BytesIO(png)) as image:
            image.load()
    except (OSError, SyntaxError) as exc:
        raise ValueError("Invalid PNG image data") from exc
    return {"width": width, "height": height}

