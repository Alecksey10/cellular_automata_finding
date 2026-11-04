"""Минимальный PNG-энкодер без внешних зависимостей."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def png_bytes(rgb: np.ndarray) -> bytes:
    """Закодировать ``(height, width, 3)`` uint8 массив в PNG (в памяти)."""

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("ожидался RGB-массив")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def write_png(path: str | Path, rgb: np.ndarray) -> Path:
    """Записать ``(height, width, 3)`` uint8 массив в PNG-файл."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(rgb))
    return path


__all__ = ["png_bytes", "write_png"]
