"""Минимальный PNG-писатель (RGB, 8 бит) на stdlib + numpy."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def encode_png(pixels: np.ndarray) -> bytes:
    """Собрать PNG-байты из массива ``(height, width, 3)`` uint8."""

    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Ожидается (h, w, 3), получено {array.shape}")
    height, width = array.shape[:2]
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


def write_png(path: str | Path, pixels: np.ndarray) -> Path:
    """Записать массив ``(height, width, 3)`` uint8 в PNG."""

    payload = encode_png(pixels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path
