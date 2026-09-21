from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import BinaryIO

CHUNK = 8 << 20


def align_up(value: int, alignment: int = 4096) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def require_sized(path: Path, expected: int) -> Path:
    try:
        actual = path.stat().st_size
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing input plane: {path}") from exc
    if actual != expected:
        raise ValueError(f"{path}: size {actual} != expected {expected}")
    return path


def copy_path_at(path: Path, out: BinaryIO, out_offset: int, expected: int | None = None) -> int:
    if expected is not None:
        require_sized(path, expected)
    written = 0
    with path.open("rb", buffering=0) as src:
        out.seek(out_offset)
        while True:
            block = src.read(CHUNK)
            if not block:
                break
            out.write(block)
            written += len(block)
    if expected is not None and written != expected:
        raise IOError(f"{path}: copied {written} != expected {expected}")
    return written


def write_at(out: BinaryIO, offset: int, data: bytes | bytearray | memoryview) -> None:
    out.seek(offset)
    out.write(data)


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def create_exclusive(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    return os.fdopen(fd, "w+b", buffering=0)


def finalize(out: BinaryIO, size: int) -> None:
    out.truncate(size)
    out.flush()
    os.fsync(out.fileno())
