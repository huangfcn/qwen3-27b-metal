#!/usr/bin/env python3
"""Minimal dependency-free Safetensors metadata/payload reader.

Equivalent to qwen38_safetensors.c/.h plus qwen38_safetensors_inspect.c.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_HEADER_BYTES = 256 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1
MAX_RANK = 8
MAX_DTYPE_BYTES = 15


class SafetensorsError(Exception):
    def __init__(self, message: str, status: int = 1):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class TensorView:
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_length: int

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def data_end(self) -> int:
        return self.data_start + self.data_length

    def as_dict(self, name: str | None = None) -> dict:
        d = {"dtype": self.dtype, "shape": list(self.shape),
             "data_start": self.data_start, "data_length": self.data_length}
        return {"name": name, **d} if name is not None else d


def _u64(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= UINT64_MAX):
        raise ValueError(what)
    return value


def _load_header(path: Path) -> tuple[dict, int, int]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SafetensorsError(f"open {path}: {exc}", 2) from exc
    if size < 8:
        raise SafetensorsError(f"{path}: missing safetensors header", 3)
    with path.open("rb", buffering=0) as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SafetensorsError(f"{path}: cannot read header length", 4)
        (n,) = struct.unpack("<Q", raw)
        if n == 0 or n > MAX_HEADER_BYTES or n > size - 8:
            raise SafetensorsError(f"{path}: invalid header length {n}", 5)
        b = f.read(n)
        if len(b) != n:
            raise SafetensorsError(f"{path}: cannot allocate/read header", 6)
    try:
        header = json.loads(b)
    except Exception as exc:
        raise SafetensorsError(f"{path}: malformed safetensors JSON header", 7) from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"{path}: malformed safetensors JSON header", 7)
    return header, n, size


def _parse_view(path: Path, name: str, d: object, header_length: int,
                file_size: int, require_payload: bool) -> TensorView:
    if not isinstance(d, dict):
        raise SafetensorsError(f"{path}: tensor {name} has malformed descriptor", 7)
    try:
        dtype = d["dtype"]; shape0 = d["shape"]; offsets = d["data_offsets"]
    except KeyError as exc:
        raise SafetensorsError(f"{path}: incomplete descriptor for {name}", 8) from exc
    if not isinstance(dtype, str) or not dtype or len(dtype.encode()) > MAX_DTYPE_BYTES:
        raise SafetensorsError(f"{path}: invalid dtype for {name}", 9)
    if not isinstance(shape0, list) or len(shape0) > MAX_RANK:
        raise SafetensorsError(f"{path}: invalid shape for {name}", 10)
    try:
        shape = tuple(_u64(x, "shape") for x in shape0)
    except ValueError as exc:
        raise SafetensorsError(f"{path}: invalid shape for {name}", 10) from exc
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise SafetensorsError(f"{path}: invalid data range for {name}", 11)
    try:
        lo, hi = (_u64(offsets[0], "lo"), _u64(offsets[1], "hi"))
    except ValueError as exc:
        raise SafetensorsError(f"{path}: invalid data range for {name}", 11) from exc
    if hi < lo:
        raise SafetensorsError(f"{path}: invalid data range for {name}", 11)
    payload = 8 + header_length
    if lo > UINT64_MAX - payload or hi > UINT64_MAX - payload:
        raise SafetensorsError(f"{path}: data range overflow for {name}", 12)
    start = payload + lo; length = hi - lo
    if require_payload and (start > file_size or length > file_size - start):
        raise SafetensorsError(
            f"{path}: payload for {name} is truncated (need end {start + length}, file is {file_size})", 13)
    return TensorView(dtype, shape, start, length)


def qwen38_safetensors_find(path: str | os.PathLike[str], tensor_name: str,
                            require_payload: bool = True) -> TensorView:
    p = Path(path)
    h, n, size = _load_header(p)
    if tensor_name not in h:
        raise SafetensorsError(f"{p}: tensor {tensor_name} not found", 7)
    return _parse_view(p, tensor_name, h[tensor_name], n, size, require_payload)


class SafeTensorFile:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.header, self.header_length, self.file_size = _load_header(self.path)
        self.payload_start = 8 + self.header_length
        self._fh = None
        self._mm = None

    def __enter__(self):
        self._map(); return self

    def __exit__(self, *_):
        self.close()

    def _map(self):
        if self._mm is None:
            self._fh = self.path.open("rb")
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm

    def close(self):
        if self._mm is not None:
            self._mm.close(); self._mm = None
        if self._fh is not None:
            self._fh.close(); self._fh = None

    def keys(self) -> list[str]:
        return [k for k in self.header if k != "__metadata__"]

    def find(self, name: str, require_payload: bool = True) -> TensorView:
        if name not in self.header:
            raise SafetensorsError(f"{self.path}: tensor {name} not found", 7)
        return _parse_view(self.path, name, self.header[name], self.header_length,
                           self.file_size, require_payload)

    tensor_info = find

    def raw_view(self, name: str) -> memoryview:
        v = self.find(name, True)
        return memoryview(self._map())[v.data_start:v.data_end]

    def read_raw(self, name: str) -> bytes:
        return bytes(self.raw_view(name))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--header-only", action="store_true")
    ap.add_argument("file")
    ap.add_argument("tensor", nargs="+")
    a = ap.parse_args(argv)
    result = {"file": a.file, "payload_required": not a.header_only, "tensors": []}
    try:
        sf = SafeTensorFile(a.file)
        for name in a.tensor:
            result["tensors"].append(sf.find(name, not a.header_only).as_dict(name))
    except SafetensorsError as exc:
        print(exc, file=sys.stderr); return exc.status
    print(json.dumps(result, indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
