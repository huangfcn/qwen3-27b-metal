"""Replacement for qwen38_sha256.c/.h."""
from __future__ import annotations
import hashlib
from pathlib import Path

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path, chunk_size=8<<20) -> str:
    h=hashlib.sha256()
    with open(path,"rb",buffering=0) as f:
        while True:
            b=f.read(chunk_size)
            if not b: break
            h.update(b)
    return h.hexdigest()
