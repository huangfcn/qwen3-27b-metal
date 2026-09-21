#!/usr/bin/env python3
"""Python equivalent of qwen38_tokenizer_pack.c for Qwen3.8-27B."""
from __future__ import annotations

import argparse, hashlib, json, os, struct, sys
from pathlib import Path

from qwen38_constants import (
    TOKENIZER_SOURCE_SHA256, TOKENIZER_VERSION, TOKENIZER_HEADER_BYTES,
    TOKENIZER_VOCAB, TOKENIZER_BASE_VOCAB, TOKENIZER_ADDED_FIRST,
    TOKENIZER_ADDED_COUNT, TOKENIZER_MERGES,
)
from qwen38_formats import build_tokenizer_header
from qwen38_io_utils import align_up, create_exclusive, finalize, write_at

DIR_ENTRY = struct.Struct("<QII")
MERGE_ENTRY = struct.Struct("<IIII")


def _token_bytes(s: object, what: str) -> bytes:
    if not isinstance(s, str): raise ValueError(f"{what}: expected JSON string")
    return s.encode("utf-8")


def _build_byte_unicode() -> list[int]:
    used=[False]*256
    for a,b in ((33,126),(161,172),(174,255)):
        for x in range(a,b+1): used[x]=True
    nxt=256; m=[]
    for x in range(256):
        if used[x]: m.append(x)
        else: m.append(nxt); nxt+=1
    return m


def _bytelevel_decode(raw: bytes, reverse: dict[int,int]) -> bytes:
    try: text=raw.decode("utf-8")
    except UnicodeDecodeError as exc: raise ValueError("invalid UTF-8 byte-level token") from exc
    out=bytearray()
    for ch in text:
        cp=ord(ch)
        if cp not in reverse: raise ValueError(f"byte-level codepoint U+{cp:04X} is not mapped")
        out.append(reverse[cp])
    return bytes(out)


def _load_tokens(doc: dict) -> tuple[list[bytes],dict[bytes,int]]:
    try: vocab=doc["model"]["vocab"]
    except Exception as exc: raise ValueError("tokenizer missing model.vocab") from exc
    if not isinstance(vocab,dict): raise ValueError("model.vocab is not an object")
    tokens: list[bytes | None] = [None] * TOKENIZER_VOCAB
    count=0
    for text,id0 in vocab.items():
        if isinstance(id0,bool) or not isinstance(id0,int) or not (0<=id0<TOKENIZER_BASE_VOCAB):
            raise ValueError(f"invalid base vocab id {id0!r}")
        if tokens[id0] is not None: raise ValueError(f"duplicate base vocab id {id0}")
        tokens[id0]=_token_bytes(text,"vocab token"); count+=1
    if count!=TOKENIZER_BASE_VOCAB or any(tokens[i] is None for i in range(TOKENIZER_BASE_VOCAB)):
        raise ValueError(f"base vocab count {count} != {TOKENIZER_BASE_VOCAB}")
    added=doc.get("added_tokens")
    if not isinstance(added,list) or len(added)<TOKENIZER_ADDED_COUNT:
        raise ValueError("tokenizer missing pinned added_tokens")
    for obj in added[:TOKENIZER_ADDED_COUNT]:
        if not isinstance(obj,dict) or "id" not in obj or "content" not in obj: raise ValueError("malformed added token")
        tid=obj["id"]
        if isinstance(tid,bool) or not isinstance(tid,int) or not (TOKENIZER_ADDED_FIRST<=tid<TOKENIZER_VOCAB):
            raise ValueError(f"invalid added token id {tid!r}")
        if tokens[tid] is not None: raise ValueError(f"duplicate token id {tid}")
        tokens[tid]=_token_bytes(obj["content"],"added token")
    if any(x is None for x in tokens):
        missing=next(i for i,x in enumerate(tokens) if x is None)
        raise ValueError(f"missing tokenizer token {missing}")
    concrete=[x for x in tokens if x is not None]
    lookup={}
    for i,t in enumerate(concrete):
        if t in lookup: raise ValueError(f"duplicate tokenizer token bytes at ids {lookup[t]} and {i}")
        lookup[t]=i
    return concrete,lookup


def _parse_merges(doc:dict, lookup:dict[bytes,int]) -> list[tuple[int,int,int,int]]:
    try: source=doc["model"]["merges"]
    except Exception as exc: raise ValueError("tokenizer missing model.merges") from exc
    if not isinstance(source,list) or len(source)<TOKENIZER_MERGES:
        raise ValueError(f"merge count {len(source) if isinstance(source,list) else 'invalid'} < {TOKENIZER_MERGES}")
    out=[]
    for rank,item in enumerate(source[:TOKENIZER_MERGES]):
        if isinstance(item,list):
            if len(item)!=2: raise ValueError(f"merge {rank}: expected pair")
            left=_token_bytes(item[0],f"merge {rank} left"); right=_token_bytes(item[1],f"merge {rank} right")
        elif isinstance(item,str):
            pair=item.encode("utf-8"); sep=pair.find(b" ")
            if sep<=0 or sep==len(pair)-1: raise ValueError(f"merge {rank}: malformed string pair")
            left,right=pair[:sep],pair[sep+1:]
        else: raise ValueError(f"merge {rank}: invalid entry")
        try: li=lookup[left]; ri=lookup[right]; result=lookup[left+right]
        except KeyError as exc: raise ValueError(f"merge {rank}: token not found in vocabulary") from exc
        out.append((li,ri,result,rank))
    out.sort(key=lambda e:(e[0],e[1]))
    for a,b in zip(out,out[1:]):
        if a[0]==b[0] and a[1]==b[1]: raise ValueError(f"duplicate merge pair {a[0]},{a[1]}")
    return out


def pack_tokenizer(source: str|os.PathLike[str], output: str|os.PathLike[str], *,
                   allow_unpinned: bool=False) -> dict:
    source=Path(source); out_path=Path(output)
    raw=source.read_bytes(); actual=hashlib.sha256(raw).hexdigest()
    if not allow_unpinned and actual!=TOKENIZER_SOURCE_SHA256:
        raise ValueError(f"tokenizer SHA-256 mismatch: {actual}")
    try: doc=json.loads(raw)
    except Exception as exc: raise ValueError("cannot parse tokenizer JSON") from exc
    tokens,lookup=_load_tokens(doc)
    merges=_parse_merges(doc,lookup)
    byte_unicode=_build_byte_unicode(); reverse={cp:b for b,cp in enumerate(byte_unicode)}
    byte_token_ids=[]
    for b,cp in enumerate(byte_unicode):
        encoded=chr(cp).encode("utf-8")
        if encoded not in lookup: raise ValueError(f"missing byte-level base token {b}")
        byte_token_ids.append(lookup[encoded])
    decoded=[]; directory=[]; blob_bytes=0
    for tid,t in enumerate(tokens):
        d=_bytelevel_decode(t,reverse) if tid<TOKENIZER_BASE_VOCAB else bytes(t)
        decoded.append(d); flags=1 if tid>=TOKENIZER_ADDED_FIRST else 0
        directory.append((blob_bytes,len(d),flags)); blob_bytes+=len(d)
    directory_offset=TOKENIZER_HEADER_BYTES
    directory_bytes=TOKENIZER_VOCAB*DIR_ENTRY.size
    blob_offset=directory_offset+directory_bytes
    merges_offset=align_up(blob_offset+blob_bytes)
    merges_bytes=TOKENIZER_MERGES*MERGE_ENTRY.size
    file_bytes=merges_offset+merges_bytes
    values=dict(vocab_size=TOKENIZER_VOCAB,base_vocab_size=TOKENIZER_BASE_VOCAB,
                added_first=TOKENIZER_ADDED_FIRST,added_count=TOKENIZER_ADDED_COUNT,
                merge_count=TOKENIZER_MERGES,token_entry_bytes=DIR_ENTRY.size,
                token_directory_offset=directory_offset,token_directory_bytes=directory_bytes,
                token_blob_offset=blob_offset,token_blob_bytes=blob_bytes,
                merges_offset=merges_offset,merges_bytes=merges_bytes,
                source_sha256=actual.encode("ascii"),byte_token_ids=byte_token_ids)
    try:
        with create_exclusive(out_path) as out:
            write_at(out,0,build_tokenizer_header(values))
            out.seek(directory_offset)
            for e in directory: out.write(DIR_ENTRY.pack(*e))
            out.seek(blob_offset)
            for d in decoded: out.write(d)
            out.seek(merges_offset)
            for e in merges: out.write(MERGE_ENTRY.pack(*e))
            finalize(out,file_bytes)
    except Exception:
        try: out_path.unlink()
        except FileNotFoundError: pass
        raise
    return {"source":str(source),"source_sha256":actual,"output":str(out_path),
            "bytes":file_bytes,"vocab":TOKENIZER_VOCAB,"merges":TOKENIZER_MERGES}


def main(argv=None):
    ap=argparse.ArgumentParser(description="Pack Qwen3.8 tokenizer.json into tokenizer.q38tok")
    ap.add_argument("source"); ap.add_argument("output"); ap.add_argument("--allow-unpinned",action="store_true")
    a=ap.parse_args(argv)
    try:r=pack_tokenizer(a.source,a.output,allow_unpinned=a.allow_unpinned)
    except Exception as exc: print(f"pack_tokenizer: {exc}",file=sys.stderr); return 6
    print(json.dumps(r,separators=(",",":"))); return 0
if __name__=="__main__": raise SystemExit(main())
