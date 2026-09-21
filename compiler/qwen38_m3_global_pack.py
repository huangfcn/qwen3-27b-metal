#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from qwen38_constants import *
from qwen38_formats import build_global_header
from qwen38_io_utils import create_exclusive, finalize, write_at, copy_path_at
from qwen38_planes import interleave_f16, plane_path, copy_vector_f32


def pack_global(q4_dir: str | os.PathLike[str], output: str | os.PathLike[str]) -> dict:
    q4=Path(q4_dir); out_path=Path(output)
    quant_bytes=VOCAB_SIZE*HIDDEN_SIZE//2
    vocab_groups=VOCAB_SIZE*(HIDDEN_SIZE//Q4_GROUP_SIZE)
    meta_bytes=vocab_groups*4
    v=dict(vocab_size=VOCAB_SIZE,hidden_size=HIDDEN_SIZE,group_size=Q4_GROUP_SIZE,constants_f32_count=HIDDEN_SIZE)
    offset=GLOBAL_HEADER_BYTES
    def seg(name,n):
        nonlocal offset
        v[name+"_offset"]=offset; v[name+"_bytes"]=n; offset+=n
    seg("embedding_quants",quant_bytes); seg("embedding_metadata",meta_bytes)
    seg("lm_head_quants",quant_bytes); seg("lm_head_metadata",meta_bytes)
    seg("constants",HIDDEN_SIZE*4)
    try:
        with create_exclusive(out_path) as out:
            write_at(out,0,build_global_header(v))
            copy_path_at(plane_path(q4,None,"embed_tokens","_codes.u8"),out,v["embedding_quants_offset"],quant_bytes)
            interleave_f16(plane_path(q4,None,"embed_tokens","_scale.f16"),plane_path(q4,None,"embed_tokens","_bias.f16"),out,v["embedding_metadata_offset"],vocab_groups)
            copy_path_at(plane_path(q4,None,"lm_head","_codes.u8"),out,v["lm_head_quants_offset"],quant_bytes)
            interleave_f16(plane_path(q4,None,"lm_head","_scale.f16"),plane_path(q4,None,"lm_head","_bias.f16"),out,v["lm_head_metadata_offset"],vocab_groups)
            copy_vector_f32(q4,None,"model__norm",out,v["constants_offset"],HIDDEN_SIZE)
            finalize(out,offset)
    except Exception:
        try: out_path.unlink()
        except FileNotFoundError: pass
        raise
    return {"output":str(out_path),"bytes":offset}


def main(argv=None):
    ap=argparse.ArgumentParser(description="Pack global Q4 embedding/lm_head/final norm image")
    ap.add_argument("q4_dir"); ap.add_argument("output"); a=ap.parse_args(argv)
    try:r=pack_global(a.q4_dir,a.output)
    except Exception as exc: print(f"pack_global: {exc}",file=__import__('sys').stderr); return 6
    print(json.dumps(r,separators=(",",":"))); return 0
if __name__=="__main__": raise SystemExit(main())
