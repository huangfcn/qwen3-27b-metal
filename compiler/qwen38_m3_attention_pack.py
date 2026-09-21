#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from qwen38_constants import *
from qwen38_formats import build_attention_header
from qwen38_io_utils import align_up, create_exclusive, finalize, write_at
from qwen38_planes import put_q4, copy_vector_f32


def pack_attention(q4_dir: str | os.PathLike[str], output: str | os.PathLike[str], layer: int) -> dict:
    if not (0 <= layer < LAYER_COUNT) or layer % 4 != 3:
        raise ValueError("layer must be an attention layer (0..63, layer%4==3)")
    q4 = Path(q4_dir); out_path = Path(output)
    hidden_groups = HIDDEN_SIZE // Q4_GROUP_SIZE
    mlp_weight = MLP_SIZE * HIDDEN_SIZE // 2
    mlp_meta = MLP_SIZE * hidden_groups * 2
    q_weight = ATTENTION_Q_ROWS * HIDDEN_SIZE // 2
    kv_weight = ATTENTION_K_ROWS * HIDDEN_SIZE // 2
    o_cols = ATTENTION_HEADS * ATTENTION_HEAD_SIZE
    o_weight = HIDDEN_SIZE * o_cols // 2
    v = dict(layer_index=layer, hidden_size=HIDDEN_SIZE, intermediate_size=MLP_SIZE,
             group_size=Q4_GROUP_SIZE, q_heads=ATTENTION_HEADS, kv_heads=ATTENTION_KV_HEADS,
             head_size=ATTENTION_HEAD_SIZE, rotary_size=ATTENTION_ROTARY_SIZE,
             input_rows=ATTENTION_INPUT_ROWS, input_groups_per_row=hidden_groups,
             output_rows=HIDDEN_SIZE, output_groups_per_row=o_cols // Q4_GROUP_SIZE)
    offset = ATTENTION_HEADER_BYTES
    def seg(name, n):
        nonlocal offset
        v[name + "_offset"] = offset; v[name + "_bytes"] = n; offset += n
    seg("gate_quants", mlp_weight); seg("gate_metadata", mlp_meta * 2)
    seg("up_quants", mlp_weight); seg("up_metadata", mlp_meta * 2)
    seg("down_quants", mlp_weight); seg("down_metadata", mlp_meta * 2)
    offset = align_up(offset)
    v["input_norm_constants_index"] = 0
    v["post_norm_constants_index"] = HIDDEN_SIZE
    v["q_norm_constants_index"] = 2 * HIDDEN_SIZE
    v["k_norm_constants_index"] = v["q_norm_constants_index"] + ATTENTION_HEAD_SIZE
    v["constants_f32_count"] = v["k_norm_constants_index"] + ATTENTION_HEAD_SIZE
    v["constants_offset"] = offset; v["constants_bytes"] = align_up(v["constants_f32_count"] * 4); offset += v["constants_bytes"]
    seg("attention_input_quants", q_weight + 2 * kv_weight)
    seg("attention_input_metadata", ATTENTION_INPUT_ROWS * hidden_groups * 4)
    seg("attention_output_quants", o_weight)
    seg("attention_output_metadata", HIDDEN_SIZE * v["output_groups_per_row"] * 4)
    final_size = offset
    try:
        with create_exclusive(out_path) as out:
            write_at(out, 0, build_attention_header(v))
            goff, gm = put_q4(q4, layer, "mlp.gate_proj", out, v["gate_quants_offset"], v["gate_metadata_offset"], MLP_SIZE, HIDDEN_SIZE)
            uoff, um = put_q4(q4, layer, "mlp.up_proj", out, v["up_quants_offset"], v["up_metadata_offset"], MLP_SIZE, HIDDEN_SIZE)
            doff, dm = put_q4(q4, layer, "mlp.down_proj", out, v["down_quants_offset"], v["down_metadata_offset"], HIDDEN_SIZE, MLP_SIZE)
            copy_vector_f32(q4, layer, "input_layernorm", out, v["constants_offset"] + v["input_norm_constants_index"]*4, HIDDEN_SIZE)
            copy_vector_f32(q4, layer, "post_attention_layernorm", out, v["constants_offset"] + v["post_norm_constants_index"]*4, HIDDEN_SIZE)
            copy_vector_f32(q4, layer, "self_attn__q_norm", out, v["constants_offset"] + v["q_norm_constants_index"]*4, ATTENTION_HEAD_SIZE)
            copy_vector_f32(q4, layer, "self_attn__k_norm", out, v["constants_offset"] + v["k_norm_constants_index"]*4, ATTENTION_HEAD_SIZE)
            iq, im = v["attention_input_quants_offset"], v["attention_input_metadata_offset"]
            iq, im = put_q4(q4, layer, "self_attn.q_proj", out, iq, im, ATTENTION_Q_ROWS, HIDDEN_SIZE)
            iq, im = put_q4(q4, layer, "self_attn.k_proj", out, iq, im, ATTENTION_K_ROWS, HIDDEN_SIZE)
            iq, im = put_q4(q4, layer, "self_attn.v_proj", out, iq, im, ATTENTION_V_ROWS, HIDDEN_SIZE)
            ooff, om = put_q4(q4, layer, "self_attn.o_proj", out, v["attention_output_quants_offset"], v["attention_output_metadata_offset"], HIDDEN_SIZE, o_cols)
            checks = [
                (iq, v["attention_input_quants_offset"]+v["attention_input_quants_bytes"], "attention input codes"),
                (im, v["attention_input_metadata_offset"]+v["attention_input_metadata_bytes"], "attention input metadata"),
                (goff, v["gate_quants_offset"]+v["gate_quants_bytes"], "gate codes"),
                (ooff, v["attention_output_quants_offset"]+v["attention_output_quants_bytes"], "attention output codes"),
            ]
            for got, expected, name in checks:
                if got != expected: raise RuntimeError(f"{name} cursor {got} != {expected}")
            finalize(out, final_size)
    except Exception:
        try: out_path.unlink()
        except FileNotFoundError: pass
        raise
    return {"output": str(out_path), "layer": layer, "bytes": final_size}


def main(argv=None):
    ap=argparse.ArgumentParser(description="Pack one folder-only Q4 full-attention layer")
    ap.add_argument("q4_dir"); ap.add_argument("output"); ap.add_argument("layer",type=int); a=ap.parse_args(argv)
    try: r=pack_attention(a.q4_dir,a.output,a.layer)
    except Exception as exc: print(f"pack_attention: {exc}",file=__import__('sys').stderr); return 6
    print(json.dumps(r,separators=(",",":"))); return 0
if __name__=="__main__": raise SystemExit(main())
