#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os
from pathlib import Path

from qwen38_constants import *
from qwen38_formats import build_delta_header
from qwen38_io_utils import align_up, create_exclusive, finalize, write_at
from qwen38_planes import put_q4, put_q8_delta_input, copy_vector_f32


def pack_delta(q4_dir: str | os.PathLike[str], q8_dir: str | os.PathLike[str],
               output: str | os.PathLike[str], layer: int) -> dict:
    if not (0 <= layer < LAYER_COUNT) or layer % 4 == 3:
        raise ValueError("layer must be a DeltaNet layer (0..63, layer%4!=3)")
    q4 = Path(q4_dir); q8 = Path(q8_dir); out_path = Path(output)
    gpr = HIDDEN_SIZE // Q4_GROUP_SIZE
    weight_bytes = MLP_SIZE * HIDDEN_SIZE // 2
    meta_plane = MLP_SIZE * gpr * 2
    di_groups = gpr
    dqkv = DELTA_QKV_ROWS * HIDDEN_SIZE // 2
    dz = DELTA_Z_ROWS * HIDDEN_SIZE // 2
    dsc = DELTA_SCALAR_ROWS * HIDDEN_SIZE // 2
    do_groups = DELTA_OUTPUT_INPUTS // Q4_GROUP_SIZE
    do_weight = HIDDEN_SIZE * DELTA_OUTPUT_INPUTS // 2

    v = dict(
        hidden_size=HIDDEN_SIZE, rows=MLP_SIZE, group_size=Q4_GROUP_SIZE,
        down_rows=HIDDEN_SIZE, down_groups_per_row=MLP_SIZE // Q4_GROUP_SIZE,
    )
    v["gate_quants_offset"] = IMAGE_HEADER_BYTES; v["gate_quants_bytes"] = weight_bytes
    v["gate_metadata_offset"] = v["gate_quants_offset"] + weight_bytes; v["gate_metadata_bytes"] = meta_plane * 2
    v["up_quants_offset"] = v["gate_metadata_offset"] + v["gate_metadata_bytes"]; v["up_quants_bytes"] = weight_bytes
    v["up_metadata_offset"] = v["up_quants_offset"] + weight_bytes; v["up_metadata_bytes"] = meta_plane * 2
    v["down_quants_offset"] = v["up_metadata_offset"] + v["up_metadata_bytes"]; v["down_quants_bytes"] = weight_bytes
    v["down_metadata_offset"] = v["down_quants_offset"] + weight_bytes; v["down_metadata_bytes"] = meta_plane * 2
    v.update(layer_index=layer, delta_input_rows=DELTA_INPUT_ROWS,
             delta_input_groups_per_row=di_groups, delta_output_rows=HIDDEN_SIZE,
             delta_output_groups_per_row=do_groups)
    v["input_norm_constants_index"] = 0
    v["post_norm_constants_index"] = HIDDEN_SIZE
    v["conv_constants_index"] = 2 * HIDDEN_SIZE
    v["a_log_constants_index"] = v["conv_constants_index"] + DELTA_CONV_VALUES
    v["dt_bias_constants_index"] = v["a_log_constants_index"] + DELTA_SCALAR_ROWS
    v["recurrent_norm_constants_index"] = v["dt_bias_constants_index"] + DELTA_SCALAR_ROWS
    v["constants_f32_count"] = v["recurrent_norm_constants_index"] + DELTA_HEAD_SIZE
    v["constants_offset"] = align_up(v["down_metadata_offset"] + v["down_metadata_bytes"])
    v["constants_bytes"] = align_up(v["constants_f32_count"] * 4)
    v["delta_input_quants_offset"] = v["constants_offset"] + v["constants_bytes"]
    v["delta_input_precision"] = 1
    v["delta_output_precision"] = 0
    v["delta_input_quants_bytes"] = 2 * (dqkv + dz + 2 * dsc)
    v["delta_input_metadata_offset"] = v["delta_input_quants_offset"] + v["delta_input_quants_bytes"]
    di_meta_payload = DELTA_INPUT_ROWS * di_groups * 4
    v["delta_input_metadata_bytes"] = align_up(di_meta_payload)
    v["delta_output_quants_offset"] = v["delta_input_metadata_offset"] + v["delta_input_metadata_bytes"]
    v["delta_output_quants_bytes"] = do_weight
    v["delta_output_metadata_offset"] = v["delta_output_quants_offset"] + do_weight
    v["delta_output_metadata_bytes"] = HIDDEN_SIZE * do_groups * 4
    final_size = v["delta_output_metadata_offset"] + v["delta_output_metadata_bytes"]

    try:
        with create_exclusive(out_path) as out:
            write_at(out, 0, build_delta_header(v))
            goff, gm = put_q4(q4, layer, "mlp.gate_proj", out, v["gate_quants_offset"], v["gate_metadata_offset"], MLP_SIZE, HIDDEN_SIZE)
            uoff, um = put_q4(q4, layer, "mlp.up_proj", out, v["up_quants_offset"], v["up_metadata_offset"], MLP_SIZE, HIDDEN_SIZE)
            doff, dm = put_q4(q4, layer, "mlp.down_proj", out, v["down_quants_offset"], v["down_metadata_offset"], HIDDEN_SIZE, MLP_SIZE)
            copy_vector_f32(q4, layer, "input_layernorm", out, v["constants_offset"] + v["input_norm_constants_index"] * 4, HIDDEN_SIZE)
            copy_vector_f32(q4, layer, "post_attention_layernorm", out, v["constants_offset"] + v["post_norm_constants_index"] * 4, HIDDEN_SIZE)
            copy_vector_f32(q4, layer, "linear_attn__conv1d", out, v["constants_offset"] + v["conv_constants_index"] * 4, DELTA_CONV_VALUES)
            copy_vector_f32(q4, layer, "linear_attn__A_log", out, v["constants_offset"] + v["a_log_constants_index"] * 4, DELTA_SCALAR_ROWS)
            copy_vector_f32(q4, layer, "linear_attn__dt_bias", out, v["constants_offset"] + v["dt_bias_constants_index"] * 4, DELTA_SCALAR_ROWS)
            copy_vector_f32(q4, layer, "linear_attn__norm", out, v["constants_offset"] + v["recurrent_norm_constants_index"] * 4, DELTA_HEAD_SIZE)
            qc, mc = put_q8_delta_input(q8, layer, out, v["delta_input_quants_offset"], v["delta_input_metadata_offset"], DELTA_INPUT_ROWS, HIDDEN_SIZE)
            ooff, om = put_q4(q4, layer, "linear_attn.out_proj", out, v["delta_output_quants_offset"], v["delta_output_metadata_offset"], HIDDEN_SIZE, DELTA_OUTPUT_INPUTS)
            checks = [
                (qc, v["delta_input_quants_offset"] + v["delta_input_quants_bytes"], "delta input codes"),
                (mc, v["delta_input_metadata_offset"] + di_meta_payload, "delta input metadata payload"),
                (goff, v["gate_quants_offset"] + v["gate_quants_bytes"], "gate codes"),
                (ooff, v["delta_output_quants_offset"] + v["delta_output_quants_bytes"], "delta output codes"),
                (om, v["delta_output_metadata_offset"] + v["delta_output_metadata_bytes"], "delta output metadata"),
            ]
            for got, expected, name in checks:
                if got != expected: raise RuntimeError(f"{name} cursor {got} != {expected}")
            finalize(out, final_size)
    except Exception:
        try: out_path.unlink()
        except FileNotFoundError: pass
        raise
    return {"output": str(out_path), "layer": layer, "bytes": final_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pack one folder-only hybrid Q4/Q8 DeltaNet layer")
    ap.add_argument("q4_dir"); ap.add_argument("q8_dir"); ap.add_argument("output"); ap.add_argument("layer", type=int)
    a = ap.parse_args(argv)
    try: result = pack_delta(a.q4_dir, a.q8_dir, a.output, a.layer)
    except Exception as exc:
        print(f"pack_delta: {exc}", file=__import__('sys').stderr); return 8
    print(json.dumps(result, separators=(",", ":"))); return 0

if __name__ == "__main__": raise SystemExit(main())
