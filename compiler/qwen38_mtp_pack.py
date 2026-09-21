#!/usr/bin/env python3
"""Pack the pinned Qwen3.8-27B MLX MTP checkpoint into runtime images.

The source is mlx-community/Qwen3.8-27B-MTP-4bit/model.safetensors.
It contains both:
  * the single MTP transformer attention layer (runtime layer_index 64), and
  * the MTP fc projection / normalization extras.

This module can therefore build both runtime files directly from the source
checkpoint:
  * mtp-layer.q38att
  * mtp.q38mtp

The MLX checkpoint already stores affine Q4 group-64 linears as packed U32
weights plus BF16 scales/biases.  The runtime consumes the packed nibbles
unchanged; only scale/bias metadata is converted to FP16 and interleaved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from qwen38_constants import (
    ATTENTION_HEADS,
    ATTENTION_HEAD_SIZE,
    ATTENTION_HEADER_BYTES,
    ATTENTION_INPUT_ROWS,
    ATTENTION_K_ROWS,
    ATTENTION_KV_HEADS,
    ATTENTION_Q_ROWS,
    ATTENTION_ROTARY_SIZE,
    ATTENTION_V_ROWS,
    EXPECTED_MTP_SHA256,
    HIDDEN_SIZE,
    MLP_SIZE,
    MTP_HEADER_BYTES,
    Q4_GROUP_SIZE,
)
from qwen38_formats import build_attention_header, build_mtp_header
from qwen38_io_utils import align_up, create_exclusive, finalize, write_at
from qwen38_sha256 import sha256_file
from qwen38_safetensors import SafeTensorFile

MTP_LAYER_INDEX = 64
MTP_PREFIX = "layers.0."


def _bf16_to_f32_bytes(raw: bytes | memoryview) -> bytes:
    u = np.frombuffer(raw, dtype="<u2")
    return (u.astype(np.uint32) << 16).view("<f4").tobytes()


def _bf16_to_f16_bits(raw: bytes | memoryview) -> np.ndarray:
    u = np.frombuffer(raw, dtype="<u2")
    f = (u.astype(np.uint32) << 16).view("<f4")
    return f.astype("<f2").view("<u2")


def _shape(view, expected, name):
    if tuple(view.shape) != tuple(expected):
        raise ValueError(f"{name}: shape {view.shape} != {expected}")


def _verify_source(src: Path, source_sha256: str, verify_sha: bool) -> None:
    if source_sha256 != EXPECTED_MTP_SHA256:
        raise ValueError("source is not the pinned Qwen3.8 MTP file")
    if verify_sha:
        actual = sha256_file(src)
        if actual != source_sha256:
            raise ValueError(
                f"source SHA-256 mismatch: expected {source_sha256}, got {actual}"
            )


def _validate_mlx_q4(sf: SafeTensorFile, base: str, rows: int, cols: int) -> None:
    """Validate one MLX affine-Q4 tensor triplet: weight/scales/biases."""
    if cols % Q4_GROUP_SIZE:
        raise ValueError(f"{base}: cols {cols} not divisible by {Q4_GROUP_SIZE}")
    w = sf.find(base + ".weight")
    s = sf.find(base + ".scales")
    b = sf.find(base + ".biases")
    if w.dtype != "U32":
        raise ValueError(f"{base}.weight: expected U32, got {w.dtype}")
    _shape(w, (rows, cols // 8), base + ".weight")
    expected_weight_bytes = rows * cols // 2
    if w.data_length != expected_weight_bytes:
        raise ValueError(
            f"{base}.weight: byte length {w.data_length} != {expected_weight_bytes}"
        )
    groups = cols // Q4_GROUP_SIZE
    for name, view in ((base + ".scales", s), (base + ".biases", b)):
        if view.dtype != "BF16":
            raise ValueError(f"{name}: expected BF16, got {view.dtype}")
        _shape(view, (rows, groups), name)
        if view.data_length != rows * groups * 2:
            raise ValueError(f"{name}: unexpected byte length {view.data_length}")


def _write_mlx_q4(
    sf: SafeTensorFile,
    base: str,
    out,
    quants_offset: int,
    metadata_offset: int,
    rows: int,
    cols: int,
) -> tuple[int, int]:
    """Copy MLX packed Q4 codes and write FP16 interleaved (scale,bias)."""
    _validate_mlx_q4(sf, base, rows, cols)
    weight = sf.raw_view(base + ".weight")
    try:
        write_at(out, quants_offset, weight)
    finally:
        weight.release()

    scales_raw = sf.read_raw(base + ".scales")
    biases_raw = sf.read_raw(base + ".biases")
    scales = _bf16_to_f16_bits(scales_raw)
    biases = _bf16_to_f16_bits(biases_raw)
    inter = np.empty(scales.size * 2, dtype="<u2")
    inter[0::2] = scales
    inter[1::2] = biases
    write_at(out, metadata_offset, inter.tobytes())

    code_bytes = rows * cols // 2
    metadata_bytes = rows * (cols // Q4_GROUP_SIZE) * 4
    return quants_offset + code_bytes, metadata_offset + metadata_bytes


def _write_bf16_vector_f32(
    sf: SafeTensorFile,
    name: str,
    out,
    offset: int,
    values: int,
) -> None:
    view = sf.find(name)
    if view.dtype != "BF16":
        raise ValueError(f"{name}: expected BF16, got {view.dtype}")
    _shape(view, (values,), name)
    write_at(out, offset, _bf16_to_f32_bytes(sf.read_raw(name)))


def _validate_inventory(sf: SafeTensorFile) -> None:
    expected = {
        "fc.weight", "fc.scales", "fc.biases",
        "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "norm.weight",
        MTP_PREFIX + "input_layernorm.weight",
        MTP_PREFIX + "post_attention_layernorm.weight",
        MTP_PREFIX + "self_attn.q_norm.weight",
        MTP_PREFIX + "self_attn.k_norm.weight",
    }
    for base in (
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj",
    ):
        p = MTP_PREFIX + base
        expected.update({p + ".weight", p + ".scales", p + ".biases"})
    got = set(sf.keys())
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(
            "unexpected MTP tensor inventory; "
            f"missing={missing[:8]} extra={extra[:8]}"
        )


def pack_mtp_layer(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    source_sha256: str = EXPECTED_MTP_SHA256,
    verify_sha: bool = True,
) -> dict:
    """Build mtp-layer.q38att directly from the MLX MTP safetensors file."""
    src = Path(source)
    out_path = Path(output)
    _verify_source(src, source_sha256, verify_sha)

    hidden_groups = HIDDEN_SIZE // Q4_GROUP_SIZE
    mlp_weight = MLP_SIZE * HIDDEN_SIZE // 2
    mlp_meta = MLP_SIZE * hidden_groups * 2
    q_weight = ATTENTION_Q_ROWS * HIDDEN_SIZE // 2
    kv_weight = ATTENTION_K_ROWS * HIDDEN_SIZE // 2
    o_cols = ATTENTION_HEADS * ATTENTION_HEAD_SIZE
    o_weight = HIDDEN_SIZE * o_cols // 2

    v = dict(
        layer_index=MTP_LAYER_INDEX,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=MLP_SIZE,
        group_size=Q4_GROUP_SIZE,
        q_heads=ATTENTION_HEADS,
        kv_heads=ATTENTION_KV_HEADS,
        head_size=ATTENTION_HEAD_SIZE,
        rotary_size=ATTENTION_ROTARY_SIZE,
        input_rows=ATTENTION_INPUT_ROWS,
        input_groups_per_row=hidden_groups,
        output_rows=HIDDEN_SIZE,
        output_groups_per_row=o_cols // Q4_GROUP_SIZE,
        source_sha256=source_sha256.encode("ascii"),
    )
    offset = ATTENTION_HEADER_BYTES

    def seg(name: str, n: int) -> None:
        nonlocal offset
        v[name + "_offset"] = offset
        v[name + "_bytes"] = n
        offset += n

    seg("gate_quants", mlp_weight)
    seg("gate_metadata", mlp_meta * 2)
    seg("up_quants", mlp_weight)
    seg("up_metadata", mlp_meta * 2)
    seg("down_quants", mlp_weight)
    seg("down_metadata", mlp_meta * 2)
    offset = align_up(offset)

    v["input_norm_constants_index"] = 0
    v["post_norm_constants_index"] = HIDDEN_SIZE
    v["q_norm_constants_index"] = 2 * HIDDEN_SIZE
    v["k_norm_constants_index"] = v["q_norm_constants_index"] + ATTENTION_HEAD_SIZE
    v["constants_f32_count"] = v["k_norm_constants_index"] + ATTENTION_HEAD_SIZE
    v["constants_offset"] = offset
    v["constants_bytes"] = align_up(v["constants_f32_count"] * 4)
    offset += v["constants_bytes"]

    seg("attention_input_quants", q_weight + 2 * kv_weight)
    seg("attention_input_metadata", ATTENTION_INPUT_ROWS * hidden_groups * 4)
    seg("attention_output_quants", o_weight)
    seg("attention_output_metadata", HIDDEN_SIZE * v["output_groups_per_row"] * 4)
    final_size = offset

    try:
        with SafeTensorFile(src) as sf:
            _validate_inventory(sf)
            with create_exclusive(out_path) as out:
                write_at(out, 0, build_attention_header(v))

                goff, gm = _write_mlx_q4(
                    sf, MTP_PREFIX + "mlp.gate_proj", out,
                    v["gate_quants_offset"], v["gate_metadata_offset"],
                    MLP_SIZE, HIDDEN_SIZE,
                )
                uoff, um = _write_mlx_q4(
                    sf, MTP_PREFIX + "mlp.up_proj", out,
                    v["up_quants_offset"], v["up_metadata_offset"],
                    MLP_SIZE, HIDDEN_SIZE,
                )
                doff, dm = _write_mlx_q4(
                    sf, MTP_PREFIX + "mlp.down_proj", out,
                    v["down_quants_offset"], v["down_metadata_offset"],
                    HIDDEN_SIZE, MLP_SIZE,
                )

                _write_bf16_vector_f32(
                    sf, MTP_PREFIX + "input_layernorm.weight", out,
                    v["constants_offset"] + v["input_norm_constants_index"] * 4,
                    HIDDEN_SIZE,
                )
                _write_bf16_vector_f32(
                    sf, MTP_PREFIX + "post_attention_layernorm.weight", out,
                    v["constants_offset"] + v["post_norm_constants_index"] * 4,
                    HIDDEN_SIZE,
                )
                _write_bf16_vector_f32(
                    sf, MTP_PREFIX + "self_attn.q_norm.weight", out,
                    v["constants_offset"] + v["q_norm_constants_index"] * 4,
                    ATTENTION_HEAD_SIZE,
                )
                _write_bf16_vector_f32(
                    sf, MTP_PREFIX + "self_attn.k_norm.weight", out,
                    v["constants_offset"] + v["k_norm_constants_index"] * 4,
                    ATTENTION_HEAD_SIZE,
                )

                iq, im = v["attention_input_quants_offset"], v["attention_input_metadata_offset"]
                iq, im = _write_mlx_q4(
                    sf, MTP_PREFIX + "self_attn.q_proj", out, iq, im,
                    ATTENTION_Q_ROWS, HIDDEN_SIZE,
                )
                iq, im = _write_mlx_q4(
                    sf, MTP_PREFIX + "self_attn.k_proj", out, iq, im,
                    ATTENTION_K_ROWS, HIDDEN_SIZE,
                )
                iq, im = _write_mlx_q4(
                    sf, MTP_PREFIX + "self_attn.v_proj", out, iq, im,
                    ATTENTION_V_ROWS, HIDDEN_SIZE,
                )
                ooff, om = _write_mlx_q4(
                    sf, MTP_PREFIX + "self_attn.o_proj", out,
                    v["attention_output_quants_offset"],
                    v["attention_output_metadata_offset"],
                    HIDDEN_SIZE, o_cols,
                )

                checks = [
                    (goff, v["gate_quants_offset"] + v["gate_quants_bytes"], "gate codes"),
                    (gm, v["gate_metadata_offset"] + v["gate_metadata_bytes"], "gate metadata"),
                    (uoff, v["up_quants_offset"] + v["up_quants_bytes"], "up codes"),
                    (um, v["up_metadata_offset"] + v["up_metadata_bytes"], "up metadata"),
                    (doff, v["down_quants_offset"] + v["down_quants_bytes"], "down codes"),
                    (dm, v["down_metadata_offset"] + v["down_metadata_bytes"], "down metadata"),
                    (iq, v["attention_input_quants_offset"] + v["attention_input_quants_bytes"], "attention input codes"),
                    (im, v["attention_input_metadata_offset"] + v["attention_input_metadata_bytes"], "attention input metadata"),
                    (ooff, v["attention_output_quants_offset"] + v["attention_output_quants_bytes"], "attention output codes"),
                    (om, v["attention_output_metadata_offset"] + v["attention_output_metadata_bytes"], "attention output metadata"),
                ]
                for got, expected, name in checks:
                    if got != expected:
                        raise RuntimeError(f"{name} cursor {got} != {expected}")
                finalize(out, final_size)
    except Exception:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return {
        "source": str(src),
        "output": str(out_path),
        "source_sha256": source_sha256,
        "layer": MTP_LAYER_INDEX,
        "bytes": final_size,
    }


def pack_mtp(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    source_sha256: str = EXPECTED_MTP_SHA256,
    verify_sha: bool = True,
) -> dict:
    """Build mtp.q38mtp (fc projection + three norms)."""
    src = Path(source)
    out_path = Path(output)
    _verify_source(src, source_sha256, verify_sha)

    with SafeTensorFile(src) as sf:
        _validate_inventory(sf)
        names = [
            "fc.weight", "fc.scales", "fc.biases",
            "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "norm.weight",
        ]
        vw = {n: sf.find(n) for n in names}
        fc_rows = 5120
        fc_groups = 160
        fc_quant_bytes = fc_rows * fc_groups * 64 // 2
        fc_meta_bytes = fc_rows * fc_groups * 4
        if vw["fc.weight"].dtype != "U32":
            raise ValueError("fc.weight: expected U32")
        _shape(vw["fc.weight"], (fc_rows, 1280), "fc.weight")
        if vw["fc.weight"].data_length != fc_quant_bytes:
            raise ValueError("fc.weight: unexpected byte length")
        for n in ("fc.scales", "fc.biases"):
            if vw[n].dtype != "BF16":
                raise ValueError(f"{n}: expected BF16")
            _shape(vw[n], (fc_rows, fc_groups), n)
        for n in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "norm.weight"):
            if vw[n].dtype != "BF16":
                raise ValueError(f"{n}: expected BF16")
            _shape(vw[n], (5120,), n)
        fc_w = sf.read_raw("fc.weight")
        scales = sf.read_raw("fc.scales")
        biases = sf.read_raw("fc.biases")
        norms = [
            sf.read_raw(n)
            for n in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "norm.weight")
        ]

    v = dict(
        hidden_size=5120,
        fc_rows=fc_rows,
        fc_groups_per_row=fc_groups,
        group_size=64,
        constants_f32_count=3 * 5120,
    )
    v["fc_quants_offset"] = MTP_HEADER_BYTES
    v["fc_quants_bytes"] = fc_quant_bytes
    v["fc_metadata_offset"] = v["fc_quants_offset"] + fc_quant_bytes
    v["fc_metadata_bytes"] = fc_meta_bytes
    v["constants_offset"] = v["fc_metadata_offset"] + fc_meta_bytes
    v["constants_bytes"] = v["constants_f32_count"] * 4
    v["embedding_norm_constants_index"] = 0
    v["hidden_norm_constants_index"] = 5120
    v["final_norm_constants_index"] = 10240
    v["source_sha256"] = source_sha256.encode("ascii")
    total = v["constants_offset"] + v["constants_bytes"]

    try:
        with create_exclusive(out_path) as out:
            write_at(out, 0, build_mtp_header(v))
            write_at(out, v["fc_quants_offset"], fc_w)
            s = _bf16_to_f16_bits(scales)
            b = _bf16_to_f16_bits(biases)
            inter = np.empty(s.size * 2, dtype="<u2")
            inter[0::2] = s
            inter[1::2] = b
            write_at(out, v["fc_metadata_offset"], inter.tobytes())
            for idx, raw in zip((0, 5120, 10240), norms):
                write_at(out, v["constants_offset"] + idx * 4, _bf16_to_f32_bytes(raw))
            finalize(out, total)
    except Exception:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return {
        "source": str(src),
        "output": str(out_path),
        "source_sha256": source_sha256,
        "bytes": total,
    }


def pack_mtp_bundle(
    source: str | os.PathLike[str],
    layer_output: str | os.PathLike[str] | None,
    extras_output: str | os.PathLike[str] | None,
    source_sha256: str = EXPECTED_MTP_SHA256,
    verify_sha: bool = True,
) -> dict:
    """Build either or both MTP runtime files, hashing the source only once."""
    src = Path(source)
    _verify_source(src, source_sha256, verify_sha)
    result = {"source": str(src), "source_sha256": source_sha256}
    if layer_output is not None:
        result["layer"] = pack_mtp_layer(
            src, layer_output, source_sha256=source_sha256, verify_sha=False
        )
    if extras_output is not None:
        result["extras"] = pack_mtp(
            src, extras_output, source_sha256=source_sha256, verify_sha=False
        )
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pack Qwen3.8 MTP runtime image(s)")
    ap.add_argument("source", help="pinned mlx-community Qwen3.8 MTP model.safetensors")
    ap.add_argument("output", help="output mtp.q38mtp")
    ap.add_argument("source_sha256", nargs="?", default=EXPECTED_MTP_SHA256)
    ap.add_argument(
        "--layer-output",
        help="also build mtp-layer.q38att directly from the same safetensors file",
    )
    ap.add_argument(
        "--no-sha-check",
        action="store_true",
        help="debug only; headers still carry the pinned SHA",
    )
    a = ap.parse_args(argv)
    try:
        if a.layer_output:
            r = pack_mtp_bundle(
                a.source,
                a.layer_output,
                a.output,
                a.source_sha256,
                not a.no_sha_check,
            )
        else:
            r = pack_mtp(a.source, a.output, a.source_sha256, not a.no_sha_check)
    except Exception as exc:
        print(f"pack_mtp: {exc}", file=__import__("sys").stderr)
        return 6
    print(json.dumps(r, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
