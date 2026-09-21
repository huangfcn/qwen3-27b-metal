"""Explicit little-endian encoders for the runtime image headers.

The C headers use ordinary 64-bit C ABI alignment.  We compute the same offsets
instead of depending on Python/native struct ABI, then write a zero-filled
4096-byte page.  This keeps the compiler portable while matching the current
Apple runtime image contracts.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from qwen38_constants import *


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    count: int = 1


def _size_align(field: Field) -> tuple[int, int]:
    if field.kind == "u32" or field.kind == "f32": return 4 * field.count, 4
    if field.kind == "u64": return 8 * field.count, 8
    if field.kind == "bytes": return field.count, 1
    raise ValueError(field.kind)


def layout(fields: list[Field]) -> tuple[dict[str, int], int]:
    offsets: dict[str, int] = {}
    pos = 0; max_align = 1
    for f in fields:
        size, align = _size_align(f)
        pos = (pos + align - 1) // align * align
        offsets[f.name] = pos
        pos += size; max_align = max(max_align, align)
    pos = (pos + max_align - 1) // max_align * max_align
    return offsets, pos


def _page(fields: list[Field], values: dict, page_size: int = 4096) -> bytes:
    offsets, used = layout(fields)
    if used > page_size:
        raise ValueError(f"header fields use {used} bytes > page {page_size}")
    b = bytearray(page_size)
    for f in fields:
        if f.name not in values:
            continue
        off = offsets[f.name]; v = values[f.name]
        if f.kind == "u32":
            vals = [v] if f.count == 1 else list(v)
            struct.pack_into("<" + "I" * f.count, b, off, *vals)
        elif f.kind == "u64":
            vals = [v] if f.count == 1 else list(v)
            struct.pack_into("<" + "Q" * f.count, b, off, *vals)
        elif f.kind == "f32":
            vals = [v] if f.count == 1 else list(v)
            struct.pack_into("<" + "f" * f.count, b, off, *vals)
        elif f.kind == "bytes":
            raw = bytes(v)
            if len(raw) > f.count:
                raise ValueError(f"{f.name}: {len(raw)} bytes > field {f.count}")
            b[off:off + len(raw)] = raw
    return bytes(b)


DELTA_FIELDS = [
    Field("magic", "bytes", 8),
    *[Field(n, "u32") for n in (
        "version","header_bytes","hidden_size","rows","group_size","reserved0",
        "source_reference_count","reserved1","down_rows","down_groups_per_row",
        "source_mlp_reference_count","reserved2")],
    *[Field(n, "u64") for n in (
        "gate_quants_offset","gate_quants_bytes","gate_metadata_offset","gate_metadata_bytes",
        "up_quants_offset","up_quants_bytes","up_metadata_offset","up_metadata_bytes",
        "down_quants_offset","down_quants_bytes","down_metadata_offset","down_metadata_bytes")],
    Field("source_sha256", "bytes", 65), Field("reference_alignment", "bytes", 3),
    Field("source_reference_first_8", "f32", 8), Field("source_mlp_reference_first_8", "f32", 8),
    *[Field(n, "u32") for n in (
        "layer_index","delta_input_rows","delta_input_groups_per_row","delta_output_rows",
        "delta_output_groups_per_row","constants_f32_count","source_layer_reference_count","reserved3")],
    *[Field(n, "u64") for n in (
        "constants_offset","constants_bytes","input_norm_constants_index","post_norm_constants_index",
        "conv_constants_index","a_log_constants_index","dt_bias_constants_index","recurrent_norm_constants_index",
        "delta_input_quants_offset","delta_input_quants_bytes","delta_input_metadata_offset","delta_input_metadata_bytes",
        "delta_output_quants_offset","delta_output_quants_bytes","delta_output_metadata_offset","delta_output_metadata_bytes")],
    Field("delta_input_precision", "u32"), Field("delta_output_precision", "u32"),
    Field("source_layer_reference_first_8", "f32", 8),
    Field("mlp_source_sha256", "bytes", 65),
]
DELTA_OFFSETS, DELTA_USED = layout(DELTA_FIELDS)

ATTENTION_FIELDS = [
    Field("magic", "bytes", 8),
    *[Field(n,"u32") for n in (
        "version","header_bytes","layer_index","hidden_size","intermediate_size","group_size",
        "q_heads","kv_heads","head_size","rotary_size","input_rows","input_groups_per_row",
        "output_rows","output_groups_per_row","constants_f32_count","reserved0")],
    *[Field(n,"u64") for n in (
        "gate_quants_offset","gate_quants_bytes","gate_metadata_offset","gate_metadata_bytes",
        "up_quants_offset","up_quants_bytes","up_metadata_offset","up_metadata_bytes",
        "down_quants_offset","down_quants_bytes","down_metadata_offset","down_metadata_bytes",
        "constants_offset","constants_bytes","input_norm_constants_index","post_norm_constants_index",
        "q_norm_constants_index","k_norm_constants_index","attention_input_quants_offset","attention_input_quants_bytes",
        "attention_input_metadata_offset","attention_input_metadata_bytes","attention_output_quants_offset","attention_output_quants_bytes",
        "attention_output_metadata_offset","attention_output_metadata_bytes")],
    Field("source_sha256","bytes",65), Field("source_alignment","bytes",7),
]
ATTENTION_OFFSETS, ATTENTION_USED = layout(ATTENTION_FIELDS)

GLOBAL_FIELDS = [
    Field("magic","bytes",8),
    *[Field(n,"u32") for n in (
        "version","header_bytes","vocab_size","hidden_size","group_size","constants_f32_count","reserved0","reserved1")],
    *[Field(n,"u64") for n in (
        "embedding_quants_offset","embedding_quants_bytes","embedding_metadata_offset","embedding_metadata_bytes",
        "lm_head_quants_offset","lm_head_quants_bytes","lm_head_metadata_offset","lm_head_metadata_bytes",
        "constants_offset","constants_bytes")],
    Field("embedding_source_sha256","bytes",65), Field("lm_head_source_sha256","bytes",65),
    Field("source_alignment","bytes",6),
]
GLOBAL_OFFSETS, GLOBAL_USED = layout(GLOBAL_FIELDS)

# Reconstructed from qwen38_mtp_pack.c and the runtime validator.  The first
# payload starts at byte 4096, and all named fields below are consumed by the
# runtime.  Zero-filled ABI padding/reserved bytes are intentionally retained.
MTP_FIELDS = [
    Field("magic","bytes",8),
    *[Field(n,"u32") for n in (
        "version","header_bytes","hidden_size","fc_rows","fc_groups_per_row","group_size",
        "constants_f32_count","reserved0")],
    *[Field(n,"u64") for n in (
        "fc_quants_offset","fc_quants_bytes","fc_metadata_offset","fc_metadata_bytes",
        "constants_offset","constants_bytes","embedding_norm_constants_index",
        "hidden_norm_constants_index","final_norm_constants_index")],
    Field("source_sha256","bytes",65), Field("source_alignment","bytes",7),
]
MTP_OFFSETS, MTP_USED = layout(MTP_FIELDS)

TOKENIZER_FIELDS = [
    Field("magic","bytes",8),
    *[Field(n,"u32") for n in (
        "version","header_bytes","vocab_size","base_vocab_size","added_first","added_count",
        "merge_count","token_entry_bytes")],
    *[Field(n,"u64") for n in (
        "token_directory_offset","token_directory_bytes","token_blob_offset","token_blob_bytes",
        "merges_offset","merges_bytes")],
    Field("source_sha256","bytes",65), Field("source_alignment","bytes",3),
    Field("byte_token_ids","u32",256),
]
TOKENIZER_OFFSETS, TOKENIZER_USED = layout(TOKENIZER_FIELDS)


def build_delta_header(values: dict) -> bytes:
    return _page(DELTA_FIELDS, {"magic": IMAGE_MAGIC, "version": IMAGE_VERSION,
                                "header_bytes": IMAGE_HEADER_BYTES, **values}, IMAGE_HEADER_BYTES)


def build_attention_header(values: dict) -> bytes:
    return _page(ATTENTION_FIELDS, {"magic": ATTENTION_MAGIC, "version": ATTENTION_IMAGE_VERSION,
                                    "header_bytes": ATTENTION_HEADER_BYTES, **values}, ATTENTION_HEADER_BYTES)


def build_global_header(values: dict) -> bytes:
    return _page(GLOBAL_FIELDS, {"magic": GLOBAL_MAGIC, "version": GLOBAL_IMAGE_VERSION,
                                 "header_bytes": GLOBAL_HEADER_BYTES, **values}, GLOBAL_HEADER_BYTES)


def build_mtp_header(values: dict) -> bytes:
    return _page(MTP_FIELDS, {"magic": MTP_MAGIC, "version": MTP_IMAGE_VERSION,
                              "header_bytes": MTP_HEADER_BYTES, **values}, MTP_HEADER_BYTES)


def build_tokenizer_header(values: dict) -> bytes:
    return _page(TOKENIZER_FIELDS, {"magic": TOKENIZER_MAGIC, "version": TOKENIZER_VERSION,
                                    "header_bytes": TOKENIZER_HEADER_BYTES, **values}, TOKENIZER_HEADER_BYTES)

# Sanity checks against offsets already consumed by verify_hybrid.py / C headers.
assert DELTA_OFFSETS["gate_quants_offset"] == 56
assert DELTA_OFFSETS["layer_index"] == 284
assert DELTA_OFFSETS["delta_input_quants_offset"] == 384
assert DELTA_OFFSETS["delta_input_precision"] == 448
assert ATTENTION_OFFSETS["gate_quants_offset"] == 72
assert ATTENTION_OFFSETS["attention_input_quants_offset"] == 216
assert GLOBAL_OFFSETS["embedding_quants_offset"] == 40
assert TOKENIZER_OFFSETS["token_directory_offset"] == 40
