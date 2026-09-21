#!/usr/bin/env python3
"""One-command builder for a complete Qwen3.8-27B Apple runtime directory.

Builds the current hybrid target:
  * Q4 backbone/global weights
  * Q8 DeltaNet input projections
  * tokenizer
and optionally adds:
  * MTP: mtp-layer.q38att + mtp.q38mtp
  * DFlash2: dflash2.q38df2

The target Q4/Q8 images are verified byte-for-byte against the emitted plane
files by qwen38_verify_hybrid.py unless --no-verify is supplied.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from qwen38_constants import LAYER_COUNT
from qwen38_m3_pack import pack_delta
from qwen38_m3_attention_pack import pack_attention
from qwen38_m3_global_pack import pack_global
from qwen38_tokenizer_pack import pack_tokenizer
from qwen38_mtp_pack import pack_mtp, pack_mtp_bundle


def _run_module(module: str, args: list[str]) -> None:
    script = Path(__file__).resolve().parent / f"{module}.py"
    cmd = [sys.executable, str(script), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _write_or_resume(path: Path, resume: bool) -> bool:
    if path.exists():
        if resume:
            print(f"resume: {path} already exists", file=sys.stderr)
            return False
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def _copy(path_in: Path, path_out: Path, resume: bool) -> None:
    if not path_in.is_file():
        raise FileNotFoundError(path_in)
    if _write_or_resume(path_out, resume):
        shutil.copy2(path_in, path_out)


def _add_mtp(args, out: Path) -> None:
    """Build the complete two-file MTP payload from its source checkpoint."""
    if args.mtp_from:
        src = Path(args.mtp_from)
        if src.is_dir():
            src = src / "model.safetensors"
        if not src.is_file():
            raise FileNotFoundError(src)

        layer_dst = out / "mtp-layer.q38att"
        extras_dst = out / "mtp.q38mtp"
        make_layer = _write_or_resume(layer_dst, args.resume)
        make_extras = _write_or_resume(extras_dst, args.resume)
        if make_layer or make_extras:
            result = pack_mtp_bundle(
                src,
                layer_dst if make_layer else None,
                extras_dst if make_extras else None,
            )
            print(json.dumps(result, separators=(",", ":")))
        return

    # Legacy compatibility path: callers may still provide a prepacked layer
    # plus the source checkpoint for the extras image. --mtp-from never copies.
    if args.mtp_checkpoint or args.mtp_layer:
        if not (args.mtp_checkpoint and args.mtp_layer):
            raise ValueError("--mtp-checkpoint and --mtp-layer must be supplied together")
        _copy(Path(args.mtp_layer), out / "mtp-layer.q38att", args.resume)
        dst = out / "mtp.q38mtp"
        if _write_or_resume(dst, args.resume):
            print(json.dumps(pack_mtp(args.mtp_checkpoint, dst), separators=(",", ":")))


def build(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir) if args.work_dir else out.parent / (out.name + ".build")
    work.mkdir(parents=True, exist_ok=True)

    q4 = Path(args.q4_dir) if args.q4_dir else work / "q4_all"
    q8 = Path(args.q8_dir) if args.q8_dir else work / "q8_all"

    if args.bf16_index:
        index = Path(args.bf16_index)
        if not index.is_file():
            raise FileNotFoundError(index)
        if not q4.exists() or not any(q4.iterdir()):
            _run_module(
                "qwen38_q4_quantize_all",
                [str(index), str(q4)] + (["--no-check"] if args.no_quant_check else []),
            )
        else:
            print(f"reuse Q4 planes: {q4}")
        if not q8.exists() or not any(q8.iterdir()):
            _run_module(
                "qwen38_q8_quantize_all",
                [str(index), str(q8)] + (["--no-check"] if args.no_quant_check else []),
            )
        else:
            print(f"reuse Q8 planes: {q8}")
    elif not (q4.is_dir() and q8.is_dir()):
        raise ValueError("provide --bf16-index or both --q4-dir and --q8-dir")

    # 64 target layers: every fourth layer is full attention; the rest are DeltaNet.
    for layer in range(LAYER_COUNT):
        if layer % 4 == 3:
            path = out / f"layer-{layer:02d}.q38att"
            if _write_or_resume(path, args.resume):
                print(json.dumps(pack_attention(q4, path, layer), separators=(",", ":")))
        else:
            path = out / f"layer-{layer:02d}.q38delta"
            if _write_or_resume(path, args.resume):
                print(json.dumps(pack_delta(q4, q8, path, layer), separators=(",", ":")))

    global_path = out / "global.q38global"
    if _write_or_resume(global_path, args.resume):
        print(json.dumps(pack_global(q4, global_path), separators=(",", ":")))

    tokenizer = Path(args.tokenizer_json) if args.tokenizer_json else None
    if tokenizer is None and args.bf16_index:
        candidate = Path(args.bf16_index).parent / "tokenizer.json"
        if candidate.exists():
            tokenizer = candidate
    if tokenizer is not None:
        tok_path = out / "tokenizer.q38tok"
        if _write_or_resume(tok_path, args.resume):
            print(
                json.dumps(
                    pack_tokenizer(
                        tokenizer,
                        tok_path,
                        allow_unpinned=args.allow_unpinned_tokenizer,
                    ),
                    separators=(",", ":"),
                )
            )
    elif not (out / "tokenizer.q38tok").exists():
        print("warning: tokenizer.q38tok was not created", file=sys.stderr)

    # DFlash2 is self-contained and is packed directly from its official checkpoint.
    if args.dflash2_dir:
        dflash_path = out / "dflash2.q38df2"
        if _write_or_resume(dflash_path, args.resume):
            _run_module(
                "qwen38_dflash2_pack",
                ["--draft-dir", args.dflash2_dir, "--out", str(dflash_path)],
            )

    # MTP remains a two-file payload.  See _add_mtp() for supported provenance.
    _add_mtp(args, out)

    if not args.no_verify:
        _run_module("qwen38_verify_hybrid", [str(out), str(q4), str(q8)])

    if args.cleanup_planes and args.bf16_index and not args.q4_dir and not args.q8_dir:
        shutil.rmtree(q4, ignore_errors=True)
        shutil.rmtree(q8, ignore_errors=True)

    artifacts = {
        "target_layers": sum(1 for _ in out.glob("layer-*.q38*")),
        "global": (out / "global.q38global").is_file(),
        "tokenizer": (out / "tokenizer.q38tok").is_file(),
        "mtp_layer": (out / "mtp-layer.q38att").is_file(),
        "mtp_extras": (out / "mtp.q38mtp").is_file(),
        "dflash2": (out / "dflash2.q38df2").is_file(),
    }
    return {
        "model_dir": str(out),
        "q4_dir": str(q4),
        "q8_dir": str(q8),
        "verified_target": not args.no_verify,
        "artifacts": artifacts,
        "complete_mtp": artifacts["mtp_layer"] and artifacts["mtp_extras"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build Qwen3.8-27B Q4+Q8 DeltaNet runtime images in Python, "
            "optionally including MTP and DFlash2"
        )
    )
    ap.add_argument("--bf16-index", help="BF16 model.safetensors.index.json")
    ap.add_argument("--q4-dir", help="reuse pre-quantized Q4 plane directory")
    ap.add_argument("--q8-dir", help="reuse pre-quantized Q8 DeltaNet input plane directory")
    ap.add_argument("--tokenizer-json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-quant-check", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--cleanup-planes", action="store_true")
    ap.add_argument("--allow-unpinned-tokenizer", action="store_true")

    ap.add_argument(
        "--dflash2-dir",
        help="official Qwen3.8-27B-DFlash2 directory; writes dflash2.q38df2",
    )
    mtp = ap.add_mutually_exclusive_group()
    mtp.add_argument(
        "--mtp-from",
        help=(
            "pinned mlx-community/Qwen3.8-27B-MTP-4bit/model.safetensors "
            "(or its containing directory); builds mtp-layer.q38att + mtp.q38mtp"
        ),
    )
    mtp.add_argument(
        "--mtp-checkpoint",
        help="pinned Qwen3.8 MTP model.safetensors used to build mtp.q38mtp",
    )
    ap.add_argument(
        "--mtp-layer",
        help="known-good mtp-layer.q38att; required with --mtp-checkpoint",
    )

    args = ap.parse_args(argv)
    if not args.bf16_index and not (args.q4_dir and args.q8_dir):
        ap.error("provide --bf16-index or both --q4-dir and --q8-dir")
    if args.mtp_checkpoint and not args.mtp_layer:
        ap.error("--mtp-layer is required with --mtp-checkpoint")
    if args.mtp_layer and not args.mtp_checkpoint:
        ap.error("--mtp-checkpoint is required with --mtp-layer")

    try:
        result = build(args)
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    except Exception as exc:
        print(f"qwen38_build_model: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
