from __future__ import annotations

from pathlib import Path
import numpy as np

from qwen38_io_utils import require_sized, copy_path_at

META_CHUNK_VALUES = 1 << 20


def _dir(root: Path, layer: int | None) -> Path:
    return root / ("global" if layer is None else f"layer-{layer:02d}")


def plane_path(root: Path, layer: int | None, base: str, suffix: str) -> Path:
    return _dir(root, layer) / f"{base}{suffix}"


def copy_vector_f32(root: Path, layer: int | None, base: str, out, offset: int, values: int) -> None:
    copy_path_at(plane_path(root, layer, base, ".f32"), out, offset, values * 4)


def interleave_f16(scale_path: Path, bias_path: Path, out, offset: int, groups: int) -> None:
    require_sized(scale_path, groups * 2)
    require_sized(bias_path, groups * 2)
    with scale_path.open("rb", buffering=0) as sf, bias_path.open("rb", buffering=0) as bf:
        out.seek(offset)
        done = 0
        while done < groups:
            count = min(META_CHUNK_VALUES, groups - done)
            sraw = sf.read(count * 2); braw = bf.read(count * 2)
            if len(sraw) != count * 2 or len(braw) != count * 2:
                raise IOError("short metadata read")
            s = np.frombuffer(sraw, dtype="<u2")
            b = np.frombuffer(braw, dtype="<u2")
            combined = np.empty(count * 2, dtype="<u2")
            combined[0::2] = s; combined[1::2] = b
            out.write(combined.tobytes())
            done += count


def put_q4(root: Path, layer: int | None, base: str, out,
           quants_offset: int, metadata_offset: int, rows: int, cols: int) -> tuple[int, int]:
    code_bytes = rows * cols // 2
    groups = rows * (cols // 64)
    copy_path_at(plane_path(root, layer, base, "_codes.u8"), out, quants_offset, code_bytes)
    interleave_f16(
        plane_path(root, layer, base, "_scale.f16"),
        plane_path(root, layer, base, "_bias.f16"),
        out, metadata_offset, groups)
    return quants_offset + code_bytes, metadata_offset + groups * 4


def put_q8_delta_input(root: Path, layer: int, out,
                       quants_offset: int, metadata_offset: int,
                       rows: int, cols: int) -> tuple[int, int]:
    code_bytes = rows * cols
    groups = rows * (cols // 64)
    base = "delta_input_q8"
    copy_path_at(plane_path(root, layer, base, "_codes.i8"), out, quants_offset, code_bytes)
    interleave_f16(
        plane_path(root, layer, base, "_scale.f16"),
        plane_path(root, layer, base, "_bias.f16"),
        out, metadata_offset, groups)
    return quants_offset + code_bytes, metadata_offset + groups * 4
