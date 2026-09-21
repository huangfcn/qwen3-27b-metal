# qwen27b-apple-metal

A custom C / Objective-C runtime with Metal kernels that runs the Qwen 27B
hybrid (DeltaNet + full-attention) language models end to end on Apple
Silicon. It supports batched prompt prefill, incremental conversation state,
greedy and sampled decoding, adaptive MTP and DFlash2 speculative decoding,
and an OpenAI-compatible local server.

The measured record in this repository is for **Qwen3.8-27B** on an **Apple
M3 Pro** (36 GB). Qwen3.6-27B shares the same graph shape (64 layers, 5,120
hidden, 48 DeltaNet + 16 full-attention layers, 17,408 intermediate) and the
same runtime; its checkpoint pins and measurements are a separate
validation.

This runtime is a model-specific compiler output: the offline tools in
`compiler/` turn the pinned checkpoint shards into packed runtime images
(`.q38*` files) that the runtime maps directly. There is no general model
loader and no external dependency — only the C/Objective-C/Metal stdlib and
the system Metal, Foundation, CoreFoundation and ICU libraries.

## Hardware support

One Metal runtime serves every chip. Chip-specific behavior (chunk sizes,
memory limits, weight-pinning policy, context limits) is configured per chip;
a separate kernel is added only when measurements demonstrate a chip needs
one.

| Chip | Status |
|---|---|
| Apple M3 Pro (11-core CPU, 14-core GPU, 36 GB) | extensively tested — full records in this README |
| Apple M2 / M2 Pro / M2 Max | compatible — same unified-memory Metal 3 runtime, testing needed |
| Apple M3 / M3 Max | compatible — same runtime, testing needed |
| Apple M4 / M4 Pro / M4 Max | compatible — same runtime, testing needed |
| Apple M5 | compatible — same runtime, testing needed |

## Target and artifact (M3 Pro record)

| Property | Pinned value |
|---|---|
| Machine | MacBook Pro `Mac15,6` |
| SoC | Apple M3 Pro, 11-core CPU, 14-core Metal 3 GPU |
| Memory | 36 GB unified memory |
| Operating system | macOS 15.7.3 |
| Weight format | affine Q4, group size 64 |
| Compiled text image | 65 mapped files, 15,138,643,968 bytes, plus tokenizer |
| MTP images | 209,436,672-byte draft layer and 29,556,736-byte projection/norm image |
| Runtime state | 158.9 MB recurrent/convolution state; FP16 KV uses 64 KiB per context token |

The target maps immutable layer images rather than loading a general model
graph. Forty-eight DeltaNet layers use persistent recurrent/convolution
state; sixteen full-attention layers use grouped-query KV cache. Metal
kernels are specialized for the fixed matrix shapes, affine Q4 layout and
prompt-length buckets used by this graph.

## Repository layout

| Path | Contents |
|---|---|
| [`runtime/`](runtime/) | C/Objective-C runtime, image-format headers, Metal kernels, tokenizer, sampler |
| [`compiler/`](compiler/) | Pure-Python offline checkpoint inspection, Q4/Q8 quantizers, runtime-image packers and the one-command builder |
| [`commands/`](commands/) | Chat (`qwen38_m3_chat.c`) and one-shot (`qwen38_m3_generate.c`) front ends |
| [`server/`](server/) | OpenAI-compatible / Responses server (`qwen38_serve.py`) |

## Compiled image format

| File | Contents |
|---|---|
| `global.q38global` | embedding, final normalization and output projection |
| `layer-NN.q38delta` | one DeltaNet transformer layer |
| `layer-NN.q38att` | one full-attention transformer layer |
| `tokenizer.q38tok` | packed vocabulary, merges and special-token tables |
| `mtp-layer.q38att` | MTP draft attention layer |
| `mtp.q38mtp` | MTP input projection and normalization tensors |
| `dflash2.q38df2` | DFlash2 draft model (5 draft layers targeting layers 5/19/33/47/61) |

Every packer checks the expected source SHA-256 before writing an image.
Every image carries its format magic, version, source hash and tensor
metadata; the runtime rejects the wrong model or format instead of
attempting compatibility fallback.

## Build

Apple Silicon with the macOS command-line tools (or Xcode) and a C11 /
Objective-C compiler:

```sh
make                  # metallib + chat/generate binaries
make qwen38-m3-chat   # runtime only
```

Generating the model images is pure Python 3 (numpy) — the `compiler/`
directory has no build step.

Outputs land in `build/`:

| Artifact | Purpose |
|---|---|
| `build/qwen38-m3-q4.metallib` | compiled Metal kernels, loaded by the runtime |
| `build/qwen38-m3-chat` | resident terminal / machine-protocol chat process |
| `build/qwen38-m3-generate` | one-shot generation |

## Run

Prepare the compiled images once, either from the prebuilt collection or
from the pinned checkpoints (every packer re-checks the pinned SHA-256
values before writing):

```sh
# Prebuilt Q4+Q8 + MTP + DFlash2 images (65 files)
huggingface-cli download huangfcn/Qwen3.8-27B-DFlash2-Metal \
      --local-dir ./models/qwen38-runtime

# ...or generate every image from the pinned sources (Python 3 + numpy)
python3 compiler/qwen38_build_model.py \
      --bf16-index /path/to/qwen3.8-27b-bf16/model.safetensors.index.json \
      --tokenizer-json /path/to/qwen3.8-27b-bf16/tokenizer.json \
      --dflash2-dir /path/to/Qwen3.8-27B-DFlash2 \
      --mtp-from /path/to/Qwen3.8-27B-MTP-4bit/model.safetensors \
      --out ./models/qwen38-runtime \
      --cleanup-planes
```

The builder quantizes the Q4 and Q8 planes from the BF16 index, packs the 64
layers, `global.q38global` and `tokenizer.q38tok`, then adds
`mtp-layer.q38att` + `mtp.q38mtp` and `dflash2.q38df2`, and byte-verifies
the target images with `qwen38_verify_hybrid.py` (disable with
`--no-verify`).

Then choose an interface:

```sh
# Terminal chat (resident process)
QWEN38_MODEL_DIR=./models/qwen38-runtime commands/qwen38_chat.sh --terminal

# One prompt
QWEN38_MODEL_DIR=./models/qwen38-runtime commands/qwen38_chat.sh 'Explain pages.'

# OpenAI-compatible server (http://127.0.0.1:8080/v1)
python3 server/qwen38_serve.py --model-dir ./models/qwen38-runtime
```

The server serves both `POST /v1/chat/completions` and
`POST /v1/responses` (the Responses wire API Codex CLI speaks). It accepts
standard sampling fields; `reasoning_effort` enables the thinking templates,
and streaming responses separate `reasoning_content` from `content`. The
Python process implements only HTTP and template adaptation — inference
stays inside the resident C/Metal process.

## Measured throughput

The following five workloads ran through one resident chat process per arm
with greedy seed 42. End-to-end throughput is completion tokens divided by
the full request wall, including prompt prefill, first token and decode. The
speculative arm uses adaptive MTP with replay-free partial accepts, a
wide-tile eight-wide half-MMA verify, a depth-7 chain ceiling, four-gram
context-lookup drafting and a restricted draft-head vocabulary.

| Case | Output tokens | Plain end-to-end | Adaptive MTP end-to-end | Speedup | Request wall, plain / MTP |
|---|---:|---:|---:|---:|---:|
| C `max2` function | 28 | 6.36 tok/s | **15.01 tok/s** | 2.36× | 4.4 / 1.9 s |
| Hash-table explanation | 531 / 524 | 8.19 tok/s | **10.84 tok/s** | 1.32× | 64.8 / 48.3 s |
| Python `LRUCache` class | 1,155 | 7.98 tok/s | **17.59 tok/s** | 2.20× | 144.7 / 65.7 s |
| Virtual-memory essay | 1,463 | 8.01 tok/s | **9.94 tok/s** | 1.24× | 182.7 / 147.2 s |
| Notes summary, 159-token prompt | 128 | 6.99 tok/s | **9.87 tok/s** | 1.41× | 18.3 / 13.0 s |
| **Aggregate** | **3,305 / 3,298** | **7.96 tok/s** | **11.95 tok/s** | **1.50×** | **415.0 / 276.0 s** |

Decode scales down with conversation length because every step attends over
the whole KV cache. On a fixed 200-token generation (plain arm):

| Context | Decode |
|---:|---:|
| 1,314 tokens | 9.9 tok/s |
| 5,154 tokens | 9.3 tok/s |
| 12,834 tokens | 6.5 tok/s |

Code-heavy cases run at 18.0–40.8 decode tok/s because their draft chains
accept deeply; prose cases sit near 10.0–11.0, acceptance-limited. The full
per-case records (including llama.cpp and mlx-lm / oMLX comparisons) are in
`results.json` in the upstream `llm-in-c` repository.

## Notes

- Model weights and the packed images are not committed; every packer
  re-checks the pinned SHA-256 values before writing, and the runtime
  rejects the wrong model or format instead of attempting compatibility
  fallback.
- The model identifier served is `qwen3.8-27b`; the compiled image
  collection for this model lives on Hugging Face as
  `huangfcn/Qwen3.8-27B-DFlash2-Metal`.
- This repository is extracted from the larger `llm-in-c` effort; the
  benchmark records referenced above remain upstream.
