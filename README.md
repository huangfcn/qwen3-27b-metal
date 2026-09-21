# qwen27b-apple-metal

A custom C / Objective-C runtime with Metal kernels that runs the Qwen 27B
hybrid (DeltaNet + full-attention) language models end to end on Apple
Silicon. It supports batched prompt prefill, incremental conversation state,
greedy and sampled decoding, adaptive MTP and DFlash2 speculative decoding,
and an OpenAI-compatible local server for Codex and opencode.

The measured record in this repository is for
**Qwen3.8-27B** on an **Apple M3 Pro** (36 GB). Qwen3.6-27B shares the same
graph shape (64 layers, 5,120 hidden, 48 DeltaNet + 16 full-attention layers,
17,408 intermediate) and the same runtime; its checkpoint pins and
measurements are a separate validation.

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
| Apple M3 Pro (11-core CPU, 14-core GPU, 36 GB) | extensively tested — full records in [runtime/README.md](runtime/README.md) |
| Apple M2 / M2 Pro / M2 Max | compatible — same unified-memory Metal 3 runtime, testing needed |
| Apple M3 / M3 Max | compatible — same runtime, testing needed |
| Apple M4 / M4 Pro / M4 Max | compatible — same runtime, testing needed |
| Apple M5 | compatible — same runtime, testing needed |

## Repository layout

| Path | Contents |
|---|---|
| [`runtime/`](runtime/) | C/Objective-C runtime, image-format headers, Metal kernels, tokenizer, sampler; the M3 Pro target record |
| [`compiler/`](compiler/) | Offline checkpoint inspection, Q4/Q8 quantizers, runtime-image packers, image-compile driver |
| [`commands/`](commands/) | Chat (`qwen38_m3_chat.c`) and one-shot (`qwen38_m3_generate.c`) front ends |
| [`server/`](server/) | OpenAI-compatible / Responses server (`qwen38_serve.py`) and Codex/opencode launchers |

## Build

Apple Silicon with the macOS command-line tools (or Xcode) and a C11 /
Objective-C compiler:

```sh
make                # metallib + chat/generate binaries + all compiler tools
make qwen38-m3-chat # runtime only
make qwen38-tools   # offline compiler tools only
```

Outputs land in `build/`:

| Artifact | Purpose |
|---|---|
| `build/qwen38-m3-q4.metallib` | compiled Metal kernels, loaded by the runtime |
| `build/qwen38-m3-chat` | resident terminal / machine-protocol chat process |
| `build/qwen38-m3-generate` | one-shot generation |
| `build/qwen38-m3-pack` | packs one DeltaNet or attention layer image |
| `build/qwen38-m3-attention-pack` / `qwen38-m3-global-pack` | attention-layer and global-image packers |
| `build/qwen38-mtp-pack` | MTP draft-layer / projection packer |
| `build/qwen38-tokenizer-pack` | tokenizer image packer |
| `build/qwen38-safetensors-inspect` | checkpoint inspection |

## Run

Prepare the compiled images once from the pinned checkpoints (see
[runtime/README.md](runtime/README.md) for the exact packer invocations and
SHA-256 pins):

```sh
make qwen38-tools
compiler/qwen38_compile_runtime_images.sh \
   model-00001-of-00003.safetensors \
   model-00002-of-00003.safetensors \
   model-00003-of-00003.safetensors \
   tokenizer.json \
   ./models/qwen38-runtime ./tmp/q8_all
```

Then choose an interface:

```sh
# Terminal chat (resident process)
QWEN38_MODEL_DIR=./models/qwen38-runtime commands/qwen38_chat.sh --terminal

# One prompt
QWEN38_MODEL_DIR=./models/qwen38-runtime commands/qwen38_chat.sh 'Explain pages.'

# OpenAI-compatible server (http://127.0.0.1:8199/v1)
python3 server/qwen38_serve.py --model-dir ./models/qwen38-runtime

# Codex CLI / opencode against this runtime
server/codex-qwen
server/opencode-qwen
```

The server accepts standard sampling fields; `reasoning_effort` enables the
thinking templates, and streaming responses separate `reasoning_content` from
`content`. The Python process implements only HTTP and template adaptation —
inference stays inside the resident C/Metal process.

## Notes

- Model weights and the packed images are not committed; every packer re-checks
  the pinned SHA-256 values before writing, and the runtime rejects the wrong
  model or format instead of attempting compatibility fallback.
- The model identifier served is `qwen3.8-27b`; the compiled image collection
  for this model lives on Hugging Face as
  `baryhuang/Qwen3.8-27B-Q4Q8-Apple-Metal`.
- This repository is extracted from the larger `llm-in-c` effort; the
  benchmark records referenced in [runtime/README.md](runtime/README.md)
  remain upstream.
