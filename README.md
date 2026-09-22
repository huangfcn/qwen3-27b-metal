# qwen27b-apple-metal

A custom C / Objective-C runtime with Metal kernels that runs the Qwen 27B
hybrid (DeltaNet + full-attention) language models end to end on Apple
Silicon. It supports batched prompt prefill, incremental conversation state,
greedy and sampled decoding, adaptive MTP and DFlash2 speculative decoding,
and an OpenAI-compatible local server.

The measured record in this repository is for **Qwen3.8-27B** on an **Apple
M3 Pro** (36 GB). Qwen3.6-27B shares the same graph shape (64 layers, 5,120
hidden, 48 DeltaNet + 16 full-attention layers, 17,408 intermediate) and the
same runtime; its checkpoint pins and measurements are a separate validation.

This runtime is designed around **agent workloads, not short-context headline
throughput**. Coding agents repeatedly append source files, tool results,
patches, compiler output and conversation history, so useful sessions quickly
grow far beyond 8K tokens. Both prompt-prefill speed and generation speed at
32K–80K context therefore matter more than peak throughput on a short prompt.

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
| Apple M3 Pro (11-core CPU, 14-core GPU, 36 GB) | extensively tested |
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

## Agent throughput at long context

Short-context generation speed is a poor proxy for coding-agent usability.
An agent may begin below 8K tokens, but source files, search results, tool
schemas, compiler output and previous edits can push the active context into
the tens of thousands of tokens. At that point:

- **Prefill speed** determines how quickly newly appended tool output and code
  can be incorporated before the next model step.
- **Generation speed** determines the latency of reasoning, tool calls and
  edits after prefill.
- A runtime that is fast only below 8K context can feel substantially slower
  during the later stages of a real agent session.

The table below summarizes the context-scaling behavior observed on the pinned
M3 Pro configuration with Flash prefill enabled and DFlash2 speculative
decoding. Prefill values are smoothed from the long-running agent trace; the
trace reaches about 78K tokens before an agent compaction. Values at 80K are
therefore near-measured estimates, while 96K is an extrapolation. DFlash2
generation is workload-dependent because draft acceptance varies with the
generated text, so generation is shown as a representative range rather than
a single deterministic number.

| Active context | Prefill | DFlash2 generation |
|---:|---:|---:|
| 1K | ~89 tok/s | ~18–20 tok/s |
| 2K | ~88 tok/s | ~18–19 tok/s |
| 4K | ~86 tok/s | ~17–19 tok/s |
| 8K | ~82 tok/s | ~16–18 tok/s |
| 16K | ~75 tok/s | ~15–17 tok/s |
| 24K | ~69 tok/s | ~14–16 tok/s |
| 32K | ~63 tok/s | ~13–15 tok/s |
| 48K | ~54 tok/s | ~12–14 tok/s |
| 64K | ~48 tok/s | ~11–14 tok/s |
| 80K | ~43–44 tok/s | ~10–12 tok/s |
| 96K* | ~39–40 tok/s | ~9–11 tok/s |

\* 96K is extrapolated from the measured long-context trend; the recorded
agent session compacted before reaching that context length.

The important point is not the peak number at 1K context. It is that the
runtime remains useful as an agent session grows: prefill is still roughly
**63 tok/s at 32K, 54 tok/s at 48K, 48 tok/s at 64K and 43–44 tok/s near
80K**, while DFlash2 generation remains commonly in the low-to-mid teens over
much of the measured long-context range.

For reference, earlier local comparisons on this machine put generation at
about **5.6 tok/s for llama.cpp** and **8 tok/s for MLX**. Those figures are
configuration- and workload-dependent and should not be read as
quality-equivalent benchmark scores. The practical distinction this project
targets is sustained throughput under the **long contexts that coding agents
actually create**, rather than maximizing a short-prompt benchmark.

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
# Prebuilt Q4+Q8 + MTP + DFlash2 images
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
QWEN38_MODEL_DIR=./models/qwen38-runtime \
    commands/qwen38_chat.sh 'Explain pages.'

# OpenAI-compatible server (http://127.0.0.1:8080/v1)
python3 server/qwen38_serve.py --model-dir ./models/qwen38-runtime
```

The server serves both `POST /v1/chat/completions` and
`POST /v1/responses`. It accepts standard sampling fields;
`reasoning_effort` enables the thinking templates, and streaming responses
separate `reasoning_content` from `content`. The Python process implements
only HTTP and template adaptation — inference stays inside the resident
C/Metal process.

## Notes

- Model weights and the packed images are not committed; every packer
  re-checks the pinned SHA-256 values before writing, and the runtime
  rejects the wrong model or format instead of attempting compatibility
  fallback.
- The model identifier served is `qwen3.8-27b`; the compiled image
  collection for this model lives on Hugging Face as
  `huangfcn/Qwen3.8-27B-DFlash2-Metal`.
- This repository is extracted from the larger `llm-in-c` effort.
