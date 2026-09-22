# qwen27b-apple-metal
#
# Build the Apple Silicon (Metal) runtime for the Qwen 27B hybrid
# (DeltaNet + full-attention) language model.
#
# Layout:
#     runtime/   C / Objective-C runtime + Metal kernels
#     compiler/  offline Python tools (checkpoint inspection, quantizers and
#                runtime-image packers) — interpreted, nothing to compile
#     commands/  chat and one-shot generation front ends
#     server/    OpenAI-compatible / Responses server and agent launchers
#        tests/        CPU and live-model test programs (see the test targets)
#        benchmarks/ per-module GPU micro-benchmarks with CPU error references
#
# Only the C / Objective-C / Metal stdlib and the system Metal, Foundation,
# CoreFoundation and ICU libraries are used. There is no external
# dependency and no generated code beyond the compiled metallib. The
# compiler tools in compiler/ are plain Python 3 scripts and need no build
# step (qwen38-tools is kept as a no-op for compatibility).

CC         := cc
XCRUN      := xcrun
SDK        := macosx
BUILD_DIR  := build

CFLAGS     := -O3 -std=c11 -Wall -Wextra -Wpedantic
OBJCFLAGS  := -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc

RUNTIME_DIR  := runtime
COMMANDS_DIR := commands
TESTS_DIR        := tests
BENCH_DIR        := benchmarks

# Default goal.
all: metallib qwen38-m3-chat qwen38-m3-generate

# ---------------------------------------------------------------------------
# Metal
# ---------------------------------------------------------------------------

QWEN38_M3_AIR            := $(BUILD_DIR)/qwen38-m3-q4.air
QWEN38_M3_DELTANET_AIR   := $(BUILD_DIR)/qwen38-m3-deltanet.air
QWEN38_M3_LAYER_AIR      := $(BUILD_DIR)/qwen38-m3-layer.air
QWEN38_M3_LAYER_Q8_AIR   := $(BUILD_DIR)/qwen38-m3-layer-q8.air
QWEN38_M3_ATTENTION_AIR  := $(BUILD_DIR)/qwen38-m3-attention.air
QWEN38_M3_GLOBAL_AIR     := $(BUILD_DIR)/qwen38-m3-global.air
QWEN38_M3_PREFILL_AIR    := $(BUILD_DIR)/qwen38-m3-prefill.air
QWEN38_M3_METALLIB       := $(BUILD_DIR)/qwen38-m3-q4.metallib

$(QWEN38_M3_METALLIB): $(QWEN38_M3_AIR) $(QWEN38_M3_DELTANET_AIR) \
                        $(QWEN38_M3_LAYER_AIR) $(QWEN38_M3_LAYER_Q8_AIR) \
                        $(QWEN38_M3_ATTENTION_AIR) $(QWEN38_M3_GLOBAL_AIR) \
                        $(QWEN38_M3_PREFILL_AIR)
	mkdir -p $(BUILD_DIR)
	$(XCRUN) -sdk $(SDK) metallib $^ -o $@

$(QWEN38_M3_AIR): $(RUNTIME_DIR)/qwen38_q4.metal
	mkdir -p $(BUILD_DIR)
	$(XCRUN) -sdk $(SDK) metal -c $< -o $@

$(QWEN38_M3_DELTANET_AIR): $(RUNTIME_DIR)/qwen38_deltanet.metal
	mkdir -p $(BUILD_DIR)
	$(XCRUN) -sdk $(SDK) metal -c $< -o $@

$(QWEN38_M3_LAYER_AIR): $(RUNTIME_DIR)/qwen38_layer.metal
	mkdir -p $(BUILD_DIR)
	$(XCRUN) -sdk $(SDK) metal -c $< -o $@

$(QWEN38_M3_LAYER_Q8_AIR): $(RUNTIME_DIR)/qwen38_layer_q8.metal
	mkdir -p $(BUILD_DIR)
	$(XCRUN) -sdk $(SDK) metal -c $< -o $@

$(QWEN38_M3_ATTENTION_AIR): $(RUNTIME_DIR)/qwen38_attention.metal
	mkdir -p $(BUILD_DIR)
	$(XCRUN) -sdk $(SDK) metal -c $< -o $@

$(QWEN38_M3_GLOBAL_AIR): $(RUNTIME_DIR)/qwen38_global.metal
	mkdir -p $(BUILD_DIR)
	$(XCRUN) -sdk $(SDK) metal -c $< -o $@

$(QWEN38_M3_PREFILL_AIR): $(RUNTIME_DIR)/qwen38_prefill.metal
	mkdir -p $(BUILD_DIR)
	$(XCRUN) -sdk $(SDK) metal -c $< -o $@

metallib: $(QWEN38_M3_METALLIB)

# ---------------------------------------------------------------------------
# Runtime objects
# ---------------------------------------------------------------------------

QWEN38_M3_RUNTIME_OBJECT := $(BUILD_DIR)/qwen38-m3.o
QWEN38_M3_DECODE_OBJECT   := $(BUILD_DIR)/qwen38-m3-decode.o
QWEN38_TOKENIZER_OBJECT   := $(BUILD_DIR)/qwen38-tokenizer.o
QWEN38_SAMPLER_OBJECT     := $(BUILD_DIR)/qwen38-sampler.o

$(QWEN38_M3_RUNTIME_OBJECT): $(RUNTIME_DIR)/qwen38_m3.m $(RUNTIME_DIR)/qwen38_m3.h \
                              $(RUNTIME_DIR)/qwen38_m3_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -I$(RUNTIME_DIR) -c $< -o $@

$(QWEN38_M3_DECODE_OBJECT): $(RUNTIME_DIR)/qwen38_m3_decode.m \
                             $(RUNTIME_DIR)/qwen38_m3_decode.h \
                             $(RUNTIME_DIR)/qwen38_m3.h \
                             $(RUNTIME_DIR)/qwen38_m3_image.h \
                             $(RUNTIME_DIR)/qwen38_m3_attention_image.h \
                             $(RUNTIME_DIR)/qwen38_m3_global_image.h \
                             $(RUNTIME_DIR)/qwen38_m3_mtp_image.h \
                             $(RUNTIME_DIR)/qwen38_m3_dflash2_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -I$(RUNTIME_DIR) -c $< -o $@

$(QWEN38_TOKENIZER_OBJECT): $(RUNTIME_DIR)/qwen38_tokenizer.c \
                             $(RUNTIME_DIR)/qwen38_tokenizer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) -c $< -o $@

$(QWEN38_SAMPLER_OBJECT): $(RUNTIME_DIR)/qwen38_sampler.c \
                           $(RUNTIME_DIR)/qwen38_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) -c $< -o $@

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

QWEN38_M3_GENERATE_OBJECT := $(BUILD_DIR)/qwen38-m3-generate-cli.o
QWEN38_M3_GENERATE        := $(BUILD_DIR)/qwen38-m3-generate
QWEN38_M3_CHAT_OBJECT     := $(BUILD_DIR)/qwen38-m3-chat-cli.o
QWEN38_M3_CHAT            := $(BUILD_DIR)/qwen38-m3-chat

$(QWEN38_M3_GENERATE_OBJECT): $(COMMANDS_DIR)/qwen38_m3_generate.c \
                               $(RUNTIME_DIR)/qwen38_m3_decode.h \
                               $(RUNTIME_DIR)/qwen38_tokenizer.h \
                               $(RUNTIME_DIR)/qwen38_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) -c $< -o $@

$(QWEN38_M3_GENERATE): $(QWEN38_M3_DECODE_OBJECT) \
                        $(QWEN38_TOKENIZER_OBJECT) $(QWEN38_SAMPLER_OBJECT) \
                        $(QWEN38_M3_GENERATE_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal \
	-framework CoreFoundation -licucore -lm

$(QWEN38_M3_CHAT_OBJECT): $(COMMANDS_DIR)/qwen38_m3_chat.c \
                           $(RUNTIME_DIR)/qwen38_m3_decode.h \
                           $(RUNTIME_DIR)/qwen38_tokenizer.h \
                           $(RUNTIME_DIR)/qwen38_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) -c $< -o $@

$(QWEN38_M3_CHAT): $(QWEN38_M3_DECODE_OBJECT) \
                    $(QWEN38_TOKENIZER_OBJECT) $(QWEN38_SAMPLER_OBJECT) \
                    $(QWEN38_M3_CHAT_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal \
	-framework CoreFoundation -licucore -lm

qwen38-m3-generate: $(QWEN38_M3_GENERATE) $(QWEN38_M3_METALLIB)
qwen38-m3-chat: $(QWEN38_M3_CHAT) $(QWEN38_M3_METALLIB)

# ---------------------------------------------------------------------------
# Tests
#
# qwen38-sampler-test is CPU-only and needs no model. The live tests
# (api-state, prefill-parity, dflash2) take <model-directory> and the
# metallib as arguments; `live-test` runs them against MODEL_DIR
# (default models/qwen38-runtime).
# ---------------------------------------------------------------------------

QWEN38_SAMPLER_TEST               := $(BUILD_DIR)/qwen38-sampler-test
QWEN38_M3_API_STATE_TEST          := $(BUILD_DIR)/qwen38-m3-api-state-test
QWEN38_M3_PREFILL_PARITY_TEST     := $(BUILD_DIR)/qwen38-m3-prefill-parity-test
QWEN38_M3_DFLASH2_TEST            := $(BUILD_DIR)/qwen38-m3-dflash2-test

$(QWEN38_SAMPLER_TEST): $(TESTS_DIR)/qwen38_sampler_test.c \
		   $(RUNTIME_DIR)/qwen38_sampler.c $(RUNTIME_DIR)/qwen38_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) \
	   $(TESTS_DIR)/qwen38_sampler_test.c $(RUNTIME_DIR)/qwen38_sampler.c \
	   -o $@ -lm

$(QWEN38_M3_API_STATE_TEST): $(TESTS_DIR)/qwen38_m3_api_state_test.c \
	   $(QWEN38_M3_DECODE_OBJECT) $(QWEN38_TOKENIZER_OBJECT) \
	   $(QWEN38_SAMPLER_OBJECT) $(RUNTIME_DIR)/qwen38_m3_decode.h \
	   $(RUNTIME_DIR)/qwen38_m3_global_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) \
	   $(TESTS_DIR)/qwen38_m3_api_state_test.c $(QWEN38_M3_DECODE_OBJECT) \
	   $(QWEN38_TOKENIZER_OBJECT) $(QWEN38_SAMPLER_OBJECT) \
	   -o $@ -framework Foundation -framework Metal \
	   -framework CoreFoundation -licucore -lm

$(QWEN38_M3_PREFILL_PARITY_TEST): $(TESTS_DIR)/qwen38_m3_prefill_parity_test.c \
	   $(QWEN38_M3_DECODE_OBJECT) $(QWEN38_TOKENIZER_OBJECT) \
	   $(QWEN38_SAMPLER_OBJECT) $(RUNTIME_DIR)/qwen38_m3_decode.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) \
	   $(TESTS_DIR)/qwen38_m3_prefill_parity_test.c $(QWEN38_M3_DECODE_OBJECT) \
	   $(QWEN38_TOKENIZER_OBJECT) $(QWEN38_SAMPLER_OBJECT) \
	   -o $@ -framework Foundation -framework Metal \
	   -framework CoreFoundation -licucore -lm

$(QWEN38_M3_DFLASH2_TEST): $(TESTS_DIR)/qwen38_m3_dflash2_test.c \
	   $(QWEN38_M3_DECODE_OBJECT) $(QWEN38_TOKENIZER_OBJECT) \
	   $(QWEN38_SAMPLER_OBJECT) $(RUNTIME_DIR)/qwen38_m3_decode.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) \
	   $(TESTS_DIR)/qwen38_m3_dflash2_test.c $(QWEN38_M3_DECODE_OBJECT) \
	   $(QWEN38_TOKENIZER_OBJECT) $(QWEN38_SAMPLER_OBJECT) \
	   -o $@ -framework Foundation -framework Metal \
	   -framework CoreFoundation -licucore -lm

qwen38-sampler-test: $(QWEN38_SAMPLER_TEST)
qwen38-m3-api-state-test: $(QWEN38_M3_API_STATE_TEST) $(QWEN38_M3_METALLIB)
qwen38-m3-prefill-parity-test: $(QWEN38_M3_PREFILL_PARITY_TEST) $(QWEN38_M3_METALLIB)
qwen38-m3-dflash2-test: $(QWEN38_M3_DFLASH2_TEST) $(QWEN38_M3_METALLIB)

# Build every test binary and run the model-free CPU test.
test: qwen38-sampler-test qwen38-m3-api-state-test \
	qwen38-m3-prefill-parity-test qwen38-m3-dflash2-test
	$(QWEN38_SAMPLER_TEST)

# Live model tests; needs the packed images (see the Run section in README).
MODEL_DIR ?= models/qwen38-runtime
live-test: qwen38-m3-api-state-test qwen38-m3-prefill-parity-test \
	qwen38-m3-dflash2-test
	$(QWEN38_M3_API_STATE_TEST) $(MODEL_DIR) $(QWEN38_M3_METALLIB)
	$(QWEN38_M3_PREFILL_PARITY_TEST) $(MODEL_DIR) $(QWEN38_M3_METALLIB)
	$(QWEN38_M3_DFLASH2_TEST) $(MODEL_DIR) $(QWEN38_M3_METALLIB)

# ---------------------------------------------------------------------------
# Benchmarks
#
# Per-module GPU micro-benchmarks. Each one reports timing plus CPU
# reference error (max_abs_error_* fields), so a wrong module on any chip
# shows up as large errors or NaNs. The layer and attention ones take a
# packed layer image first, then the metallib.
# ---------------------------------------------------------------------------

QWEN38_M3_DELTANET_OBJECT    := $(BUILD_DIR)/qwen38-m3-deltanet.o
QWEN38_M3_LAYER_OBJECT       := $(BUILD_DIR)/qwen38-m3-layer.o
QWEN38_M3_ATTENTION_OBJECT   := $(BUILD_DIR)/qwen38-m3-attention.o
QWEN38_M3_MLP_BENCH           := $(BUILD_DIR)/qwen38-m3-mlp-bench
QWEN38_M3_DELTANET_BENCH      := $(BUILD_DIR)/qwen38-m3-deltanet-bench
QWEN38_M3_LAYER_BENCH          := $(BUILD_DIR)/qwen38-m3-layer-bench
QWEN38_M3_ATTENTION_BENCH      := $(BUILD_DIR)/qwen38-m3-attention-bench

$(QWEN38_M3_DELTANET_OBJECT): $(RUNTIME_DIR)/qwen38_m3_deltanet.m \
	   $(RUNTIME_DIR)/qwen38_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -I$(RUNTIME_DIR) -c $< -o $@

$(QWEN38_M3_LAYER_OBJECT): $(RUNTIME_DIR)/qwen38_m3_layer.m \
	   $(RUNTIME_DIR)/qwen38_m3.h $(RUNTIME_DIR)/qwen38_m3_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -I$(RUNTIME_DIR) -c $< -o $@

$(QWEN38_M3_ATTENTION_OBJECT): $(RUNTIME_DIR)/qwen38_m3_attention.m \
	   $(RUNTIME_DIR)/qwen38_m3.h $(RUNTIME_DIR)/qwen38_m3_image.h \
	   $(RUNTIME_DIR)/qwen38_m3_attention_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -I$(RUNTIME_DIR) -c $< -o $@

$(QWEN38_M3_MLP_BENCH): $(BENCH_DIR)/qwen38_m3_mlp_bench.c \
	   $(QWEN38_M3_RUNTIME_OBJECT) $(RUNTIME_DIR)/qwen38_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) \
	   $(BENCH_DIR)/qwen38_m3_mlp_bench.c $(QWEN38_M3_RUNTIME_OBJECT) \
	   -o $@ -framework Foundation -framework Metal -lm

$(QWEN38_M3_DELTANET_BENCH): $(BENCH_DIR)/qwen38_m3_deltanet_bench.c \
	   $(QWEN38_M3_DELTANET_OBJECT) $(RUNTIME_DIR)/qwen38_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) \
	   $(BENCH_DIR)/qwen38_m3_deltanet_bench.c $(QWEN38_M3_DELTANET_OBJECT) \
	   -o $@ -framework Foundation -framework Metal -lm

$(QWEN38_M3_LAYER_BENCH): $(BENCH_DIR)/qwen38_m3_layer_bench.c \
	   $(QWEN38_M3_LAYER_OBJECT) $(RUNTIME_DIR)/qwen38_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) \
	   $(BENCH_DIR)/qwen38_m3_layer_bench.c $(QWEN38_M3_LAYER_OBJECT) \
	   -o $@ -framework Foundation -framework Metal -lm

$(QWEN38_M3_ATTENTION_BENCH): $(BENCH_DIR)/qwen38_m3_attention_bench.c \
	   $(QWEN38_M3_ATTENTION_OBJECT) $(RUNTIME_DIR)/qwen38_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUNTIME_DIR) \
	   $(BENCH_DIR)/qwen38_m3_attention_bench.c $(QWEN38_M3_ATTENTION_OBJECT) \
	   -o $@ -framework Foundation -framework Metal -lm

benchmarks: qwen38-m3-mlp-bench qwen38-m3-deltanet-bench \
	qwen38-m3-layer-bench qwen38-m3-attention-bench

qwen38-m3-mlp-bench: $(QWEN38_M3_MLP_BENCH) $(QWEN38_M3_METALLIB)
qwen38-m3-deltanet-bench: $(QWEN38_M3_DELTANET_BENCH) $(QWEN38_M3_METALLIB)
qwen38-m3-layer-bench: $(QWEN38_M3_LAYER_BENCH) $(QWEN38_M3_METALLIB)
qwen38-m3-attention-bench: $(QWEN38_M3_ATTENTION_BENCH) $(QWEN38_M3_METALLIB)

# ---------------------------------------------------------------------------
# Compiler tools
#
# The offline compiler (checkpoint inspection, Q4/Q8 quantizers, layer /
# attention / global / tokenizer / MTP / DFlash2 packers) is plain Python 3
# in compiler/ and needs no build step. Kept as a no-op so existing scripts
# and documentation that call `make qwen38-tools` keep working.
# ---------------------------------------------------------------------------

qwen38-tools:
	@echo "compiler/ is pure Python; nothing to build."

# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

.PHONY: all metallib qwen38-m3-chat qwen38-m3-generate qwen38-tools \
	qwen38-sampler-test qwen38-m3-api-state-test qwen38-m3-prefill-parity-test \
	qwen38-m3-dflash2-test test live-test benchmarks qwen38-m3-mlp-bench \
	qwen38-m3-deltanet-bench qwen38-m3-layer-bench qwen38-m3-attention-bench clean

clean:
	rm -f $(QWEN38_M3_AIR) $(QWEN38_M3_DELTANET_AIR) $(QWEN38_M3_LAYER_AIR) \
	  $(QWEN38_M3_LAYER_Q8_AIR) $(QWEN38_M3_ATTENTION_AIR) \
	  $(QWEN38_M3_GLOBAL_AIR) $(QWEN38_M3_PREFILL_AIR) \
	  $(QWEN38_M3_METALLIB) \
	  $(QWEN38_M3_RUNTIME_OBJECT) $(QWEN38_M3_DECODE_OBJECT) \
	  $(QWEN38_TOKENIZER_OBJECT) $(QWEN38_SAMPLER_OBJECT) \
	  $(QWEN38_M3_DELTANET_OBJECT) $(QWEN38_M3_LAYER_OBJECT) \
	  $(QWEN38_M3_ATTENTION_OBJECT) \
	  $(QWEN38_M3_GENERATE_OBJECT) $(QWEN38_M3_GENERATE) \
	  $(QWEN38_M3_CHAT_OBJECT) $(QWEN38_M3_CHAT) \
	  $(QWEN38_SAMPLER_TEST) $(QWEN38_M3_API_STATE_TEST) \
	  $(QWEN38_M3_PREFILL_PARITY_TEST) $(QWEN38_M3_DFLASH2_TEST) \
	  $(QWEN38_M3_MLP_BENCH) $(QWEN38_M3_DELTANET_BENCH) \
	  $(QWEN38_M3_LAYER_BENCH) $(QWEN38_M3_ATTENTION_BENCH)
	rm -rf $(BUILD_DIR)
