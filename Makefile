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

.PHONY: all metallib qwen38-m3-chat qwen38-m3-generate qwen38-tools clean

clean:
	rm -f $(QWEN38_M3_AIR) $(QWEN38_M3_DELTANET_AIR) $(QWEN38_M3_LAYER_AIR) \
	  $(QWEN38_M3_LAYER_Q8_AIR) $(QWEN38_M3_ATTENTION_AIR) \
	  $(QWEN38_M3_GLOBAL_AIR) $(QWEN38_M3_PREFILL_AIR) \
	  $(QWEN38_M3_METALLIB) \
	  $(QWEN38_M3_RUNTIME_OBJECT) $(QWEN38_M3_DECODE_OBJECT) \
	  $(QWEN38_TOKENIZER_OBJECT) $(QWEN38_SAMPLER_OBJECT) \
	  $(QWEN38_M3_GENERATE_OBJECT) $(QWEN38_M3_GENERATE) \
	  $(QWEN38_M3_CHAT_OBJECT) $(QWEN38_M3_CHAT)
	rm -rf $(BUILD_DIR)
