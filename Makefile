# Makefile for nn project

PROJECT_NAME = nn
BUILD_DIR = build
INCLUDE_DIR = .
BREW_PREFIX := /opt/homebrew

CC = gcc
ARCH_FLAGS := $(shell file "$(BREW_PREFIX)/lib/libjpeg.dylib" 2>/dev/null | grep -q "arm64" && gcc -dumpmachine 2>/dev/null | grep -q "^x86_64" && echo -arch arm64)
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -O2 $(ARCH_FLAGS)
LDFLAGS = $(ARCH_FLAGS)
LDLIBS = -lm
JPEG_FALLBACK_CFLAGS := $(if $(wildcard $(BREW_PREFIX)/include/jpeglib.h),-I$(BREW_PREFIX)/include,)
JPEG_FALLBACK_LDFLAGS := $(if $(wildcard $(BREW_PREFIX)/lib/libjpeg.dylib),-L$(BREW_PREFIX)/lib -ljpeg,)
JPEG_PKG_CFLAGS := $(shell command -v pkg-config >/dev/null 2>&1 && pkg-config --cflags libjpeg 2>/dev/null)
JPEG_PKG_LDFLAGS := $(shell command -v pkg-config >/dev/null 2>&1 && pkg-config --libs libjpeg 2>/dev/null)
JPEG_CFLAGS ?= $(if $(strip $(JPEG_PKG_CFLAGS)),$(JPEG_PKG_CFLAGS),$(JPEG_FALLBACK_CFLAGS))
JPEG_LDFLAGS ?= $(if $(strip $(JPEG_PKG_LDFLAGS)),$(JPEG_PKG_LDFLAGS),$(JPEG_FALLBACK_LDFLAGS))

SRC_APP = main.c nn_tests.c
SRC_COMMON = resnet18.c nn_conv2d.c nn_fc.c nn_image.c nn_matrix.c nn_layer.c nn_function.c nn_utils.c

APP_OBJECTS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRC_APP))
COMMON_OBJECTS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRC_COMMON))
TEST1_OBJECT = $(BUILD_DIR)/test_resnet18_64x64.o
TEST2_OBJECT = $(BUILD_DIR)/test_resnet18_tiny_imagenet.o

TARGET = $(BUILD_DIR)/$(PROJECT_NAME)
TEST1 = $(BUILD_DIR)/test_resnet18_64x64
TEST2 = $(BUILD_DIR)/test_resnet18_tiny_imagenet

TINY_IMAGENET_ROOT ?= ../tiny-imagenet-200
TINY_VAL_ANNOTATIONS ?= $(TINY_IMAGENET_ROOT)/val/val_annotations.txt
TINY_VAL_IMAGES ?= $(TINY_IMAGENET_ROOT)/val/images
TINY_WNIDS ?= $(TINY_IMAGENET_ROOT)/wnids.txt
LIMIT ?=

.DEFAULT_GOAL := all

.PHONY: all build test tiny tiny-val clean prepare

all: prepare build

prepare:
	@mkdir -p $(BUILD_DIR)

build: $(TARGET)

$(TARGET): $(APP_OBJECTS) $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@
	@echo "Build completed: $(TARGET)"

$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) $(JPEG_CFLAGS) -c $< -o $@

$(TEST1): $(COMMON_OBJECTS) $(TEST1_OBJECT)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@
	@echo "Build completed: $(TEST1)"

$(TEST2): $(COMMON_OBJECTS) $(TEST2_OBJECT)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) $(JPEG_LDFLAGS) $(LDLIBS) -o $@
	@echo "Build completed: $(TEST2)"

test: $(TARGET) $(TEST1)
	./$(TARGET)
	./$(TEST1)

tiny: $(TEST2)

tiny-val: $(TEST2)
	./$(TEST2) "$(TINY_VAL_ANNOTATIONS)" "$(TINY_VAL_IMAGES)" "$(TINY_WNIDS)" "$(LIMIT)"

clean:
	$(RM) $(BUILD_DIR)/*.o $(TARGET) $(TEST1) $(TEST2)
