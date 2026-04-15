PROJECT_NAME = nn
VERSION = 11.45.14

SRC_DIR = .
BUILD_DIR = build
INCLUDE_DIR = .

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS =

C_SOURCES = $(wildcard $(SRC_DIR)/*.c)
C_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))

TARGET = $(BUILD_DIR)/$(PROJECT_NAME)

.DEFAULT_GOAL := all

.PHONY: all build clean

all: prepare build

prepare:
	@mkdir -p $(BUILD_DIR)

build: $(TARGET)

$(TARGET): $(C_OBJECTS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@
	@echo "Build completed: $(TARGET)"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@
