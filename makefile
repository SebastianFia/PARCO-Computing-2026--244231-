.PHONY: all run clean rebuild rerun

CC = gcc
TARGET = build/main
SRC = $(wildcard src/*.c)
OBJ = $(SRC:src/%.c=build/%.o)

OPTIM_FLAGS = -O3
EXTRA_FLAGS =
CORE_FLAGS = -Iinclude -std=c99 -mavx2 -mfma -lrt -lm -fopenmp -Wall -D_POSIX_C_SOURCE=200112L 
CFLAGS = $(CORE_FLAGS) $(OPTIM_FLAGS) $(EXTRA_FLAGS)

CFLAGS += $(USER_CFLAGS)

all: $(TARGET)

rebuild: clean $(TARGET)

rerun: rebuild run

$(TARGET): $(OBJ)
	$(CC) $(OBJ) $(CFLAGS) -o $(TARGET)

build/%.o: src/%.c
	@mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	$(TARGET)

clean:
	rm -rf build
