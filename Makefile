CC = gcc
NASM = nasm
PYTHON = python3
PYTHON_INCLUDES = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_EXT_SUFFIX = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CFLAGS = -O3 -march=native -mavx2 -msse4.2 -flto -fPIC -Wall -Wextra \
         -Wno-unused-parameter -I include -DUSE_ASM
LDFLAGS = -shared -lpthread -lrt
ASM_FLAGS = -f elf64

BUILD = build
SRC = src
ASM = asm
INCLUDE = include

LIB = $(BUILD)/libfastdb.so
BENCH = $(BUILD)/benchmark
PYEXT = python/fastdb/_fastdb_ext$(PYTHON_EXT_SUFFIX)

ASM_SRC = $(ASM)/fastdb_hash.asm
ASM_OBJ = $(BUILD)/fastdb_hash.o
C_SRC = $(SRC)/fastdb.c
C_OBJ = $(BUILD)/fastdb.o
PYEXT_SRC = python/fastdb/_fastdb_ext.c
PYEXT_OBJ = $(BUILD)/_fastdb_ext.o

.PHONY: all clean bench test lib pyext

all: lib pyext

lib: $(LIB)

pyext: $(PYEXT)

$(BUILD):
	mkdir -p $(BUILD)

$(ASM_OBJ): $(ASM_SRC) | $(BUILD)
	$(NASM) $(ASM_FLAGS) -o $@ $<

$(C_OBJ): $(C_SRC) $(INCLUDE)/fastdb.h | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(LIB): $(C_OBJ) $(ASM_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Python C extension
$(PYEXT_OBJ): $(PYEXT_SRC) $(INCLUDE)/fastdb.h | $(BUILD)
	$(CC) $(CFLAGS) -I$(PYTHON_INCLUDES) -c -o $@ $<

$(PYEXT): $(PYEXT_OBJ) $(C_OBJ) $(ASM_OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $^ -lpthread -lrt

# Benchmark binary (statically linked for speed measurement)
$(BENCH): benchmarks/benchmark.c $(C_SRC) $(ASM_SRC) $(INCLUDE)/fastdb.h | $(BUILD)
	$(NASM) $(ASM_FLAGS) -o $(ASM_OBJ) $(ASM_SRC)
	$(CC) -O3 -march=native -mavx2 -msse4.2 -flto -I include -DUSE_ASM \
		-o $@ benchmarks/benchmark.c $(C_SRC) $(ASM_OBJ) \
		-lpthread -lrt -lm

bench: $(BENCH)
	./$(BENCH)

clean:
	rm -rf $(BUILD) python/fastdb/_fastdb_ext*.so
