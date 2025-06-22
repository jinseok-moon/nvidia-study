# INCLUDE_DIR = ../utils

all:
	nvcc -o a.out -O3 gemm.cu

clean:
	rm -f a.out

run: all
	./a.out

.PHONY: all clean run