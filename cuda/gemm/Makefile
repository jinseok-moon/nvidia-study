# INCLUDE_DIR = ../utils

all:
	nvcc -o a.out -Xcompiler -O3 -Xcompiler -fopenmp -O3 gemm.cu -lcublas

clean:
	rm -f a.out

run: all
	./a.out

profile:
	sudo /usr/local/cuda/bin/ncu --set full -fo profile_result ./a.out
.PHONY: all clean run profile