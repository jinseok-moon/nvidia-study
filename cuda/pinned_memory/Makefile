# INCLUDE_DIR = ../utils

all:
	nvcc -o a.out -O3 pinned_memory.cu 

clean:
	rm -f a.out

run: all
	./a.out

.PHONY: all clean run