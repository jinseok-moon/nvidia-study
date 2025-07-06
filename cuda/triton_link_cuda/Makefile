# INCLUDE_DIR = ../utils

all:
	export TRITON_ROOT=$(pip show triton | grep Location | cut -d' ' -f2)
	python ${TRITON_ROOT}/triton/tools/compile.py     add_triton.py     -n eltadd     \
	 -o aot/eltadd     --out-name eltadd     -w 4     -ns 1     \
	 -s "*i32, *i32, *i32, i32, 64" \
	 -g "1024, 1024, 1"

	nvcc -o a.out -O0 -g -G -DDEBUG main.cu aot/eltadd.eb3e0ac1_01234.c -lcuda -lcudart -Iaot

clean:
	rm -f a.out

run: all
	./a.out

.PHONY: all clean run