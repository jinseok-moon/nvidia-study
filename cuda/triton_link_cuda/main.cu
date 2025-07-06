#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

extern "C" {
#include "aot/eltadd.eb3e0ac1_01234.h"
}
#include <iostream>
int main() {
    load_eltadd_eb3e0ac1_01234();

    int* x1;
    int* x2;
    int* output;
    int size = 1024;
    cudaMalloc(&x1, size * sizeof(int));
    cudaMalloc(&x2, size * sizeof(int));
    cudaMalloc(&output, size * sizeof(int));
    cudaDeviceSynchronize();

    eltadd_eb3e0ac1_01234(cudaStreamDefault,CUdeviceptr(x1), CUdeviceptr(x2), CUdeviceptr(output), size);    
    
    unload_eltadd_eb3e0ac1_01234();
}