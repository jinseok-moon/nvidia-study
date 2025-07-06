#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_eltadd_eb3e0ac1_01234(void);
void load_eltadd_eb3e0ac1_01234(void);
// tt-linker: eltadd_eb3e0ac1_01234:CUdeviceptr x1, CUdeviceptr x2, CUdeviceptr output, int32_t size:64_warps4xstages1
CUresult eltadd_eb3e0ac1_01234(CUstream stream, CUdeviceptr x1, CUdeviceptr x2, CUdeviceptr output, int32_t size);
