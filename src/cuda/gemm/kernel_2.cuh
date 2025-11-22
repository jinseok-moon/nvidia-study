#pragma once
#include <cuda_runtime.h>
#include <cassert>

template <int BM, int BN, int BK, int TM>
__global__ void gemm_gpu_2_sram_1d_tiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    int bkRow = blockIdx.y;
    int bkCol = blockIdx.x;

    A += K * BM * bkRow;
    B += BN * bkCol;
    C += N * BM * bkRow + BN * bkCol;

    assert(BK == TM && "BK Should be same with TM");

    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    int tRow = threadIdx.x / BN;
    int tCol = threadIdx.x % BN;

    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;

    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;

    float sum[TM] = {
        0.0f,
    };

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        sA[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        sB[innerRowB * BN + innerColB] = B[innerRowB * N + innerRowB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++)
        {
            float _b = sB[dotIdx * BN + tCol];
            for (int resIdx = 0; resIdx < TM; resIdx++)
            {
                sum[resIdx] += sA[(tRow * TM + resIdx) * BK + dotIdx] * _b;
            }
        }
        __syncthreads();
    }

    for (int resIdx = 0; resIdx < TM; resIdx++)
    {
        C[(tRow * TM + resIdx) * BN + tCol] = alpha * sum[resIdx] + beta * C[(tRow * TM + resIdx) * BN + tCol];
    }
}
