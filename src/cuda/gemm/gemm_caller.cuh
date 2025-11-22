#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.h>
#include "kernel_0.cuh"
#include "kernel_1.cuh"
#include "kernel_2.cuh"


void launch_kernel_cublas(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C,
                              cublasHandle_t handle) {
  cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N);
}

void launch_kernel_0_naive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
  int BLOCKSIZE = 256;
  dim3 block(BLOCKSIZE);
  dim3 grid(ceil_div(M*N, BLOCKSIZE));
  gemm_gpu_0_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

void launch_kernel_0_dram_coalescing(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  int BLOCKSIZE = 256;
  dim3 block(BLOCKSIZE);
  dim3 grid(ceil_div(M*N, BLOCKSIZE));
  gemm_gpu_0_dram_coalescing<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

void launch_kernel_1_sram_caching(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const int BLOCKSIZE = 32;
  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(ceil_div(N, BLOCKSIZE), ceil_div(M, BLOCKSIZE));
  gemm_gpu_1_sram_caching<BLOCKSIZE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

void launch_kernel_2_sram_1d_tiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const int BM = 64;
  const int BN = 64;
  const int BK = 8;
  const int TM = 8;
  dim3 block((BM * BN) / TM);
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  gemm_gpu_2_sram_1d_tiling<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
