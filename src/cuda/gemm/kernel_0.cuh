#pragma once
#include <cuda_runtime.h>


__global__ void gemm_gpu_0_naive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = tid % N;
  int col = tid / N;

  if (row >= M || col >= N) return;

  float sum = 0.0f;
  for (int k=0; k<K; k++)
  {
    sum += A[row*K + k] * B[k*N + col];
  }
  
  C[row*N + col] = alpha*sum + beta*C[row*N + col];
}


__global__ void gemm_gpu_0_dram_coalescing(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = tid / N;
  int col = tid % N;

  if (row >= M || col >= N) return;

  float sum = 0.0f;
  for (int k=0; k<K; k++)
  {
    sum += A[row*K + k] * B[k*N + col];
  }
  
  C[row*N + col] = alpha*sum + beta*C[row*N + col];
}
