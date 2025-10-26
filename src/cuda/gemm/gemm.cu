#include "../utils/utils.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define BLOCKSIZE 32
constexpr float EPS = 1e-5;
#define TIMES 1
#define SIZE 1024

bool check_result(float *ref, float *result, int size, float threshold = 0.01) {
  for (int i = 0; i < size; i++) {
    float diff = abs(result[i] - ref[i]);
    float ref_abs = abs(ref[i]);

    // Use relative error if reference value is not too small
    float relative_error = diff / (ref_abs + EPS);
    if (relative_error > threshold) {
      std::cout << "result[" << i << "] = " << result[i] << " != ref[" << i
                << "] = " << ref[i]
                << " (relative error: " << relative_error * 100 << "%)"
                << std::endl;
      return false;
    }
  }
  return true;
}

// Naive GEMM implementation in CPU
void gemm_cpu(float *A, float *B, float *C, int M, int N, int K) {
#pragma omp parallel for collapse(2)
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0;
      for (int k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
}

// Naive GEMM implementation in GPU
__global__ void gemm_gpu_0_bad(float *A, float *B, float *C, int M, int N,
                               int K) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (n >= N || m >= M)
    return;

  float sum = 0;
  for (int k = 0; k < K; k++) {
    sum += A[m * K + k] * B[k * N + n];
  }

  C[m * N + n] = sum;
}

__global__ void gemm_gpu_0_good(float *A, float *B, float *C, int M, int N,
                                int K) {
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n >= N || m >= M)
    return;

  float sum = 0;
  for (int k = 0; k < K; k++) {
    sum += A[m * K + k] * B[k * N + n];
  }

  C[m * N + n] = sum;
}

// Naive implementation, with block loop
// Need lots of register
__global__ __maxnreg__(64) void gemm_gpu_0(float *A, float *B, float *C, int M,
                                           int N, int K) {
  A += blockIdx.y * blockDim.y * K;                           // (bM,0)
  B += blockIdx.x * blockDim.x;                               // (0,bN)
  C += blockIdx.y * blockDim.y * N + blockIdx.x * blockDim.x; // (bM,bN)

  float sum = 0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++) {
      sum += A[threadIdx.y * K + dotIdx] * B[dotIdx * N + threadIdx.x];
    }
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;
  }
  C[threadIdx.y * N + threadIdx.x] = sum;
}

__global__ void gemm_gpu_1(float *A, float *B, float *C, int M, int N, int K) {
  A += blockIdx.y * blockDim.y * K;                           // (bM,0)
  B += blockIdx.x * blockDim.x;                               // (0,bN)
  C += blockIdx.y * blockDim.y * N + blockIdx.x * blockDim.x; // (bM,bN)

  __shared__ float A_shared[BLOCKSIZE * BLOCKSIZE];
  __shared__ float B_shared[BLOCKSIZE * BLOCKSIZE];

  float sum = 0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    A_shared[threadIdx.y * blockDim.x + threadIdx.x] =
        A[threadIdx.y * K + threadIdx.x];
    B_shared[threadIdx.y * blockDim.x + threadIdx.x] =
        B[threadIdx.y * N + threadIdx.x];
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++) {
      sum += A_shared[threadIdx.y * BLOCKSIZE + dotIdx] *
             B_shared[dotIdx * BLOCKSIZE + threadIdx.x];
    }

    __syncthreads();
  }
  C[threadIdx.y * N + threadIdx.x] = sum;
}

void launch_gpu_kernel_cublas(float *A, float *B, float *C, int M, int N, int K, cublasHandle_t handle) {
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N);
}

void launch_gpu_kernel_0_bad(float *A, float *B, float *C, int M, int N,
                             int K) {
  dim3 block(BLOCKSIZE, BLOCKSIZE, 1);
  dim3 grid((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  gemm_gpu_0_bad<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_gpu_kernel_0_good(float *A, float *B, float *C, int M, int N,
                              int K) {
  dim3 block(BLOCKSIZE, BLOCKSIZE, 1);
  dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);
  gemm_gpu_0_good<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_gpu_kernel_0(float *A, float *B, float *C, int M, int N, int K) {
  dim3 block(BLOCKSIZE, BLOCKSIZE, 1);
  dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);
  gemm_gpu_0<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_gpu_kernel_1(float *A, float *B, float *C, int M, int N, int K) {
  dim3 block(BLOCKSIZE, BLOCKSIZE, 1);
  dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);
  gemm_gpu_1<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char *argv[]) {
  // Default values
  int M = SIZE, N = SIZE, K = SIZE;

  // Parse command line arguments
  if (argc >= 2)
    M = std::atoi(argv[1]);
  if (argc >= 3)
    N = std::atoi(argv[2]);
  if (argc >= 4)
    K = std::atoi(argv[3]);

  std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K
            << std::endl;

  // Initialize matrices
  float *A = (float *)malloc(M * K * sizeof(float));
  float *B = (float *)malloc(K * N * sizeof(float));
  float *C = (float *)malloc(M * N * sizeof(float));
  float *dev_A = nullptr;
  float *dev_B = nullptr;
  float *dev_C = nullptr;
  float *host_C = nullptr;
  cudaMalloc((void **)&dev_A, M * K * sizeof(float));
  cudaMalloc((void **)&dev_B, K * N * sizeof(float));
  cudaMalloc((void **)&dev_C, M * N * sizeof(float));
  host_C = (float *)malloc(M * N * sizeof(float));

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  for (int i = 0; i < M * K; i++)
    A[i] = dis(gen);
  for (int i = 0; i < K * N; i++)
    B[i] = dis(gen);
  for (int i = 0; i < M * N; i++)
    C[i] = 0;

  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaMemcpy(dev_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
  warmup(1);
  launch_gpu_kernel_cublas(dev_A, dev_B, dev_C, M, N, K, handle);

  
  // Run CPU GEMM using timing wrapper
  profiler.time_function("CPU GEMM", 1, gemm_cpu,
                           A, B, C, M, N, K);

  // Run CPU GEMM using timing wrapper
  profiler.time_function("CUBLAS GEMM", TIMES, launch_gpu_kernel_cublas,
                           dev_A, dev_B, dev_C, M, N, K, handle);

  cudaMemcpy(host_C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  check_result(C, host_C, M * N);
  cudaMemset(dev_C, 0, M * N * sizeof(float));

  profiler.time_function("GPU GEMM 0 BAD CASE", TIMES, launch_gpu_kernel_0_bad,
                           dev_A, dev_B, dev_C, M, N, K);

  cudaMemcpy(host_C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  check_result(C, host_C, M * N);
  cudaMemset(dev_C, 0, M * N * sizeof(float));
  profiler.time_function("GPU GEMM 0 GOOD CASE", TIMES, launch_gpu_kernel_0_good,
                           dev_A, dev_B, dev_C, M, N, K);

  cudaMemcpy(host_C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  check_result(C, host_C, M * N);
  cudaMemset(dev_C, 0, M * N * sizeof(float));

  profiler.time_function("GPU GEMM 0", TIMES, launch_gpu_kernel_0, dev_A, dev_B,
                           dev_C, M, N, K);
  cudaMemcpy(host_C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  check_result(C, host_C, M * N);
  cudaMemset(dev_C, 0, M * N * sizeof(float));

  profiler.time_function("GPU GEMM 1", TIMES, launch_gpu_kernel_1, dev_A, dev_B,
                           dev_C, M, N, K);
  cudaMemcpy(host_C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  check_result(C, host_C, M * N);
  cudaMemset(dev_C, 0, M * N * sizeof(float));

  return 0;
}