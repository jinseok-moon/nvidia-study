#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <random>
#include "../utils/utils.hpp"

bool check_result(int* ref, int* result, int size) {
  for (int i = 0; i < size; i++) {
    if (result[i] != ref[i]) {
      return false;
    }
  }
  return true;
}

// Naive GEMM implementation in CPU
void gemm_cpu(int* A, int* B, int* C, int M, int N, int K) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

// Naive GEMM implementation in GPU
__global__ void gemm_gpu_0(int* A, int* B, int* C, int M, int N, int K) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= M || y >= N)
    return;

  int sum = 0;
  for (int k = 0; k < K; k++) {
    sum += A[x * K + k] * B[k * N + y];
  }

  C[x * N + y] = sum;
}

void launch_gpu_kernel(int* A, int* B, int* C, int M, int N, int K) {
  int block_size = 32;
  dim3 block(block_size, block_size, 1);
  dim3 grid((M + block_size - 1) / block_size,
            (N + block_size - 1) / block_size);
  gemm_gpu_0<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char* argv[]) {
  // Default values
  int M = 1024, N = 1024, K = 1024;

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
  int* A = (int*)malloc(M * K * sizeof(int));
  int* B = (int*)malloc(K * N * sizeof(int));
  int* C = (int*)malloc(M * N * sizeof(int));
  int* dev_A = nullptr;
  int* dev_B = nullptr;
  int* dev_C = nullptr;
  int* host_C = nullptr;
  cudaMalloc((void**)&dev_A, M * K * sizeof(int));
  cudaMalloc((void**)&dev_B, K * N * sizeof(int));
  cudaMalloc((void**)&dev_C, M * N * sizeof(int));
  host_C = (int*)malloc(M * N * sizeof(int));
  cudaEvent_t evt_start, evt_end;
  CUDA_CHECK(cudaEventCreate(&evt_start));
  CUDA_CHECK(cudaEventCreate(&evt_end));

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 100);

  for (int i = 0; i < M * K; i++)
    A[i] = dis(gen);
  for (int i = 0; i < K * N; i++)
    B[i] = dis(gen);
  for (int i = 0; i < M * N; i++)
    C[i] = 0;

  cudaMemcpy(dev_A, A, M * K * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, K * N * sizeof(int), cudaMemcpyHostToDevice);

  // Run CPU GEMM
  auto start = std::chrono::high_resolution_clock::now();
  gemm_cpu(A, B, C, M, N, K);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "CPU GEMM time: " << duration.count() << " ms" << std::endl;

  // Run GPU GEMM
  CUDA_CHECK(cudaEventRecord(evt_start));
  launch_gpu_kernel(dev_A, dev_B, dev_C, M, N, K);
  CUDA_CHECK(cudaEventRecord(evt_end));
  CUDA_CHECK(cudaEventSynchronize(evt_end));
  float elapsed_time;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, evt_start, evt_end));
  std::cout << "GPU GEMM time: " << elapsed_time << " ms" << std::endl;

  cudaMemcpy(host_C, dev_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
  if (check_result(C, host_C, M * N)) {
    std::cout << "GPU GEMM result is correct" << std::endl;
  } else {
    std::cout << "GPU GEMM result is incorrect" << std::endl;
  }
  return 0;
}