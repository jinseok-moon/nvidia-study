#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <random>
#include "../utils/utils.hpp"

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
void gemm_gpu(int* A, int* B, int* C, int M, int N, int K) {
  // TODO: Implement GPU GEMM
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

  // Run CPU GEMM
  auto start = std::chrono::high_resolution_clock::now();
  gemm_cpu(A, B, C, M, N, K);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "CPU GEMM time: " << duration.count() << " ms" << std::endl;

  // // Run GPU GEMM
  // start = std::chrono::high_resolution_clock::now();
  // gemm_gpu(A, B, C, M, N, K);
  // end = std::chrono::high_resolution_clock::now();
  return 0;
}