#include "gemm_caller.cuh"
#include <utils.h>
#include <random>
static LatencyProfiler profiler;

constexpr float EPS = 1e-5;

bool copy_and_check_result(float *ref, float *dev_result, float *host_result, int size, float threshold = 0.01, bool print_error = false)
{
  CUDA_CHECK(cudaDeviceSynchronize());
  memset(host_result, 0, size * sizeof(float));
  cudaMemcpy(host_result, dev_result, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemset(dev_result, 0, size * sizeof(float));

  for (int i = 0; i < size; i++)
  {
    float diff = abs(host_result[i] - ref[i]);
    float ref_abs = abs(ref[i]);

    float relative_error = diff / (ref_abs + EPS);
    if (relative_error > threshold)
    {
      if (print_error)
      {
        std::cout << "result[" << i << "] = " << host_result[i] << " != ref[" << i
                  << "] = " << ref[i]
                  << " (relative error: " << relative_error * 100 << "%)"
                  << std::endl;
      }
      return false;
    }
  }
  return true;
}

int main(int argc, char *argv[])
{
  // Default values
  int M = 1024;
  int N = 1024;
  int K = 1024;

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
  float *C = (float *)malloc(M * N * sizeof(float)); // Reference result from cuBLAS
  float *dev_A = nullptr;
  float *dev_B = nullptr;
  float *dev_C = nullptr;
  float *host_C = nullptr;
  cudaMalloc((void **)&dev_A, M * K * sizeof(float));
  cudaMalloc((void **)&dev_B, K * N * sizeof(float));
  cudaMalloc((void **)&dev_C, M * N * sizeof(float));
  host_C = (float *)malloc(M * N * sizeof(float));

  cublasHandle_t handle;
  cublasCreate(&handle);

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

  cudaMemcpy(dev_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  profiler.benchmark_kernel("CUBLAS GEMM", [&]()
                            { launch_kernel_cublas(M, N, K, 1.0f, dev_A, dev_B, 0.0f, dev_C, handle); });
  // Make Reference
  cudaMemcpy(C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemset(dev_C, 0, M * N * sizeof(float));

  profiler.benchmark_and_validate_kernel("GPU GEMM 0 NAIVE", [&]()
                                         { launch_kernel_0_naive(M, N, K, 1.0f, dev_A, dev_B, 0.0f, dev_C); }, [&]()
                                         { return copy_and_check_result(C, dev_C, host_C, M * N); });

  profiler.benchmark_and_validate_kernel("GPU GEMM 0 DRAM COALESCING", [&]()
                                         { launch_kernel_0_dram_coalescing(M, N, K, 1.0f, dev_A, dev_B, 0.0f, dev_C); }, [&]()
                                         { return copy_and_check_result(C, dev_C, host_C, M * N); });

  profiler.benchmark_and_validate_kernel("GPU GEMM 1 SRAM CACHING", [&]()
                                         { launch_kernel_1_sram_caching(M, N, K, 1.0f, dev_A, dev_B, 0.0f, dev_C); }, [&]()
                                         { return copy_and_check_result(C, dev_C, host_C, M * N); });

  profiler.benchmark_and_validate_kernel("GPU GEMM 2 SRAM 1D TILING", [&]()
                                         { launch_kernel_1_sram_caching(M, N, K, 1.0f, dev_A, dev_B, 0.0f, dev_C); }, [&]()
                                         { return copy_and_check_result(C, dev_C, host_C, M * N); });

  CUDA_CHECK(cudaDeviceSynchronize());

  return 0;
}