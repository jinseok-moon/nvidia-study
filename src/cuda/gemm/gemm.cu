#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <utils.h>

static LatencyProfiler profiler;

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

template <int BLOCKSIZE>
__global__ void gemm_gpu_1(float *A, float *B, float *C, int M, int N, int K) {
  int cRow = blockIdx.y;
  int cCol = blockIdx.x;

  A += cRow * BLOCKSIZE * K;                    // (bM,0)
  B += cCol * BLOCKSIZE;                        // (0,bN)
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // (bM,bN)

  __shared__ float A_shared[BLOCKSIZE * BLOCKSIZE];
  __shared__ float B_shared[BLOCKSIZE * BLOCKSIZE];

  float sum = 0;
  int threadRow = threadIdx.x / BLOCKSIZE;
  int threadCol = threadIdx.x % BLOCKSIZE;

  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    A_shared[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    B_shared[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++) {
      sum += A_shared[threadRow * BLOCKSIZE + dotIdx] *
             B_shared[dotIdx * BLOCKSIZE + threadCol];
    }

    __syncthreads();
  }
  C[threadRow * N + threadCol] = sum;
}

template <int BM, int BN, int BK, int TM>
__global__ void gemm_gpu_2(float *A, float *B, float *C, int M, int N, int K) {

  int cRow = blockIdx.y;
  int cCol = blockIdx.x;
  A += cRow * BM * K;             // (bM,0)
  B += cCol * BN;                 // (0,bN)
  C += cRow * BM * N + cCol * BN; // (bM,bN)

  __shared__ float A_shared[BM * BK];
  __shared__ float B_shared[BK * BN];

  int threadRow = threadIdx.x / BN;
  int threadCol = threadIdx.x % BN;

  int innerRowA = threadIdx.x / BK;
  int innerColA = threadIdx.x % BK;

  int innerRowB = threadIdx.x / BN;
  int innerColB = threadIdx.x % BN;

  float threadResults[TM] = {0.0};
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    A_shared[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    B_shared[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
      float _b = B_shared[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            A_shared[(threadRow * TM + resIdx) * BK + dotIdx] * _b;
      }
    }
    __syncthreads();
  }

  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
  }
}

void launch_gpu_kernel_cublas(float *A, float *B, float *C, int M, int N, int K,
                              cublasHandle_t handle) {
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N);
}

template <int BLOCKSIZE>
void launch_gpu_kernel_0_bad(float *A, float *B, float *C, int M, int N,
                             int K) {
  dim3 block(BLOCKSIZE, BLOCKSIZE, 1);
  dim3 grid((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  gemm_gpu_0_bad<<<grid, block>>>(A, B, C, M, N, K);
}

template <int BLOCKSIZE>
void launch_gpu_kernel_0_good(float *A, float *B, float *C, int M, int N,
                              int K) {
  dim3 block(BLOCKSIZE, BLOCKSIZE, 1);
  dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);
  gemm_gpu_0_good<<<grid, block>>>(A, B, C, M, N, K);
}

template <int BLOCKSIZE>
void launch_gpu_kernel_1(float *A, float *B, float *C, int M, int N, int K) {
  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);
  gemm_gpu_1<BLOCKSIZE><<<grid, block>>>(A, B, C, M, N, K);
}

template <int BM, int BN, int BK, int TM>
void launch_gpu_kernel_2(float *A, float *B, float *C, int M, int N, int K) {
  dim3 block((BM * BN) / TM);
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  gemm_gpu_2<BM, BN, BK, TM><<<grid, block>>>(A, B, C, M, N, K);
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
  launch_gpu_kernel_cublas(dev_A, dev_B, dev_C, M, N, K, handle);

  profiler.benchmark_kernel("CPU_GEMM", [&]() { gemm_cpu(A, B, C, M, N, K); });

  profiler.benchmark_kernel("CUBLAS GEMM", [&]() {
    launch_gpu_kernel_cublas(dev_A, dev_B, dev_C, M, N, K, handle);
  });

  cudaMemcpy(host_C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  check_result(C, host_C, M * N);
  cudaMemset(dev_C, 0, M * N * sizeof(float));

  profiler.benchmark_kernel("GPU GEMM 0 BAD CASE", [&]() {
    launch_gpu_kernel_0_bad<32>(dev_A, dev_B, dev_C, M, N, K);
  });

  cudaMemcpy(host_C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  check_result(C, host_C, M * N);
  cudaMemset(dev_C, 0, M * N * sizeof(float));
  profiler.benchmark_kernel("GPU GEMM 0 GOOD CASE", [&]() {
    launch_gpu_kernel_0_good<32>(dev_A, dev_B, dev_C, M, N, K);
  });

  cudaMemcpy(host_C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  check_result(C, host_C, M * N);
  cudaMemset(dev_C, 0, M * N * sizeof(float));

  profiler.benchmark_kernel("GPU GEMM 1", [&]() {
    launch_gpu_kernel_1<32>(dev_A, dev_B, dev_C, M, N, K);
  });
  cudaMemcpy(host_C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  check_result(C, host_C, M * N);
  cudaMemset(dev_C, 0, M * N * sizeof(float));

  profiler.benchmark_kernel("GPU GEMM 2", [&]() {
    launch_gpu_kernel_2<64, 64, 8, 8>(dev_A, dev_B, dev_C, M, N, K);
  });
  cudaMemcpy(host_C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  check_result(C, host_C, M * N);
  cudaMemset(dev_C, 0, M * N * sizeof(float));
  return 0;
}