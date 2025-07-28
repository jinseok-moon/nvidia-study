#pragma once
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                << cudaGetErrorString(err) << std::endl;                    \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

__global__ void warm_up_gpu() {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

void warmup(int times = 10) {
  for (int i = 0; i < times; i++) {
    warm_up_gpu<<<1, 1>>>();
  }
  cudaDeviceSynchronize();
}

class Profiler {
public:
  Profiler() {
    cudaEventCreate(&evt_start);
    cudaEventCreate(&evt_end);
  }
  ~Profiler() {
    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_end);
  }

  void print_mean_time(std::string name) {
    auto mean_time = std::accumulate(time_vec.begin(), time_vec.end(), 0.0) /
                     time_vec.size();
    std::cout << name << " mean time: " << mean_time << " ms" << std::endl;
  }

  void clear_time_vec() { time_vec.clear(); }

  // Timing wrapper function using std::function
  template <typename Func, typename... Args>
  float time_function(const std::string &name, Func &&func, Args &&...args) {
    cudaEventRecord(evt_start, 0);

    // Execute the function and capture the result (if any)
    func(std::forward<Args>(args)...);
    cudaEventRecord(evt_end, 0);
    cudaEventSynchronize(evt_end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, evt_start, evt_end);
    time_vec.push_back(elapsed_time);
    return elapsed_time;
  }

private:
  cudaEvent_t evt_start, evt_end;
  std::vector<float> time_vec;
};

static Profiler profiler;