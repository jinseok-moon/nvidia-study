#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


class LatencyProfiler {
public:
  LatencyProfiler() {
    cudaEventCreate(&evt_start);
    cudaEventCreate(&evt_end);
  }
  ~LatencyProfiler() {
    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_end);
  }

  // Function to perform warmup and benchmark runs
  float benchmark_kernel(const std::string &name,
                         std::function<void()> kernel_func, int warmup_runs,
                         int benchmark_runs) {
    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
      kernel_func();
    }

    // Benchmark runs
    std::vector<float> times;
    for (int i = 0; i < benchmark_runs; ++i) {
      float time = time_function(kernel_func);
      times.push_back(time);
    }

    // Calculate average time
    float avg_time =
        std::accumulate(times.begin(), times.end(), 0.0f) / benchmark_runs;
    std::cout << "[RESULT][AVG LATENCY][WARMUP: " << warmup_runs
              << "][RUN: " << benchmark_runs << "]"
              << " [" << name << "] " << avg_time << " [ms]" << std::endl;
    std::cout << std::endl;
    return avg_time;
  }

  // Timing wrapper function using std::function
  float time_function(std::function<void()> kernel_func) {
    cudaEventRecord(evt_start, 0);
    // Execute the function and capture the result (if any)
    kernel_func();
    cudaEventRecord(evt_end, 0);
    cudaEventSynchronize(evt_end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, evt_start, evt_end);
    return elapsed_time;
  }

private:
  cudaEvent_t evt_start, evt_end;
};

// Lightweight warm-up kernel and helper used by samples
__global__ void warm_up_gpu() {}

inline void warmup(int times) {
  for (int i = 0; i < times; ++i) {
    warm_up_gpu<<<1, 1>>>();
  }
  cudaDeviceSynchronize();
}

// Sample profiler with an interface expected by the CUDA samples
class Profiler {
public:
  Profiler() = default;

  template <typename Func, typename... Args>
  float time_function(const std::string &name, Func &&func, Args &&...args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Execute the callable
    std::forward<Func>(func)(std::forward<Args>(args)...);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    records_[name].push_back(elapsed_ms);
    return elapsed_ms;
  }

  template <typename Func, typename... Args>
  float time_function(const std::string &name, int times, Func &&func,
                      Args &&...args) {
    float last = 0.0f;
    for (int i = 0; i < times; ++i) {
      last = time_function(name, std::forward<Func>(func),
                           std::forward<Args>(args)...);
    }
    return last;
  }

  void print_mean_time(const std::string &name) const {
    auto it = records_.find(name);
    if (it == records_.end() || it->second.empty()) {
      std::cout << name << ": no records" << std::endl;
      return;
    }
    float sum = std::accumulate(it->second.begin(), it->second.end(), 0.0f);
    float mean = sum / static_cast<float>(it->second.size());
    std::cout << name << " mean: " << mean << " ms over "
              << it->second.size() << " runs" << std::endl;
  }

  void clear_time_vec() { records_.clear(); }

private:
  std::unordered_map<std::string, std::vector<float>> records_;
};

// Global profiler instance expected by samples
inline Profiler profiler;