#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
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

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

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
                         std::function<void()> kernel_func,
                         int warmup_runs = 10, int benchmark_runs = 20)
  {
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

    // ANSI color codes
    const char *CYAN = "\033[36m";
    const char *BOLD = "\033[1m";
    const char *GREEN = "\033[32m";
    const char *DIM = "\033[2m";
    const char *RESET = "\033[0m";

    std::cout << CYAN << "[BENCHMARK] " << RESET << BOLD << std::right
              << std::setw(30) << name << RESET << " │ " << GREEN << std::fixed
              << std::setprecision(6) << avg_time << " ms" << RESET << DIM
              << " (w:" << warmup_runs << " r:" << benchmark_runs << ")"
              << RESET << std::endl;

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