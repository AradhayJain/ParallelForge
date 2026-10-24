#pragma once

#include <parallelforge/benchmark/benchmark_runner.hpp>

namespace parallelforge::benchmarks {

void run_bench_matrix(benchmark::BenchmarkRunner& runner);
void run_bench_convolution(benchmark::BenchmarkRunner& runner);
void run_bench_vector(benchmark::BenchmarkRunner& runner);
void run_bench_memory(benchmark::BenchmarkRunner& runner);

} // namespace parallelforge::benchmarks
