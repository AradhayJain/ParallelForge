#include <parallelforge/benchmark/benchmark_runner.hpp>
#include <parallelforge/memory/memory_benchmarks.hpp>
#include <parallelforge/core/do_not_optimize.hpp>

namespace parallelforge::benchmarks {

void run_bench_memory(benchmark::BenchmarkRunner& runner) {
    runner.run_memory_suite(64);
}

} // namespace parallelforge::benchmarks
