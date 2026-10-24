#include <parallelforge/benchmark/benchmark_runner.hpp>
#include <parallelforge/algorithms/vector_ops.hpp>
#include <parallelforge/core/do_not_optimize.hpp>

namespace parallelforge::benchmarks {

void run_bench_vector(benchmark::BenchmarkRunner& runner) {
    runner.run_vector_suite(10'000'000, 0);
}

} // namespace parallelforge::benchmarks
