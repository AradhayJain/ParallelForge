#include <parallelforge/benchmark/benchmark_runner.hpp>
#include <parallelforge/algorithms/matrix_multiply.hpp>
#include <parallelforge/core/do_not_optimize.hpp>

namespace parallelforge::benchmarks {

void run_bench_matrix(benchmark::BenchmarkRunner& runner) {
    runner.run_matrix_suite(512, 0);
}

} // namespace parallelforge::benchmarks
