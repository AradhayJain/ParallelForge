#include <parallelforge/benchmark/benchmark_runner.hpp>
#include <parallelforge/algorithms/convolution.hpp>
#include <parallelforge/core/do_not_optimize.hpp>

namespace parallelforge::benchmarks {

void run_bench_convolution(benchmark::BenchmarkRunner& runner) {
    runner.run_convolution_suite(1920, 1080, 0);
}

} // namespace parallelforge::benchmarks
