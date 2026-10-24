#pragma once

#include <parallelforge/benchmark/metric_calculator.hpp>

#include <string>
#include <vector>
#include <cstddef>
#include <chrono>

namespace parallelforge::benchmark {

struct BenchmarkResult {
    std::string suite_name;
    std::string test_name;
    std::string algorithm_name;
    std::size_t workload_size{0};
    std::size_t dimension_m{0};
    std::size_t dimension_n{0};
    std::size_t dimension_k{0};
    std::size_t tile_size{0};
    unsigned int threads{1};
    std::size_t warmup_iterations{0};
    std::size_t repetitions{0};

    std::vector<double> sample_times_ms;
    AggregateMetrics metrics;

    bool passed_verification{true};
    std::string verification_details{"Passed"};

    // Hardware and Environment Metadata
    std::string os_name;
    std::string architecture;
    std::string compiler;
    std::string cpu_model;
    std::string timestamp;
};

} // namespace parallelforge::benchmark
