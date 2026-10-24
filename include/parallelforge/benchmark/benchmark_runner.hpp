#pragma once

#include <parallelforge/benchmark/benchmark_result.hpp>
#include <parallelforge/diagnostics/system_info.hpp>

#include <functional>
#include <vector>
#include <string>

namespace parallelforge::benchmark {

struct BenchmarkOptions {
    std::size_t warmup{3};
    std::size_t repetitions{10};
    unsigned int threads{0};
    std::size_t matrix_size{512};
    std::size_t vector_size{10'000'000};
    std::size_t image_width{1920};
    std::size_t image_height{1080};
    std::size_t memory_size_mib{64};
    bool emit_raw_samples{false};
};

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(BenchmarkOptions options = {});

    /**
     * @brief Measures a custom benchmark kernel with pre-validation and warmup.
     */
    BenchmarkResult measure(const std::string& suite,
                            const std::string& test_name,
                            const std::string& algorithm,
                            std::function<void()> kernel,
                            std::function<bool()> validator,
                            unsigned int workers = 1,
                            double baseline_mean_ms = 0.0,
                            double total_ops = 0.0,
                            std::size_t workload_size = 0);

    // Standard suites
    std::vector<BenchmarkResult> run_matrix_suite(std::size_t size = 512, unsigned int threads = 0);
    std::vector<BenchmarkResult> run_convolution_suite(std::size_t width = 1920, std::size_t height = 1080, unsigned int threads = 0);
    std::vector<BenchmarkResult> run_vector_suite(std::size_t size = 10'000'000, unsigned int threads = 0);
    std::vector<BenchmarkResult> run_memory_suite(std::size_t size_mib = 64);
    std::vector<BenchmarkResult> run_sort_suite(std::size_t size = 1'000'000, unsigned int threads = 0);
    std::vector<BenchmarkResult> run_all();

    [[nodiscard]] const std::vector<BenchmarkResult>& results() const noexcept { return results_; }
    void clear_results() noexcept { results_.clear(); }

private:
    BenchmarkOptions options_;
    diagnostics::SystemInfo sys_info_;
    std::vector<BenchmarkResult> results_;

    std::string get_current_timestamp() const;
};

} // namespace parallelforge::benchmark
