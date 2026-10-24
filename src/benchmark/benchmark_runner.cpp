#include <parallelforge/benchmark/benchmark_runner.hpp>
#include <parallelforge/algorithms/matrix_multiply.hpp>
#include <parallelforge/algorithms/convolution.hpp>
#include <parallelforge/algorithms/vector_ops.hpp>
#include <parallelforge/algorithms/parallel_sort.hpp>
#include <parallelforge/memory/memory_benchmarks.hpp>
#include <parallelforge/memory/false_sharing.hpp>
#include <parallelforge/core/do_not_optimize.hpp>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace parallelforge::benchmark {

BenchmarkRunner::BenchmarkRunner(BenchmarkOptions options)
    : options_(options), sys_info_(diagnostics::SystemInfo::collect()) {}

std::string BenchmarkRunner::get_current_timestamp() const {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&tt, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

BenchmarkResult BenchmarkRunner::measure(const std::string& suite,
                                        const std::string& test_name,
                                        const std::string& algorithm,
                                        std::function<void()> kernel,
                                        std::function<bool()> validator,
                                        unsigned int workers,
                                        double baseline_mean_ms,
                                        double total_ops,
                                        std::size_t workload_size) {
    BenchmarkResult res;
    res.suite_name = suite;
    res.test_name = test_name;
    res.algorithm_name = algorithm;
    res.workload_size = workload_size;
    res.threads = workers;
    res.warmup_iterations = options_.warmup;
    res.repetitions = options_.repetitions;

    // 1. Correctness Validation
    if (validator) {
        try {
            res.passed_verification = validator();
            res.verification_details = res.passed_verification ? "Passed" : "Verification check failed";
        } catch (const std::exception& ex) {
            res.passed_verification = false;
            res.verification_details = std::string("Exception during verification: ") + ex.what();
        }
    } else {
        res.passed_verification = true;
        res.verification_details = "No validator supplied";
    }

    // 2. Warm-up Iterations
    for (std::size_t i = 0; i < options_.warmup; ++i) {
        kernel();
        core::clobber_memory();
    }

    // 3. Measured Repetitions
    std::vector<double> samples;
    samples.reserve(options_.repetitions);

    for (std::size_t i = 0; i < options_.repetitions; ++i) {
        core::clobber_memory();
        const auto start = std::chrono::steady_clock::now();
        kernel();
        const auto end = std::chrono::steady_clock::now();
        core::clobber_memory();

        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        samples.push_back(ms);
    }

    res.sample_times_ms = samples;
    res.metrics = MetricCalculator::calculate(samples, baseline_mean_ms, workers, total_ops);

    // Environment metadata
    res.os_name = sys_info_.os_name + " " + sys_info_.os_release;
    res.architecture = sys_info_.architecture;
    res.compiler = sys_info_.compiler_name + " " + sys_info_.compiler_version;
    res.cpu_model = sys_info_.cpu.model_name;
    res.timestamp = get_current_timestamp();

    results_.push_back(res);
    return res;
}

std::vector<BenchmarkResult> BenchmarkRunner::run_matrix_suite(std::size_t size, unsigned int threads) {
    std::vector<BenchmarkResult> suite_results;
    const unsigned int resolved_threads = threads > 0 ? threads : sys_info_.cpu.logical_cores;
    const double total_ops = 2.0 * static_cast<double>(size) * static_cast<double>(size) * static_cast<double>(size);

    core::Matrix a = core::Matrix::random(size, size, 0.0f, 1.0f, 42);
    core::Matrix b = core::Matrix::random(size, size, 0.0f, 1.0f, 84);
    core::Matrix ref = algorithms::multiply_serial_ikj(a, b);

    // Baseline: Serial ikj
    core::Matrix out_serial;
    auto res_serial = measure(
        "matrix", "multiply_" + std::to_string(size), "serial_ikj",
        [&]() { out_serial = algorithms::multiply_serial_ikj(a, b); core::do_not_optimize(out_serial.data()); },
        [&]() { return out_serial.approx_equal(ref); },
        1, 0.0, total_ops, size
    );
    res_serial.dimension_m = size;
    res_serial.dimension_k = size;
    res_serial.dimension_n = size;
    suite_results.push_back(res_serial);
    const double baseline_mean = res_serial.metrics.mean_ms;

    // Blocked Serial
    core::Matrix out_blocked;
    auto res_blocked = measure(
        "matrix", "multiply_" + std::to_string(size), "blocked_serial",
        [&]() { out_blocked = algorithms::multiply_blocked(a, b, core::DEFAULT_TILE_SIZE); core::do_not_optimize(out_blocked.data()); },
        [&]() { return out_blocked.approx_equal(ref); },
        1, baseline_mean, total_ops, size
    );
    res_blocked.dimension_m = size;
    res_blocked.dimension_k = size;
    res_blocked.dimension_n = size;
    res_blocked.tile_size = core::DEFAULT_TILE_SIZE;
    suite_results.push_back(res_blocked);

    // OpenMP Parallel Blocked
    core::Matrix out_omp;
    auto res_omp = measure(
        "matrix", "multiply_" + std::to_string(size), "openmp_blocked",
        [&]() { out_omp = algorithms::multiply_openmp(a, b, core::DEFAULT_TILE_SIZE, resolved_threads); core::do_not_optimize(out_omp.data()); },
        [&]() { return out_omp.approx_equal(ref); },
        resolved_threads, baseline_mean, total_ops, size
    );
    res_omp.dimension_m = size;
    res_omp.dimension_k = size;
    res_omp.dimension_n = size;
    res_omp.tile_size = core::DEFAULT_TILE_SIZE;
    suite_results.push_back(res_omp);

    // Threads Parallel Blocked
    core::Matrix out_th;
    auto res_th = measure(
        "matrix", "multiply_" + std::to_string(size), "threads_blocked",
        [&]() { out_th = algorithms::multiply_threads(a, b, core::DEFAULT_TILE_SIZE, resolved_threads); core::do_not_optimize(out_th.data()); },
        [&]() { return out_th.approx_equal(ref); },
        resolved_threads, baseline_mean, total_ops, size
    );
    res_th.dimension_m = size;
    res_th.dimension_k = size;
    res_th.dimension_n = size;
    res_th.tile_size = core::DEFAULT_TILE_SIZE;
    suite_results.push_back(res_th);

    // SIMD Blocked
    core::Matrix out_simd;
    auto res_simd = measure(
        "matrix", "multiply_" + std::to_string(size), "simd_blocked",
        [&]() { out_simd = algorithms::multiply_simd(a, b, core::DEFAULT_TILE_SIZE); core::do_not_optimize(out_simd.data()); },
        [&]() { return out_simd.approx_equal(ref); },
        1, baseline_mean, total_ops, size
    );
    res_simd.dimension_m = size;
    res_simd.dimension_k = size;
    res_simd.dimension_n = size;
    res_simd.tile_size = core::DEFAULT_TILE_SIZE;
    suite_results.push_back(res_simd);

    return suite_results;
}

std::vector<BenchmarkResult> BenchmarkRunner::run_convolution_suite(std::size_t width, std::size_t height, unsigned int threads) {
    std::vector<BenchmarkResult> suite_results;
    const unsigned int resolved_threads = threads > 0 ? threads : sys_info_.cpu.logical_cores;
    const auto kernel = algorithms::Kernels::gaussian_blur_3x3();
    const double total_ops = 2.0 * static_cast<double>(width * height * kernel.size());

    core::Matrix img = core::Matrix::random(height, width, 0.0f, 255.0f, 99);
    core::Matrix ref = algorithms::convolve_serial(img, kernel, algorithms::BorderMode::ClampToEdge);

    // Serial
    core::Matrix out_serial;
    auto res_serial = measure(
        "convolution", "convolve_gaussian_" + std::to_string(width) + "x" + std::to_string(height), "serial",
        [&]() { out_serial = algorithms::convolve_serial(img, kernel, algorithms::BorderMode::ClampToEdge); core::do_not_optimize(out_serial.data()); },
        [&]() { return out_serial.approx_equal(ref); },
        1, 0.0, total_ops, width * height
    );
    suite_results.push_back(res_serial);
    const double baseline_mean = res_serial.metrics.mean_ms;

    // OpenMP
    core::Matrix out_omp;
    auto res_omp = measure(
        "convolution", "convolve_gaussian_" + std::to_string(width) + "x" + std::to_string(height), "openmp",
        [&]() { out_omp = algorithms::convolve_openmp(img, kernel, algorithms::BorderMode::ClampToEdge, resolved_threads); core::do_not_optimize(out_omp.data()); },
        [&]() { return out_omp.approx_equal(ref); },
        resolved_threads, baseline_mean, total_ops, width * height
    );
    suite_results.push_back(res_omp);

    // Threads
    core::Matrix out_th;
    auto res_th = measure(
        "convolution", "convolve_gaussian_" + std::to_string(width) + "x" + std::to_string(height), "threads",
        [&]() { out_th = algorithms::convolve_threads(img, kernel, algorithms::BorderMode::ClampToEdge, resolved_threads); core::do_not_optimize(out_th.data()); },
        [&]() { return out_th.approx_equal(ref); },
        resolved_threads, baseline_mean, total_ops, width * height
    );
    suite_results.push_back(res_th);

    return suite_results;
}

std::vector<BenchmarkResult> BenchmarkRunner::run_vector_suite(std::size_t size, unsigned int threads) {
    std::vector<BenchmarkResult> suite_results;
    const unsigned int resolved_threads = threads > 0 ? threads : sys_info_.cpu.logical_cores;

    std::vector<float> a(size, 1.5f);
    std::vector<float> b(size, 2.0f);
    const double dot_ops = 2.0 * static_cast<double>(size);
    const float ref_dot = algorithms::dot_product_serial(a, b);

    // Dot product Serial
    float dot_serial = 0.0f;
    auto res_dot_serial = measure(
        "vector", "dot_product_" + std::to_string(size), "serial",
        [&]() { dot_serial = algorithms::dot_product_serial(a, b); core::do_not_optimize(dot_serial); },
        [&]() { return algorithms::float_approx_equal(dot_serial, ref_dot); },
        1, 0.0, dot_ops, size
    );
    suite_results.push_back(res_dot_serial);
    const double baseline_dot = res_dot_serial.metrics.mean_ms;

    // Dot product OpenMP
    float dot_omp = 0.0f;
    auto res_dot_omp = measure(
        "vector", "dot_product_" + std::to_string(size), "openmp",
        [&]() { dot_omp = algorithms::dot_product_openmp(a, b, resolved_threads); core::do_not_optimize(dot_omp); },
        [&]() { return algorithms::float_approx_equal(dot_omp, ref_dot); },
        resolved_threads, baseline_dot, dot_ops, size
    );
    suite_results.push_back(res_dot_omp);

    // Dot product Threads
    float dot_th = 0.0f;
    auto res_dot_th = measure(
        "vector", "dot_product_" + std::to_string(size), "threads",
        [&]() { dot_th = algorithms::dot_product_threads(a, b, resolved_threads); core::do_not_optimize(dot_th); },
        [&]() { return algorithms::float_approx_equal(dot_th, ref_dot); },
        resolved_threads, baseline_dot, dot_ops, size
    );
    suite_results.push_back(res_dot_th);

    // Dot product SIMD
    float dot_simd = 0.0f;
    auto res_dot_simd = measure(
        "vector", "dot_product_" + std::to_string(size), "simd",
        [&]() { dot_simd = algorithms::dot_product_simd(a, b); core::do_not_optimize(dot_simd); },
        [&]() { return algorithms::float_approx_equal(dot_simd, ref_dot); },
        1, baseline_dot, dot_ops, size
    );
    suite_results.push_back(res_dot_simd);

    return suite_results;
}

std::vector<BenchmarkResult> BenchmarkRunner::run_sort_suite(std::size_t size, unsigned int threads) {
    std::vector<BenchmarkResult> suite_results;
    const unsigned int resolved_threads = threads > 0 ? threads : sys_info_.cpu.logical_cores;

    std::vector<float> original(size);
    for (std::size_t i = 0; i < size; ++i) {
        original[i] = static_cast<float>((i * 1103515245 + 12345) & 0x7FFFFFFF);
    }

    // Serial std::sort
    std::vector<float> work_serial;
    auto res_serial = measure(
        "sort", "merge_sort_" + std::to_string(size), "serial_std_sort",
        [&]() { work_serial = original; algorithms::sort_serial(work_serial); core::do_not_optimize(work_serial.data()); },
        [&]() { return algorithms::is_sorted(work_serial); },
        1, 0.0, static_cast<double>(size), size
    );
    suite_results.push_back(res_serial);
    const double baseline_mean = res_serial.metrics.mean_ms;

    // Parallel merge sort
    std::vector<float> work_parallel;
    auto res_parallel = measure(
        "sort", "merge_sort_" + std::to_string(size), "parallel_merge_sort",
        [&]() { work_parallel = original; algorithms::sort_parallel(work_parallel); core::do_not_optimize(work_parallel.data()); },
        [&]() { return algorithms::is_sorted(work_parallel); },
        resolved_threads, baseline_mean, static_cast<double>(size), size
    );
    suite_results.push_back(res_parallel);

    return suite_results;
}

std::vector<BenchmarkResult> BenchmarkRunner::run_memory_suite(std::size_t size_mib) {
    std::vector<BenchmarkResult> suite_results;
    const std::size_t size_bytes = size_mib * 1024 * 1024;

    // Sequential
    auto res_seq = measure(
        "memory", "access_sequential_" + std::to_string(size_mib) + "mib", "sequential",
        [&]() { memory::benchmark_sequential_access(size_bytes, 1); },
        [&]() { return true; },
        1, 0.0, static_cast<double>(size_bytes), size_bytes
    );
    suite_results.push_back(res_seq);
    const double baseline_mean = res_seq.metrics.mean_ms;

    // Strided (64 bytes = 16 int32s)
    auto res_stride16 = measure(
        "memory", "access_stride16_" + std::to_string(size_mib) + "mib", "strided_16_elems",
        [&]() { memory::benchmark_strided_access(size_bytes, 16, 1); },
        [&]() { return true; },
        1, baseline_mean, static_cast<double>(size_bytes), size_bytes
    );
    suite_results.push_back(res_stride16);

    // False Sharing vs Padded Counters
    const unsigned int hw_threads = sys_info_.cpu.logical_cores;
    auto res_false_sharing = measure(
        "memory", "false_sharing_" + std::to_string(hw_threads) + "threads", "unpadded_false_sharing",
        [&]() { memory::run_false_sharing_experiment(hw_threads, 5'000'000); },
        [&]() { return true; },
        hw_threads, 0.0, 5'000'000.0 * hw_threads, hw_threads
    );
    suite_results.push_back(res_false_sharing);

    auto res_padded = measure(
        "memory", "padded_cacheline_" + std::to_string(hw_threads) + "threads", "padded_isolated",
        [&]() { memory::run_false_sharing_experiment(hw_threads, 5'000'000); },
        [&]() { return true; },
        hw_threads, res_false_sharing.metrics.mean_ms, 5'000'000.0 * hw_threads, hw_threads
    );
    suite_results.push_back(res_padded);

    return suite_results;
}

std::vector<BenchmarkResult> BenchmarkRunner::run_all() {
    results_.clear();
    run_matrix_suite(options_.matrix_size, options_.threads);
    run_convolution_suite(options_.image_width, options_.image_height, options_.threads);
    run_vector_suite(options_.vector_size, options_.threads);
    run_sort_suite(1'000'000, options_.threads);
    run_memory_suite(options_.memory_size_mib);
    return results_;
}

} // namespace parallelforge::benchmark
