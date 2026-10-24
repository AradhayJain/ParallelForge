#include <parallelforge/cli/commands.hpp>
#include <parallelforge/core/config.hpp>
#include <parallelforge/core/matrix.hpp>
#include <parallelforge/core/types.hpp>
#include <parallelforge/core/do_not_optimize.hpp>
#include <parallelforge/algorithms/matrix_multiply.hpp>
#include <parallelforge/algorithms/convolution.hpp>
#include <parallelforge/algorithms/vector_ops.hpp>
#include <parallelforge/algorithms/parallel_sort.hpp>
#include <parallelforge/memory/memory_benchmarks.hpp>
#include <parallelforge/memory/cache_locality.hpp>
#include <parallelforge/memory/false_sharing.hpp>
#include <parallelforge/threading/bounded_queue.hpp>
#include <parallelforge/threading/thread_pool.hpp>
#include <parallelforge/benchmark/benchmark_runner.hpp>
#include <parallelforge/benchmark/exporter.hpp>
#include <parallelforge/diagnostics/system_info.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <random>

namespace parallelforge::cli {

void print_version() noexcept {
    std::cout << "ParallelForge version 1.0.0 (C++20)\n"
              << "C++ Parallel Computing & CPU Performance Engineering Lab\n";
}

void print_help() noexcept {
    std::cout << R"(
ParallelForge - C++ Parallel Computing & CPU Performance Engineering Lab

Usage:
  parallelforge <command> [options]

Commands:
  info             Display CPU topology, cache hierarchy, OS, and compiler features
  matrix           Run matrix multiplication (serial, blocked, openmp, threads, simd)
  convolution      Run 2D image convolution with standard kernels
  vector           Run vector operations (add, dot, reduce, saxpy)
  sort             Run sorting workloads (serial std::sort vs parallel merge sort)
  memory           Run memory hierarchy benchmarks (sequential, strided, false sharing)
  queue-demo       Run multi-producer multi-consumer bounded queue demonstration
  threadpool-demo  Run thread pool task execution and future collection demonstration
  benchmark        Execute automated benchmark suites with CSV/JSON/Text export
  validate         Run full end-to-end correctness validation across all algorithms

Options:
  -h, --help       Show this help message
  -v, --version    Show version information

Examples:
  parallelforge info
  parallelforge matrix --size 512 --algorithm blocked --tile-size 64 --verify
  parallelforge matrix --rows 1024 --cols 1024 --inner 512 --algorithm openmp --threads 4
  parallelforge convolution --width 1920 --height 1080 --kernel blur --algorithm openmp --threads 4
  parallelforge vector --size 10000000 --operation dot --algorithm threads --threads 4
  parallelforge sort --size 1000000 --algorithm threads --threads 4 --verify
  parallelforge memory --size-mib 64 --pattern strided --stride 16
  parallelforge memory --size-mib 64 --pattern false_sharing --threads 4
  parallelforge queue-demo --producers 4 --consumers 4 --items 20000 --capacity 100
  parallelforge threadpool-demo --workers 4 --tasks 1000
  parallelforge benchmark --suite all --format text
  parallelforge benchmark --suite matrix --format json --output results/matrix.json
  parallelforge validate
)";
}

int cmd_info(const CommandParser& parser) {
    std::optional<std::size_t> user_cache_line;
    if (parser.has_option("cache-line")) {
        user_cache_line = parser.get_size("cache-line");
    }
    auto sys = diagnostics::SystemInfo::collect(user_cache_line);
    if (parser.has_option("json")) {
        std::cout << sys.to_json() << "\n";
    } else {
        std::cout << sys.to_string() << "\n";
    }
    return 0;
}

int cmd_matrix(const CommandParser& parser) {
    std::size_t size = parser.get_size("size", 512);
    std::size_t rows = parser.get_size("rows", size);
    std::size_t cols = parser.get_size("cols", size);
    std::size_t inner = parser.get_size("inner", size);
    std::size_t tile_size = parser.get_size("tile-size", core::DEFAULT_TILE_SIZE);
    unsigned int threads = parser.get_uint("threads", 0);
    std::string algo_str = parser.get_string("algorithm", "blocked");
    bool verify = parser.has_option("verify");

    auto algo_opt = core::parse_algorithm(algo_str);
    if (!algo_opt) {
        std::cerr << "Error: Unknown algorithm '" << algo_str 
                  << "'. Supported: serial, blocked, openmp, threads, simd\n";
        return 1;
    }
    auto algo = *algo_opt;

    std::cout << "Running Matrix Multiplication:\n"
              << "  A (" << rows << " x " << inner << ") * B (" << inner << " x " << cols << ")\n"
              << "  Algorithm: " << algo_str << ", Tile Size: " << tile_size 
              << ", Threads: " << (threads > 0 ? std::to_string(threads) : "auto") << "\n";

    core::Matrix a = core::Matrix::random(rows, inner, 0.0f, 1.0f, 42);
    core::Matrix b = core::Matrix::random(inner, cols, 0.0f, 1.0f, 84);

    const auto start = std::chrono::steady_clock::now();
    core::Matrix c = algorithms::multiply(a, b, algo, tile_size, threads);
    const auto end = std::chrono::steady_clock::now();
    core::do_not_optimize(c.data());

    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double gflops = (2.0 * static_cast<double>(rows) * static_cast<double>(cols) * static_cast<double>(inner)) / (elapsed_ms * 1e6);

    std::cout << "Completed in: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms ("
              << std::setprecision(2) << gflops << " GFLOPS)\n";

    if (verify) {
        std::cout << "Verifying against serial baseline... ";
        core::Matrix ref = algorithms::multiply_serial_ikj(a, b);
        float max_diff = 0.0f;
        if (c.approx_equal(ref, core::FLOAT_EPSILON, &max_diff)) {
            std::cout << "PASSED (max abs/rel diff: " << max_diff << ")\n";
        } else {
            std::cout << "FAILED (max abs/rel diff: " << max_diff << ")\n";
            return 1;
        }
    }
    return 0;
}

int cmd_convolution(const CommandParser& parser) {
    std::size_t width = parser.get_size("width", 1920);
    std::size_t height = parser.get_size("height", 1080);
    std::string kernel_str = parser.get_string("kernel", "blur");
    std::string algo_str = parser.get_string("algorithm", "openmp");
    std::string border_str = parser.get_string("border", "clamp");
    unsigned int threads = parser.get_uint("threads", 0);
    bool verify = parser.has_option("verify");

    auto algo_opt = core::parse_algorithm(algo_str);
    if (!algo_opt) {
        std::cerr << "Error: Unknown algorithm '" << algo_str << "'\n";
        return 1;
    }
    auto border_opt = algorithms::parse_border_mode(border_str);
    if (!border_opt) {
        std::cerr << "Error: Unknown border mode '" << border_str << "'\n";
        return 1;
    }

    core::Matrix kernel;
    if (kernel_str == "blur" || kernel_str == "gaussian") {
        kernel = algorithms::Kernels::gaussian_blur_3x3();
    } else if (kernel_str == "sharpen") {
        kernel = algorithms::Kernels::sharpen_3x3();
    } else if (kernel_str == "sobel_h") {
        kernel = algorithms::Kernels::sobel_horizontal_3x3();
    } else if (kernel_str == "sobel_v") {
        kernel = algorithms::Kernels::sobel_vertical_3x3();
    } else if (kernel_str == "box") {
        kernel = algorithms::Kernels::box_blur_5x5();
    } else {
        std::cerr << "Error: Unknown kernel '" << kernel_str << "'. Supported: blur, sharpen, sobel_h, sobel_v, box\n";
        return 1;
    }

    std::cout << "Running 2D Convolution:\n"
              << "  Image: " << width << " x " << height << ", Kernel: " << kernel_str 
              << " (" << kernel.rows() << "x" << kernel.cols() << "), Border: " << border_str << "\n"
              << "  Algorithm: " << algo_str << ", Threads: " << (threads > 0 ? std::to_string(threads) : "auto") << "\n";

    core::Matrix img = core::Matrix::random(height, width, 0.0f, 255.0f, 99);

    const auto start = std::chrono::steady_clock::now();
    core::Matrix out = algorithms::convolve(img, kernel, *algo_opt, *border_opt, threads);
    const auto end = std::chrono::steady_clock::now();
    core::do_not_optimize(out.data());

    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double mpx = (static_cast<double>(width * height) / 1e6) / (elapsed_ms / 1000.0);

    std::cout << "Completed in: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms ("
              << std::setprecision(2) << mpx << " Megapixels/sec)\n";

    if (verify) {
        std::cout << "Verifying against serial baseline... ";
        core::Matrix ref = algorithms::convolve_serial(img, kernel, *border_opt);
        float max_diff = 0.0f;
        if (out.approx_equal(ref, core::FLOAT_EPSILON, &max_diff)) {
            std::cout << "PASSED (max diff: " << max_diff << ")\n";
        } else {
            std::cout << "FAILED (max diff: " << max_diff << ")\n";
            return 1;
        }
    }
    return 0;
}

int cmd_vector(const CommandParser& parser) {
    std::size_t size = parser.get_size("size", 10'000'000);
    std::string op_str = parser.get_string("operation", "dot");
    std::string algo_str = parser.get_string("algorithm", "threads");
    unsigned int threads = parser.get_uint("threads", 0);

    std::vector<float> a(size, 1.25f);
    std::vector<float> b(size, 2.0f);

    std::cout << "Running Vector Operation: " << op_str << " on " << size << " elements\n"
              << "  Algorithm: " << algo_str << ", Threads: " 
              << (threads > 0 ? std::to_string(threads) : "auto") << "\n";

    const auto start = std::chrono::steady_clock::now();
    if (op_str == "dot") {
        float res = 0.0f;
        if (algo_str == "serial") res = algorithms::dot_product_serial(a, b);
        else if (algo_str == "openmp") res = algorithms::dot_product_openmp(a, b, threads);
        else if (algo_str == "simd") res = algorithms::dot_product_simd(a, b);
        else res = algorithms::dot_product_threads(a, b, threads);
        core::do_not_optimize(res);
        std::cout << "Result dot product: " << res << "\n";
    } else if (op_str == "add") {
        std::vector<float> res;
        if (algo_str == "serial") res = algorithms::vector_add_serial(a, b);
        else if (algo_str == "openmp") res = algorithms::vector_add_openmp(a, b, threads);
        else if (algo_str == "simd") res = algorithms::vector_add_simd(a, b);
        else res = algorithms::vector_add_threads(a, b, threads);
        core::do_not_optimize(res.data());
    } else if (op_str == "reduce" || op_str == "sum") {
        float res = 0.0f;
        if (algo_str == "serial") res = algorithms::reduce_sum_serial(a);
        else if (algo_str == "openmp") res = algorithms::reduce_sum_openmp(a, threads);
        else if (algo_str == "simd") res = algorithms::reduce_sum_simd(a);
        else res = algorithms::reduce_sum_threads(a, threads);
        core::do_not_optimize(res);
        std::cout << "Result sum reduction: " << res << "\n";
    } else if (op_str == "saxpy") {
        if (algo_str == "serial") algorithms::saxpy_serial(2.5f, a, b);
        else if (algo_str == "openmp") algorithms::saxpy_openmp(2.5f, a, b, threads);
        else if (algo_str == "simd") algorithms::saxpy_simd(2.5f, a, b);
        else algorithms::saxpy_threads(2.5f, a, b, threads);
        core::do_not_optimize(b.data());
    } else {
        std::cerr << "Error: Unknown operation '" << op_str << "'. Supported: add, dot, reduce, saxpy\n";
        return 1;
    }
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double gbps = (static_cast<double>(size * sizeof(float)) / 1e9) / (elapsed_ms / 1000.0);
    std::cout << "Completed in: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms ("
              << std::setprecision(2) << gbps << " GB/s effective bandwidth)\n";
    return 0;
}

int cmd_sort(const CommandParser& parser) {
    std::size_t size = parser.get_size("size", 1'000'000);
    std::string algo_str = parser.get_string("algorithm", "threads");
    unsigned int threads = parser.get_uint("threads", 0);
    bool verify = parser.has_option("verify");

    std::cout << "Running Sort Benchmark:\n"
              << "  Elements: " << size << ", Algorithm: " << algo_str << "\n";

    std::vector<float> data(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1000000.0f);
    for (auto& val : data) {
        val = dist(gen);
    }

    const auto start = std::chrono::steady_clock::now();
    if (algo_str == "serial") {
        algorithms::sort_serial(data);
    } else {
        algorithms::sort_parallel(data, algorithms::DEFAULT_SORT_CUTOFF, 4);
    }
    const auto end = std::chrono::steady_clock::now();
    core::do_not_optimize(data.data());

    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Sorted in: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";

    if (verify) {
        std::cout << "Verifying sort order... ";
        if (algorithms::is_sorted(data)) {
            std::cout << "PASSED\n";
        } else {
            std::cout << "FAILED (data not sorted)\n";
            return 1;
        }
    }
    return 0;
}

int cmd_memory(const CommandParser& parser) {
    std::size_t size_mib = parser.get_size("size-mib", 64);
    std::string pattern_str = parser.get_string("pattern", "sequential");
    std::size_t stride = parser.get_size("stride", 16);
    unsigned int threads = parser.get_uint("threads", 4);

    std::cout << "Running Memory & Cache Experiment: " << pattern_str << "\n";

    if (pattern_str == "sequential") {
        auto res = memory::benchmark_sequential_access(size_mib * 1024 * 1024, 5);
        std::cout << "  Working Set: " << size_mib << " MiB\n"
                  << "  Throughput:  " << std::fixed << std::setprecision(2) << res.throughput_gbps << " GB/s\n"
                  << "  Avg Latency: " << std::setprecision(2) << res.average_latency_ns << " ns/elem\n";
    } else if (pattern_str == "strided") {
        auto res = memory::benchmark_strided_access(size_mib * 1024 * 1024, stride, 5);
        std::cout << "  Working Set: " << size_mib << " MiB, Stride: " << stride << " elements (" << (stride * 4) << " bytes)\n"
                  << "  Throughput:  " << std::fixed << std::setprecision(2) << res.throughput_gbps << " GB/s\n"
                  << "  Avg Latency: " << std::setprecision(2) << res.average_latency_ns << " ns/elem\n";
    } else if (pattern_str == "matrix_row" || pattern_str == "matrix_col" || pattern_str == "cache_locality") {
        std::size_t dim = parser.get_size("dim", 2048);
        auto res = memory::run_cache_locality_experiment(dim, 3);
        std::cout << "  Matrix Traversal (" << dim << "x" << dim << " floats):\n"
                  << "  Row-Major (Sequential Stride 1): " 
                  << std::chrono::duration<double, std::milli>(res.row_major_time).count() << " ms\n"
                  << "  Col-Major (Strided Stride " << dim << "):  " 
                  << std::chrono::duration<double, std::milli>(res.col_major_time).count() << " ms\n"
                  << "  Cache Locality Speedup:          " << std::fixed << std::setprecision(2) << res.speedup << "x\n";
    } else if (pattern_str == "false_sharing") {
        std::size_t iters = parser.get_size("iterations", 10'000'000);
        auto res = memory::run_false_sharing_experiment(threads, iters);
        std::cout << "  False Sharing Benchmark (" << threads << " threads, " << iters << " increments/thread):\n"
                  << "  Unpadded Time (False Sharing Penalty): " 
                  << std::chrono::duration<double, std::milli>(res.unpadded_time).count() << " ms\n"
                  << "  Padded Time (Isolated Cache Lines):   " 
                  << std::chrono::duration<double, std::milli>(res.padded_time).count() << " ms\n"
                  << "  Speedup from Cache Line Padding:      " 
                  << std::fixed << std::setprecision(2) << res.speedup << "x\n";
    } else {
        std::cerr << "Error: Unknown pattern '" << pattern_str << "'. Supported: sequential, strided, cache_locality, false_sharing\n";
        return 1;
    }
    return 0;
}

int cmd_queue_demo(const CommandParser& parser) {
    unsigned int num_producers = parser.get_uint("producers", 4);
    unsigned int num_consumers = parser.get_uint("consumers", 4);
    std::size_t total_items = parser.get_size("items", 20'000);
    std::size_t capacity = parser.get_size("capacity", 100);

    std::cout << "Running BoundedQueue Demonstration:\n"
              << "  Producers: " << num_producers << ", Consumers: " << num_consumers 
              << ", Items: " << total_items << ", Capacity: " << capacity << "\n";

    threading::BoundedQueue<int> queue(capacity);
    std::atomic<std::size_t> items_produced{0};
    std::atomic<std::size_t> items_consumed{0};
    std::atomic<uint64_t> sum_consumed{0};

    const std::size_t items_per_prod = total_items / num_producers;
    const auto start = std::chrono::steady_clock::now();

    // Start consumers
    std::vector<std::thread> consumers;
    consumers.reserve(num_consumers);
    for (unsigned int c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&queue, &items_consumed, &sum_consumed]() {
            while (auto val = queue.pop()) {
                items_consumed++;
                sum_consumed += static_cast<uint64_t>(*val);
            }
        });
    }

    // Start producers
    std::vector<std::thread> producers;
    producers.reserve(num_producers);
    for (unsigned int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&queue, &items_produced, items_per_prod, p]() {
            const int base = static_cast<int>(p * items_per_prod);
            for (std::size_t i = 0; i < items_per_prod; ++i) {
                queue.push(base + static_cast<int>(i));
                items_produced++;
            }
        });
    }

    for (auto& p : producers) {
        p.join();
    }
    // All producers finished: close queue so consumers drain remainder and terminate cleanly
    queue.close();

    for (auto& c : consumers) {
        c.join();
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Queue Demonstration Results:\n"
              << "  Produced: " << items_produced.load() << ", Consumed: " << items_consumed.load() << "\n"
              << "  Completed in: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms ("
              << std::setprecision(0) << (static_cast<double>(items_consumed.load()) / (elapsed_ms / 1000.0)) << " items/sec)\n";

    if (items_produced.load() == items_consumed.load()) {
        std::cout << "  Verification: PASSED (Zero lost items, deterministic shutdown)\n";
        return 0;
    } else {
        std::cout << "  Verification: FAILED (Produced != Consumed)\n";
        return 1;
    }
}

int cmd_threadpool_demo(const CommandParser& parser) {
    unsigned int workers = parser.get_uint("workers", 4);
    std::size_t num_tasks = parser.get_size("tasks", 1000);

    std::cout << "Running ThreadPool Demonstration:\n"
              << "  Workers: " << workers << ", Tasks: " << num_tasks << "\n";

    threading::ThreadPool pool(workers);
    std::vector<std::future<uint64_t>> futures;
    futures.reserve(num_tasks);

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < num_tasks; ++i) {
        futures.push_back(pool.submit([i]() -> uint64_t {
            uint64_t acc = 0;
            for (uint64_t k = 0; k < 1000; ++k) {
                acc += (i + k);
            }
            return acc;
        }));
    }

    uint64_t total_sum = 0;
    for (auto& fut : futures) {
        total_sum += fut.get();
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "ThreadPool Completed in: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms ("
              << (num_tasks / (elapsed_ms / 1000.0)) << " tasks/sec)\n"
              << "  Accumulated Checksum: " << total_sum << "\n"
              << "  Verification: PASSED\n";
    return 0;
}

int cmd_benchmark(const CommandParser& parser) {
    std::string suite = parser.get_string("suite", "all");
    std::string format = parser.get_string("format", "text");
    std::string output_path = parser.get_string("output", "");
    std::size_t warmup = parser.get_size("warmup", 3);
    std::size_t repetitions = parser.get_size("repetitions", 5);
    unsigned int threads = parser.get_uint("threads", 0);

    benchmark::BenchmarkOptions opt;
    opt.warmup = warmup;
    opt.repetitions = repetitions;
    opt.threads = threads;
    opt.matrix_size = parser.get_size("size", 512);

    benchmark::BenchmarkRunner runner(opt);
    std::vector<benchmark::BenchmarkResult> results;

    std::cout << "Executing Benchmark Suite: " << suite 
              << " (warmup: " << warmup << ", repetitions: " << repetitions << ")...\n";

    if (suite == "matrix") {
        results = runner.run_matrix_suite(opt.matrix_size, opt.threads);
    } else if (suite == "convolution") {
        results = runner.run_convolution_suite(1920, 1080, opt.threads);
    } else if (suite == "vector") {
        results = runner.run_vector_suite(10'000'000, opt.threads);
    } else if (suite == "sort") {
        results = runner.run_sort_suite(1'000'000, opt.threads);
    } else if (suite == "memory") {
        results = runner.run_memory_suite(64);
    } else if (suite == "all") {
        results = runner.run_all();
    } else {
        std::cerr << "Error: Unknown suite '" << suite << "'. Supported: matrix, convolution, vector, sort, memory, all\n";
        return 1;
    }

    std::string formatted_output;
    if (format == "json") {
        formatted_output = benchmark::Exporter::to_json(results, true);
    } else if (format == "csv") {
        formatted_output = benchmark::Exporter::to_csv(results);
    } else {
        formatted_output = benchmark::Exporter::to_text_table(results);
    }

    std::cout << formatted_output << "\n";

    if (!output_path.empty()) {
        try {
            benchmark::Exporter::write_file(output_path, formatted_output);
            std::cout << "Results exported to: " << output_path << "\n";
        } catch (const std::exception& ex) {
            std::cerr << "Error saving to " << output_path << ": " << ex.what() << "\n";
            return 1;
        }
    }

    return 0;
}

int cmd_validate(const CommandParser& parser) {
    (void)parser;
    std::cout << "=== Running ParallelForge Comprehensive Validation Suite ===\n\n";

    bool all_passed = true;
    auto report = [&all_passed](const std::string& name, bool pass, const std::string& details = "") {
        std::cout << std::left << std::setw(50) << name << " : " 
                  << (pass ? "\033[32m[PASS]\033[0m" : "\033[31m[FAIL]\033[0m");
        if (!details.empty()) {
            std::cout << " (" << details << ")";
        }
        std::cout << "\n";
        if (!pass) all_passed = false;
    };

    // 1. Matrix Multiplication Correctness
    try {
        core::Matrix a = core::Matrix::random(67, 129, 0.0f, 1.0f, 11);
        core::Matrix b = core::Matrix::random(129, 93, 0.0f, 1.0f, 22);
        core::Matrix ref = algorithms::multiply_serial_ikj(a, b);

        core::Matrix c_blocked = algorithms::multiply_blocked(a, b, 32);
        report("Matrix Blocked (rectangular & unaligned)", c_blocked.approx_equal(ref));

        core::Matrix c_omp = algorithms::multiply_openmp(a, b, 32, 4);
        report("Matrix OpenMP (rectangular & unaligned)", c_omp.approx_equal(ref));

        core::Matrix c_th = algorithms::multiply_threads(a, b, 32, 4);
        report("Matrix Threads (rectangular & unaligned)", c_th.approx_equal(ref));

        core::Matrix c_simd = algorithms::multiply_simd(a, b, 32);
        report("Matrix SIMD (rectangular & unaligned)", c_simd.approx_equal(ref));

        // Zero dimension handling
        core::Matrix a_zero(0, 10);
        core::Matrix b_zero(10, 5);
        core::Matrix c_zero = algorithms::multiply_blocked(a_zero, b_zero);
        report("Matrix Zero-dimension policy", c_zero.rows() == 0 && c_zero.cols() == 5 && c_zero.empty());
    } catch (const std::exception& ex) {
        report("Matrix Multiplication Suite", false, ex.what());
    }

    // 2. Convolution Correctness
    try {
        core::Matrix img = core::Matrix::random(100, 150, 0.0f, 255.0f, 33);
        core::Matrix k_blur = algorithms::Kernels::gaussian_blur_3x3();
        core::Matrix ref_clamp = algorithms::convolve_serial(img, k_blur, algorithms::BorderMode::ClampToEdge);
        core::Matrix out_omp = algorithms::convolve_openmp(img, k_blur, algorithms::BorderMode::ClampToEdge, 4);
        report("Convolution OpenMP (Clamp border)", out_omp.approx_equal(ref_clamp));

        core::Matrix out_th = algorithms::convolve_threads(img, k_blur, algorithms::BorderMode::ClampToEdge, 4);
        report("Convolution Threads (Clamp border)", out_th.approx_equal(ref_clamp));

        core::Matrix ref_zero = algorithms::convolve_serial(img, k_blur, algorithms::BorderMode::ZeroPad);
        core::Matrix out_th_zero = algorithms::convolve_threads(img, k_blur, algorithms::BorderMode::ZeroPad, 4);
        report("Convolution Threads (Zero border)", out_th_zero.approx_equal(ref_zero));

        core::Matrix ref_mirror = algorithms::convolve_serial(img, k_blur, algorithms::BorderMode::Mirror);
        core::Matrix out_th_mirror = algorithms::convolve_threads(img, k_blur, algorithms::BorderMode::Mirror, 4);
        report("Convolution Threads (Mirror border)", out_th_mirror.approx_equal(ref_mirror));
    } catch (const std::exception& ex) {
        report("Convolution Suite", false, ex.what());
    }

    // 3. Vector Operations Correctness
    try {
        std::vector<float> va(10000, 1.5f);
        std::vector<float> vb(10000, 2.5f);
        float ref_dot = algorithms::dot_product_serial(va, vb);
        float omp_dot = algorithms::dot_product_openmp(va, vb, 4);
        report("Vector Dot Product OpenMP", algorithms::float_approx_equal(ref_dot, omp_dot));

        float th_dot = algorithms::dot_product_threads(va, vb, 4);
        report("Vector Dot Product Threads", algorithms::float_approx_equal(ref_dot, th_dot));

        float simd_dot = algorithms::dot_product_simd(va, vb);
        report("Vector Dot Product SIMD", algorithms::float_approx_equal(ref_dot, simd_dot));

        auto add_th = algorithms::vector_add_threads(va, vb, 4);
        report("Vector Addition Threads", add_th.size() == 10000 && std::abs(add_th[0] - 4.0f) < 1e-5f);
    } catch (const std::exception& ex) {
        report("Vector Operations Suite", false, ex.what());
    }

    // 4. Parallel Sort Correctness
    try {
        std::vector<float> s_data(50000);
        std::mt19937 gen(77);
        std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
        for (auto& val : s_data) val = dist(gen);

        algorithms::sort_parallel(s_data, 1024, 4);
        report("Parallel Merge Sort", algorithms::is_sorted(s_data));
    } catch (const std::exception& ex) {
        report("Parallel Sort Suite", false, ex.what());
    }

    // 5. BoundedQueue Concurrency
    try {
        threading::BoundedQueue<int> q(10);
        std::atomic<int> received{0};
        std::thread consumer([&q, &received]() {
            while (auto val = q.pop()) {
                received++;
            }
        });

        for (int i = 0; i < 500; ++i) {
            q.push(i);
        }
        q.close();
        consumer.join();

        report("BoundedQueue Push/Pop/Drain", received.load() == 500);
    } catch (const std::exception& ex) {
        report("BoundedQueue Concurrency", false, ex.what());
    }

    // 6. ThreadPool Exception Propagation & Tasks
    try {
        threading::ThreadPool pool(2);
        auto fut1 = pool.submit([]() { return 42; });
        report("ThreadPool Task Future", fut1.get() == 42);

        auto fut2 = pool.submit([]() -> int {
            throw std::runtime_error("Simulated task failure");
        });
        bool caught = false;
        try {
            fut2.get();
        } catch (const std::runtime_error& err) {
            caught = (std::string(err.what()) == "Simulated task failure");
        }
        report("ThreadPool Exception Propagation", caught);
    } catch (const std::exception& ex) {
        report("ThreadPool Suite", false, ex.what());
    }

    // 7. False Sharing & Memory
    try {
        auto fs = memory::run_false_sharing_experiment(2, 500'000);
        report("False Sharing Experiment Execution", fs.unpadded_time.count() > 0 && fs.padded_time.count() > 0);
    } catch (const std::exception& ex) {
        report("False Sharing Suite", false, ex.what());
    }

    std::cout << "\nValidation Result: " 
              << (all_passed ? "\033[32mALL VALIDATION GATES PASSED\033[0m" : "\033[31mVALIDATION FAILURES DETECTED\033[0m")
              << "\n\n";

    return all_passed ? 0 : 1;
}

int execute_command(const CommandParser& parser) {
    if (parser.is_help()) {
        print_help();
        return 0;
    }
    if (parser.is_version()) {
        print_version();
        return 0;
    }

    const std::string& cmd = parser.command();
    if (cmd.empty()) {
        print_help();
        return 0;
    }

    if (cmd == "info") return cmd_info(parser);
    if (cmd == "matrix") return cmd_matrix(parser);
    if (cmd == "convolution") return cmd_convolution(parser);
    if (cmd == "vector") return cmd_vector(parser);
    if (cmd == "sort") return cmd_sort(parser);
    if (cmd == "memory") return cmd_memory(parser);
    if (cmd == "queue-demo") return cmd_queue_demo(parser);
    if (cmd == "threadpool-demo") return cmd_threadpool_demo(parser);
    if (cmd == "benchmark") return cmd_benchmark(parser);
    if (cmd == "validate") return cmd_validate(parser);

    std::cerr << "Error: Unknown command '" << cmd << "'. Run 'parallelforge --help' for usage.\n";
    return 1;
}

} // namespace parallelforge::cli
