#include <parallelforge/benchmark/benchmark_runner.hpp>
#include <parallelforge/benchmark/exporter.hpp>
#include <parallelforge/cli/command_parser.hpp>
#include <parallelforge/core/error.hpp>
#include "workloads/workloads.hpp"

#include <iostream>

int main(int argc, char* argv[]) {
    try {
        parallelforge::cli::CommandParser parser(argc, argv);
        std::string suite = parser.get_string("suite", "all");
        std::string format = parser.get_string("format", "text");
        std::string output = parser.get_string("output", "");
        std::size_t warmup = parser.get_size("warmup", 3);
        std::size_t reps = parser.get_size("repetitions", 5);

        parallelforge::benchmark::BenchmarkOptions options;
        options.warmup = warmup;
        options.repetitions = reps;

        parallelforge::benchmark::BenchmarkRunner runner(options);

        std::cout << "Starting ParallelForge Benchmark Suite (" << suite << ")...\n";

        if (suite == "matrix") {
            parallelforge::benchmarks::run_bench_matrix(runner);
        } else if (suite == "convolution") {
            parallelforge::benchmarks::run_bench_convolution(runner);
        } else if (suite == "vector") {
            parallelforge::benchmarks::run_bench_vector(runner);
        } else if (suite == "memory") {
            parallelforge::benchmarks::run_bench_memory(runner);
        } else {
            parallelforge::benchmarks::run_bench_matrix(runner);
            parallelforge::benchmarks::run_bench_convolution(runner);
            parallelforge::benchmarks::run_bench_vector(runner);
            parallelforge::benchmarks::run_bench_memory(runner);
        }

        const auto& results = runner.results();
        std::string out_str;
        if (format == "json") {
            out_str = parallelforge::benchmark::Exporter::to_json(results);
        } else if (format == "csv") {
            out_str = parallelforge::benchmark::Exporter::to_csv(results);
        } else {
            out_str = parallelforge::benchmark::Exporter::to_text_table(results);
        }

        std::cout << out_str << "\n";

        if (!output.empty()) {
            parallelforge::benchmark::Exporter::write_file(output, out_str);
            std::cout << "Wrote benchmark results to " << output << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Benchmark failure: " << ex.what() << "\n";
        return 1;
    }
}
