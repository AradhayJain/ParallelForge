#include <parallelforge/benchmark/exporter.hpp>
#include <parallelforge/core/error.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace parallelforge::benchmark {

std::string Exporter::to_text_table(const std::vector<BenchmarkResult>& results) {
    if (results.empty()) {
        return "No benchmark results recorded.\n";
    }

    std::ostringstream oss;
    oss << "\n"
        << std::left << std::setw(12) << "Suite"
        << std::left << std::setw(28) << "Test"
        << std::left << std::setw(18) << "Algorithm"
        << std::right << std::setw(8) << "Threads"
        << std::right << std::setw(12) << "Mean (ms)"
        << std::right << std::setw(12) << "Median (ms)"
        << std::right << std::setw(10) << "Speedup"
        << std::right << std::setw(10) << "Effic."
        << std::right << std::setw(10) << "Status"
        << "\n";
    oss << std::string(110, '-') << "\n";

    for (const auto& r : results) {
        oss << std::left << std::setw(12) << r.suite_name
            << std::left << std::setw(28) << r.test_name
            << std::left << std::setw(18) << r.algorithm_name
            << std::right << std::setw(8) << r.threads
            << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.metrics.mean_ms
            << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.metrics.median_ms
            << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.metrics.speedup << "x"
            << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.metrics.efficiency
            << std::right << std::setw(10) << (r.passed_verification ? "PASS" : "FAIL")
            << "\n";
    }
    oss << std::string(110, '-') << "\n";
    return oss.str();
}

std::string Exporter::to_json(const std::vector<BenchmarkResult>& results, bool emit_raw_samples) {
    std::ostringstream oss;
    oss << "[\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        oss << "  {\n";
        oss << "    \"suite\": \"" << r.suite_name << "\",\n";
        oss << "    \"test\": \"" << r.test_name << "\",\n";
        oss << "    \"algorithm\": \"" << r.algorithm_name << "\",\n";
        oss << "    \"workload_size\": " << r.workload_size << ",\n";
        oss << "    \"threads\": " << r.threads << ",\n";
        oss << "    \"warmup\": " << r.warmup_iterations << ",\n";
        oss << "    \"repetitions\": " << r.repetitions << ",\n";
        oss << "    \"passed_verification\": " << (r.passed_verification ? "true" : "false") << ",\n";
        oss << "    \"verification_details\": \"" << r.verification_details << "\",\n";
        oss << "    \"metrics\": {\n";
        oss << "      \"min_ms\": " << r.metrics.min_ms << ",\n";
        oss << "      \"max_ms\": " << r.metrics.max_ms << ",\n";
        oss << "      \"mean_ms\": " << r.metrics.mean_ms << ",\n";
        oss << "      \"median_ms\": " << r.metrics.median_ms << ",\n";
        oss << "      \"stddev_ms\": " << r.metrics.stddev_ms << ",\n";
        oss << "      \"speedup\": " << r.metrics.speedup << ",\n";
        oss << "      \"efficiency\": " << r.metrics.efficiency << ",\n";
        oss << "      \"throughput\": " << r.metrics.throughput << "\n";
        oss << "    },\n";

        if (emit_raw_samples) {
            oss << "    \"samples_ms\": [";
            for (std::size_t s = 0; s < r.sample_times_ms.size(); ++s) {
                if (s > 0) oss << ", ";
                oss << r.sample_times_ms[s];
            }
            oss << "],\n";
        }

        oss << "    \"metadata\": {\n";
        oss << "      \"os\": \"" << r.os_name << "\",\n";
        oss << "      \"architecture\": \"" << r.architecture << "\",\n";
        oss << "      \"compiler\": \"" << r.compiler << "\",\n";
        oss << "      \"cpu_model\": \"" << r.cpu_model << "\",\n";
        oss << "      \"timestamp\": \"" << r.timestamp << "\"\n";
        oss << "    }\n";
        oss << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    oss << "]\n";
    return oss.str();
}

std::string Exporter::to_csv(const std::vector<BenchmarkResult>& results) {
    std::ostringstream oss;
    // CSV Header
    oss << "suite,test,algorithm,workload_size,threads,warmup,repetitions,"
        << "min_ms,max_ms,mean_ms,median_ms,stddev_ms,speedup,efficiency,throughput,"
        << "passed_verification,os,architecture,compiler,cpu_model,timestamp\n";

    for (const auto& r : results) {
        oss << "\"" << r.suite_name << "\","
            << "\"" << r.test_name << "\","
            << "\"" << r.algorithm_name << "\","
            << r.workload_size << ","
            << r.threads << ","
            << r.warmup_iterations << ","
            << r.repetitions << ","
            << r.metrics.min_ms << ","
            << r.metrics.max_ms << ","
            << r.metrics.mean_ms << ","
            << r.metrics.median_ms << ","
            << r.metrics.stddev_ms << ","
            << r.metrics.speedup << ","
            << r.metrics.efficiency << ","
            << r.metrics.throughput << ","
            << (r.passed_verification ? "true" : "false") << ","
            << "\"" << r.os_name << "\","
            << "\"" << r.architecture << "\","
            << "\"" << r.compiler << "\","
            << "\"" << r.cpu_model << "\","
            << "\"" << r.timestamp << "\"\n";
    }
    return oss.str();
}

void Exporter::write_file(const std::string& filepath, const std::string& content) {
    std::filesystem::path p(filepath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    std::ofstream out(filepath);
    if (!out.is_open()) {
        throw core::ParallelForgeException("Failed to open file for writing: " + filepath);
    }
    out << content;
}

} // namespace parallelforge::benchmark
