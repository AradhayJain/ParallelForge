#pragma once

#include <parallelforge/benchmark/benchmark_result.hpp>

#include <string>
#include <vector>

namespace parallelforge::benchmark {

class Exporter {
public:
    [[nodiscard]] static std::string to_text_table(const std::vector<BenchmarkResult>& results);
    [[nodiscard]] static std::string to_json(const std::vector<BenchmarkResult>& results, bool emit_raw_samples = true);
    [[nodiscard]] static std::string to_csv(const std::vector<BenchmarkResult>& results);

    static void write_file(const std::string& filepath, const std::string& content);
};

} // namespace parallelforge::benchmark
