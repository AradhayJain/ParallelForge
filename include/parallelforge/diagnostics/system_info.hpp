#pragma once

#include <parallelforge/diagnostics/cpu_info.hpp>

#include <string>

namespace parallelforge::diagnostics {

struct SystemInfo {
    std::string os_name;
    std::string os_release;
    std::string architecture;
    std::string compiler_name;
    std::string compiler_version;
    std::string cpp_standard;
    bool openmp_enabled{false};
    std::string openmp_version{"None"};
    int openmp_max_threads{1};

    CpuInfo cpu;

    [[nodiscard]] static SystemInfo collect(std::optional<std::size_t> user_cache_line = std::nullopt);
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_json() const;
};

} // namespace parallelforge::diagnostics
