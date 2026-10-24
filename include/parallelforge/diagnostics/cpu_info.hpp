#pragma once

#include <parallelforge/diagnostics/hardware_features.hpp>

#include <cstddef>
#include <string>
#include <optional>

namespace parallelforge::diagnostics {

struct CacheLevelInfo {
    std::size_t size_bytes{0};
    std::size_t line_size_bytes{0};
    bool is_detected{false}; // True if parsed from sysfs, false if fallback
};

struct CpuInfo {
    std::string model_name{"Unknown CPU"};
    unsigned int logical_cores{1};
    unsigned int physical_cores{1};
    std::size_t cache_line_size{64};
    bool cache_line_detected{false};

    CacheLevelInfo l1d_cache;
    CacheLevelInfo l1i_cache;
    CacheLevelInfo l2_cache;
    CacheLevelInfo l3_cache;

    HardwareFeatures features;

    [[nodiscard]] static CpuInfo discover(std::optional<std::size_t> user_cache_line_override = std::nullopt);
    [[nodiscard]] std::string to_string() const;
};

} // namespace parallelforge::diagnostics
