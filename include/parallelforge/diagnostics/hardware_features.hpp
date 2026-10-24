#pragma once

#include <string>
#include <vector>

namespace parallelforge::diagnostics {

struct HardwareFeatures {
    bool has_sse{false};
    bool has_sse2{false};
    bool has_sse3{false};
    bool has_ssse3{false};
    bool has_sse4_1{false};
    bool has_sse4_2{false};
    bool has_avx{false};
    bool has_avx2{false};
    bool has_fma{false};
    bool has_avx512f{false};

    [[nodiscard]] static HardwareFeatures detect() noexcept;
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::vector<std::string> feature_list() const;
};

} // namespace parallelforge::diagnostics
