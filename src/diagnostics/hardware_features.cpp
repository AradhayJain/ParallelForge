#include <parallelforge/diagnostics/hardware_features.hpp>

#include <sstream>

namespace parallelforge::diagnostics {

HardwareFeatures HardwareFeatures::detect() noexcept {
    HardwareFeatures feat;
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    feat.has_sse    = __builtin_cpu_supports("sse");
    feat.has_sse2   = __builtin_cpu_supports("sse2");
    feat.has_sse3   = __builtin_cpu_supports("sse3");
    feat.has_ssse3  = __builtin_cpu_supports("ssse3");
    feat.has_sse4_1 = __builtin_cpu_supports("sse4.1");
    feat.has_sse4_2 = __builtin_cpu_supports("sse4.2");
    feat.has_avx    = __builtin_cpu_supports("avx");
    feat.has_avx2   = __builtin_cpu_supports("avx2");
    feat.has_fma    = __builtin_cpu_supports("fma");
    feat.has_avx512f= __builtin_cpu_supports("avx512f");
    #endif
#endif
    return feat;
}

std::vector<std::string> HardwareFeatures::feature_list() const {
    std::vector<std::string> list;
    if (has_sse) list.push_back("SSE");
    if (has_sse2) list.push_back("SSE2");
    if (has_sse3) list.push_back("SSE3");
    if (has_ssse3) list.push_back("SSSE3");
    if (has_sse4_1) list.push_back("SSE4.1");
    if (has_sse4_2) list.push_back("SSE4.2");
    if (has_avx) list.push_back("AVX");
    if (has_avx2) list.push_back("AVX2");
    if (has_fma) list.push_back("FMA");
    if (has_avx512f) list.push_back("AVX512F");
    return list;
}

std::string HardwareFeatures::to_string() const {
    const auto list = feature_list();
    if (list.empty()) return "None detected or non-x86 architecture";
    std::ostringstream oss;
    for (std::size_t i = 0; i < list.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << list[i];
    }
    return oss.str();
}

} // namespace parallelforge::diagnostics
