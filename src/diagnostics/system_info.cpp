#include <parallelforge/diagnostics/system_info.hpp>

#include <sys/utsname.h>
#include <sstream>

#if PARALLELFORGE_HAS_OPENMP
#include <omp.h>
#endif

namespace parallelforge::diagnostics {

SystemInfo SystemInfo::collect(std::optional<std::size_t> user_cache_line) {
    SystemInfo sys;

    // Operating System & Architecture
    struct utsname u{};
    if (uname(&u) == 0) {
        sys.os_name = u.sysname;
        sys.os_release = u.release;
        sys.architecture = u.machine;
    } else {
        sys.os_name = "Linux";
        sys.os_release = "Unknown";
        sys.architecture = "x86_64";
    }

    // Compiler Information
#if defined(__clang__)
    sys.compiler_name = "Clang";
    sys.compiler_version = __clang_version__;
#elif defined(__GNUC__)
    sys.compiler_name = "GCC";
    sys.compiler_version = std::to_string(__GNUC__) + "." + 
                           std::to_string(__GNUC_MINOR__) + "." + 
                           std::to_string(__GNUC_PATCHLEVEL__);
#else
    sys.compiler_name = "Unknown";
    sys.compiler_version = "Unknown";
#endif

    // C++ Standard
#if __cplusplus >= 202302L
    sys.cpp_standard = "C++23";
#elif __cplusplus >= 202002L
    sys.cpp_standard = "C++20";
#elif __cplusplus >= 201703L
    sys.cpp_standard = "C++17";
#else
    sys.cpp_standard = "C++" + std::to_string(__cplusplus);
#endif

    // OpenMP
#if PARALLELFORGE_HAS_OPENMP
    sys.openmp_enabled = true;
    sys.openmp_version = std::to_string(_OPENMP);
    sys.openmp_max_threads = omp_get_max_threads();
#else
    sys.openmp_enabled = false;
    sys.openmp_version = "Disabled";
    sys.openmp_max_threads = 1;
#endif

    // CPU discovery
    sys.cpu = CpuInfo::discover(user_cache_line);

    return sys;
}

std::string SystemInfo::to_string() const {
    std::ostringstream oss;
    oss << "=== ParallelForge System Diagnostics ===\n";
    oss << "OS:                   " << os_name << " " << os_release << " (" << architecture << ")\n";
    oss << "Compiler:             " << compiler_name << " " << compiler_version << "\n";
    oss << "C++ Standard:         " << cpp_standard << "\n";
    oss << "OpenMP Support:       " << (openmp_enabled ? "Enabled" : "Disabled") 
        << " (Specification: " << openmp_version << ", Max Threads: " << openmp_max_threads << ")\n";
    oss << "----------------------------------------\n";
    oss << cpu.to_string();
    return oss.str();
}

std::string SystemInfo::to_json() const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"os_name\": \"" << os_name << "\",\n";
    oss << "  \"os_release\": \"" << os_release << "\",\n";
    oss << "  \"architecture\": \"" << architecture << "\",\n";
    oss << "  \"compiler_name\": \"" << compiler_name << "\",\n";
    oss << "  \"compiler_version\": \"" << compiler_version << "\",\n";
    oss << "  \"cpp_standard\": \"" << cpp_standard << "\",\n";
    oss << "  \"openmp_enabled\": " << (openmp_enabled ? "true" : "false") << ",\n";
    oss << "  \"openmp_version\": \"" << openmp_version << "\",\n";
    oss << "  \"openmp_max_threads\": " << openmp_max_threads << ",\n";
    oss << "  \"cpu\": {\n";
    oss << "    \"model_name\": \"" << cpu.model_name << "\",\n";
    oss << "    \"logical_cores\": " << cpu.logical_cores << ",\n";
    oss << "    \"physical_cores\": " << cpu.physical_cores << ",\n";
    oss << "    \"cache_line_size\": " << cpu.cache_line_size << ",\n";
    oss << "    \"cache_line_detected\": " << (cpu.cache_line_detected ? "true" : "false") << ",\n";
    oss << "    \"l1d_size_bytes\": " << cpu.l1d_cache.size_bytes << ",\n";
    oss << "    \"l2_size_bytes\": " << cpu.l2_cache.size_bytes << ",\n";
    oss << "    \"l3_size_bytes\": " << cpu.l3_cache.size_bytes << "\n";
    oss << "  }\n";
    oss << "}";
    return oss.str();
}

} // namespace parallelforge::diagnostics
