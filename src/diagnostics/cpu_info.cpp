#include <parallelforge/diagnostics/cpu_info.hpp>
#include <parallelforge/core/config.hpp>

#include <fstream>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <set>

namespace parallelforge::diagnostics {

namespace {

std::size_t parse_cache_size_str(const std::string& str) {
    if (str.empty()) return 0;
    std::size_t num = 0;
    std::size_t i = 0;
    while (i < str.size() && std::isdigit(str[i])) {
        num = num * 10 + static_cast<std::size_t>(str[i] - '0');
        ++i;
    }
    if (i < str.size()) {
        char unit = str[i];
        if (unit == 'K' || unit == 'k') num *= 1024;
        else if (unit == 'M' || unit == 'm') num *= 1024 * 1024;
        else if (unit == 'G' || unit == 'g') num *= 1024 * 1024 * 1024;
    }
    return num;
}

void read_sysfs_cache(const std::string& base_path, CacheLevelInfo& info) {
    std::ifstream size_file(base_path + "/size");
    if (size_file.is_open()) {
        std::string s;
        if (size_file >> s) {
            info.size_bytes = parse_cache_size_str(s);
            info.is_detected = true;
        }
    }
    std::ifstream line_file(base_path + "/coherency_line_size");
    if (line_file.is_open()) {
        std::size_t line_sz = 0;
        if (line_file >> line_sz) {
            info.line_size_bytes = line_sz;
        }
    }
}

} // namespace

CpuInfo CpuInfo::discover(std::optional<std::size_t> user_cache_line_override) {
    CpuInfo info;

    // Logical core discovery
    info.logical_cores = std::thread::hardware_concurrency();
    if (info.logical_cores == 0) {
        long nproc = sysconf(_SC_NPROCESSORS_ONLN);
        info.logical_cores = nproc > 0 ? static_cast<unsigned int>(nproc) : 1;
    }
    info.physical_cores = info.logical_cores; // Default initial assumption

    // Parse /proc/cpuinfo on Linux
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        std::set<std::string> physical_ids;
        std::set<std::pair<std::string, std::string>> core_pairs;
        std::string current_phys_id = "0";
        bool model_found = false;

        while (std::getline(cpuinfo, line)) {
            if (!model_found && line.find("model name") != std::string::npos) {
                auto colon = line.find(':');
                if (colon != std::string::npos) {
                    std::string model = line.substr(colon + 1);
                    // Trim leading spaces
                    auto start = model.find_first_not_of(" \t");
                    if (start != std::string::npos) {
                        info.model_name = model.substr(start);
                        model_found = true;
                    }
                }
            } else if (line.find("physical id") != std::string::npos) {
                auto colon = line.find(':');
                if (colon != std::string::npos) {
                    current_phys_id = line.substr(colon + 1);
                }
            } else if (line.find("core id") != std::string::npos) {
                auto colon = line.find(':');
                if (colon != std::string::npos) {
                    std::string core_id = line.substr(colon + 1);
                    core_pairs.insert({current_phys_id, core_id});
                }
            }
        }
        if (!core_pairs.empty()) {
            info.physical_cores = static_cast<unsigned int>(core_pairs.size());
        }
    }

    // Cache discovery from Linux sysfs
    for (int idx = 0; idx < 5; ++idx) {
        std::string base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx);
        std::ifstream level_file(base + "/level");
        std::ifstream type_file(base + "/type");
        if (!level_file.is_open() || !type_file.is_open()) continue;

        int level = 0;
        std::string type;
        level_file >> level;
        type_file >> type;

        if (level == 1 && type == "Data") {
            read_sysfs_cache(base, info.l1d_cache);
        } else if (level == 1 && type == "Instruction") {
            read_sysfs_cache(base, info.l1i_cache);
        } else if (level == 2) {
            read_sysfs_cache(base, info.l2_cache);
        } else if (level == 3) {
            read_sysfs_cache(base, info.l3_cache);
        }
    }

    // Determine cache line size
    if (user_cache_line_override.has_value()) {
        info.cache_line_size = user_cache_line_override.value();
        info.cache_line_detected = false; // User configured
    } else if (info.l1d_cache.line_size_bytes > 0) {
        info.cache_line_size = info.l1d_cache.line_size_bytes;
        info.cache_line_detected = true;
    } else {
        info.cache_line_size = core::HARDWARE_DESTRUCTIVE_SIZE;
        info.cache_line_detected = false;
    }

    // Hardware instruction sets
    info.features = HardwareFeatures::detect();

    return info;
}

std::string CpuInfo::to_string() const {
    std::ostringstream oss;
    oss << "CPU Model:            " << model_name << "\n";
    oss << "Logical Cores:        " << logical_cores << "\n";
    oss << "Physical Cores:       " << physical_cores << "\n";
    oss << "Cache Line Size:      " << cache_line_size << " bytes (" 
        << (cache_line_detected ? "hardware detected" : "configured/fallback") << ")\n";

    auto fmt_cache = [](const CacheLevelInfo& c, const char* name) {
        std::ostringstream s;
        s << name << ": ";
        if (c.is_detected) {
            if (c.size_bytes >= 1024 * 1024) {
                s << (c.size_bytes / (1024 * 1024)) << " MiB";
            } else {
                s << (c.size_bytes / 1024) << " KiB";
            }
            s << " (detected)";
        } else {
            s << "Not detected";
        }
        return s.str();
    };

    oss << "  " << fmt_cache(l1d_cache, "L1 Data Cache       ") << "\n";
    oss << "  " << fmt_cache(l1i_cache, "L1 Instruction Cache") << "\n";
    oss << "  " << fmt_cache(l2_cache,  "L2 Cache            ") << "\n";
    oss << "  " << fmt_cache(l3_cache,  "L3 Cache (LLC)      ") << "\n";
    oss << "SIMD Features:        " << features.to_string() << "\n";
    return oss.str();
}

} // namespace parallelforge::diagnostics
