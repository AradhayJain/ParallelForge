#include <gtest/gtest.h>
#include <parallelforge/diagnostics/cpu_info.hpp>
#include <parallelforge/diagnostics/system_info.hpp>
#include <parallelforge/diagnostics/hardware_features.hpp>

using namespace parallelforge::diagnostics;

TEST(DiagnosticsTest, HardwareFeaturesDetect) {
    auto feat = HardwareFeatures::detect();
    EXPECT_FALSE(feat.to_string().empty());
}

TEST(DiagnosticsTest, CpuInfoDiscovery) {
    auto cpu = CpuInfo::discover();
    EXPECT_GE(cpu.logical_cores, 1u);
    EXPECT_GE(cpu.physical_cores, 1u);
    EXPECT_GT(cpu.cache_line_size, 0u);
    EXPECT_FALSE(cpu.model_name.empty());
    EXPECT_FALSE(cpu.to_string().empty());
}

TEST(DiagnosticsTest, UserCacheLineOverride) {
    auto cpu = CpuInfo::discover(128);
    EXPECT_EQ(cpu.cache_line_size, 128u);
    EXPECT_FALSE(cpu.cache_line_detected);
}

TEST(DiagnosticsTest, SystemInfoCollection) {
    auto sys = SystemInfo::collect();
    EXPECT_FALSE(sys.os_name.empty());
    EXPECT_FALSE(sys.compiler_name.empty());
    EXPECT_FALSE(sys.cpp_standard.empty());
    EXPECT_FALSE(sys.to_string().empty());
    EXPECT_FALSE(sys.to_json().empty());

    // Basic JSON syntax check
    std::string json = sys.to_json();
    EXPECT_EQ(json.front(), '{');
    EXPECT_EQ(json.back(), '}');
    EXPECT_NE(json.find("\"compiler_name\":"), std::string::npos);
}
