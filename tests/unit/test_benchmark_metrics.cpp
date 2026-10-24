#include <gtest/gtest.h>
#include <parallelforge/benchmark/metric_calculator.hpp>
#include <parallelforge/benchmark/exporter.hpp>

using namespace parallelforge::benchmark;

TEST(MetricCalculatorTest, EmptySamples) {
    auto m = MetricCalculator::calculate({});
    EXPECT_DOUBLE_EQ(m.mean_ms, 0.0);
    EXPECT_DOUBLE_EQ(m.median_ms, 0.0);
    EXPECT_DOUBLE_EQ(m.stddev_ms, 0.0);
    EXPECT_DOUBLE_EQ(m.speedup, 1.0);
}

TEST(MetricCalculatorTest, SingleSample) {
    auto m = MetricCalculator::calculate({42.0});
    EXPECT_DOUBLE_EQ(m.min_ms, 42.0);
    EXPECT_DOUBLE_EQ(m.max_ms, 42.0);
    EXPECT_DOUBLE_EQ(m.mean_ms, 42.0);
    EXPECT_DOUBLE_EQ(m.median_ms, 42.0);
    EXPECT_DOUBLE_EQ(m.stddev_ms, 0.0);
}

TEST(MetricCalculatorTest, MultiSampleStatistics) {
    // Samples: [10, 20, 30, 40, 50]
    // Mean = 30
    // Median = 30
    // Variance = (400 + 100 + 0 + 100 + 400) / 4 = 1000 / 4 = 250
    // Stddev = sqrt(250) ≈ 15.8113883
    std::vector<double> samples = {10.0, 50.0, 20.0, 40.0, 30.0};
    auto m = MetricCalculator::calculate(samples, 60.0, 4, 1000.0);

    EXPECT_DOUBLE_EQ(m.min_ms, 10.0);
    EXPECT_DOUBLE_EQ(m.max_ms, 50.0);
    EXPECT_DOUBLE_EQ(m.mean_ms, 30.0);
    EXPECT_DOUBLE_EQ(m.median_ms, 30.0);
    EXPECT_NEAR(m.stddev_ms, 15.8113883, 1e-5);
    EXPECT_DOUBLE_EQ(m.speedup, 2.0); // 60 / 30
    EXPECT_DOUBLE_EQ(m.efficiency, 0.5); // 2.0 / 4
}

TEST(MetricCalculatorTest, ZeroDivisionSafeguards) {
    EXPECT_DOUBLE_EQ(MetricCalculator::calculate_speedup(0.0, 10.0), 1.0);
    EXPECT_DOUBLE_EQ(MetricCalculator::calculate_speedup(10.0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(MetricCalculator::calculate_efficiency(2.0, 0), 1.0);
}

TEST(ExporterTest, CsvAndJsonSerialization) {
    BenchmarkResult r;
    r.suite_name = "matrix";
    r.test_name = "multiply_512";
    r.algorithm_name = "blocked";
    r.workload_size = 512;
    r.threads = 4;
    r.warmup_iterations = 2;
    r.repetitions = 3;
    r.sample_times_ms = {12.0, 10.0, 11.0};
    r.metrics = MetricCalculator::calculate(r.sample_times_ms, 20.0, 4, 512.0);
    r.passed_verification = true;
    r.os_name = "Linux";
    r.architecture = "x86_64";
    r.compiler = "GCC 13";
    r.cpu_model = "Test CPU";
    r.timestamp = "2026-10-24T12:00:00";

    std::vector<BenchmarkResult> results = {r};

    std::string csv = Exporter::to_csv(results);
    EXPECT_NE(csv.find("suite,test,algorithm"), std::string::npos);
    EXPECT_NE(csv.find("\"matrix\",\"multiply_512\",\"blocked\""), std::string::npos);

    std::string json = Exporter::to_json(results);
    EXPECT_EQ(json.front(), '[');
    EXPECT_NE(json.find("\"suite\": \"matrix\""), std::string::npos);
    EXPECT_NE(json.find("\"median_ms\": 11"), std::string::npos);
}
