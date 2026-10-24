#pragma once

#include <vector>
#include <cstddef>

namespace parallelforge::benchmark {

struct AggregateMetrics {
    double min_ms{0.0};
    double max_ms{0.0};
    double mean_ms{0.0};
    double median_ms{0.0};
    double stddev_ms{0.0};
    double speedup{1.0};
    double efficiency{1.0};
    double throughput{0.0}; // e.g. GFLOPS, GB/s, or ops/sec
};

class MetricCalculator {
public:
    /**
     * @brief Computes min, max, mean, median, and sample standard deviation from sample times in ms.
     */
    [[nodiscard]] static AggregateMetrics calculate(const std::vector<double>& samples_ms, 
                                                   double baseline_mean_ms = 0.0, 
                                                   unsigned int workers = 1,
                                                   double total_ops = 0.0) noexcept;

    /**
     * @brief Computes speedup safely without division by zero or NaN.
     */
    [[nodiscard]] static double calculate_speedup(double baseline_time, double measured_time) noexcept;

    /**
     * @brief Computes parallel efficiency safely: speedup / workers.
     */
    [[nodiscard]] static double calculate_efficiency(double speedup, unsigned int workers) noexcept;
};

} // namespace parallelforge::benchmark
