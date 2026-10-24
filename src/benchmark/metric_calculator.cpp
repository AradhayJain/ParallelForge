#include <parallelforge/benchmark/metric_calculator.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace parallelforge::benchmark {

double MetricCalculator::calculate_speedup(double baseline_time, double measured_time) noexcept {
    if (baseline_time <= 0.0 || measured_time <= 0.0) {
        return 1.0;
    }
    const double sp = baseline_time / measured_time;
    return std::isfinite(sp) ? sp : 1.0;
}

double MetricCalculator::calculate_efficiency(double speedup, unsigned int workers) noexcept {
    if (workers == 0) return 1.0;
    const double eff = speedup / static_cast<double>(workers);
    return std::isfinite(eff) ? eff : 0.0;
}

AggregateMetrics MetricCalculator::calculate(const std::vector<double>& samples_ms, 
                                             double baseline_mean_ms, 
                                             unsigned int workers,
                                             double total_ops) noexcept {
    AggregateMetrics metrics;
    if (samples_ms.empty()) {
        return metrics;
    }

    std::vector<double> sorted = samples_ms;
    std::sort(sorted.begin(), sorted.end());

    metrics.min_ms = sorted.front();
    metrics.max_ms = sorted.back();

    // Mean
    const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    metrics.mean_ms = sum / static_cast<double>(sorted.size());

    // Median
    const std::size_t n = sorted.size();
    if (n % 2 == 0) {
        metrics.median_ms = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    } else {
        metrics.median_ms = sorted[n / 2];
    }

    // Sample Standard Deviation: sqrt( sum((x - mean)^2) / (N - 1) )
    if (n > 1) {
        double variance_sum = 0.0;
        for (double val : sorted) {
            const double diff = val - metrics.mean_ms;
            variance_sum += diff * diff;
        }
        metrics.stddev_ms = std::sqrt(variance_sum / static_cast<double>(n - 1));
    } else {
        metrics.stddev_ms = 0.0;
    }

    // Speedup and Efficiency
    if (baseline_mean_ms > 0.0) {
        metrics.speedup = calculate_speedup(baseline_mean_ms, metrics.median_ms);
        metrics.efficiency = calculate_efficiency(metrics.speedup, workers);
    } else {
        metrics.speedup = 1.0;
        metrics.efficiency = 1.0;
    }

    // Throughput (e.g. ops / second or GFLOPS)
    if (total_ops > 0.0 && metrics.median_ms > 0.0) {
        const double seconds = metrics.median_ms / 1000.0;
        metrics.throughput = total_ops / seconds;
    }

    return metrics;
}

} // namespace parallelforge::benchmark
