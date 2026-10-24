#include <parallelforge/memory/false_sharing.hpp>
#include <parallelforge/core/do_not_optimize.hpp>

#include <vector>
#include <thread>
#include <algorithm>

namespace parallelforge::memory {

namespace {

// Unpadded counter: tightly packed in memory (8 bytes each).
// Multiple threads writing adjacent elements will share the same 64-byte cache line!
struct UnpaddedCounter {
    uint64_t counter{0};
};

// Padded counter: each counter occupies its own full cache line.
// Prevents false sharing and cache line invalidation ping-pong across cores.
struct alignas(core::HARDWARE_DESTRUCTIVE_SIZE) PaddedCounter {
    uint64_t counter{0};
};

} // namespace

FalseSharingResult run_false_sharing_experiment(unsigned int threads, std::size_t iterations) {
    if (threads == 0) {
        threads = std::thread::hardware_concurrency();
        if (threads == 0) threads = 4;
    }
    if (iterations == 0) {
        iterations = 10'000'000;
    }

    // 1. Run Unpadded Experiment (Suffers False Sharing)
    std::vector<UnpaddedCounter> unpadded(threads);
    core::clobber_memory();

    const auto start_unpadded = std::chrono::steady_clock::now();
    {
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (unsigned int t = 0; t < threads; ++t) {
            workers.emplace_back([&unpadded, t, iterations]() {
                for (std::size_t i = 0; i < iterations; ++i) {
                    unpadded[t].counter++;
                }
            });
        }
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
    }
    const auto end_unpadded = std::chrono::steady_clock::now();
    for (auto& c : unpadded) {
        core::do_not_optimize(c.counter);
    }

    // 2. Run Padded Experiment (Cache Line Isolated)
    std::vector<PaddedCounter> padded(threads);
    core::clobber_memory();

    const auto start_padded = std::chrono::steady_clock::now();
    {
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (unsigned int t = 0; t < threads; ++t) {
            workers.emplace_back([&padded, t, iterations]() {
                for (std::size_t i = 0; i < iterations; ++i) {
                    padded[t].counter++;
                }
            });
        }
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
    }
    const auto end_padded = std::chrono::steady_clock::now();
    for (auto& c : padded) {
        core::do_not_optimize(c.counter);
    }

    const auto unpadded_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_unpadded - start_unpadded);
    const auto padded_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_padded - start_padded);

    uint64_t total_increments = 0;
    for (const auto& c : padded) {
        total_increments += c.counter;
    }

    FalseSharingResult res;
    res.threads = threads;
    res.iterations_per_thread = iterations;
    res.unpadded_time = unpadded_elapsed;
    res.padded_time = padded_elapsed;
    res.speedup = padded_elapsed.count() > 0 
        ? static_cast<double>(unpadded_elapsed.count()) / static_cast<double>(padded_elapsed.count()) 
        : 1.0;
    res.total_increments = total_increments;

    return res;
}

} // namespace parallelforge::memory
