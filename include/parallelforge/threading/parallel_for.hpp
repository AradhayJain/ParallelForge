#pragma once

#include <parallelforge/threading/thread_pool.hpp>

#include <cstddef>
#include <functional>
#include <algorithm>
#include <vector>
#include <thread>
#include <future>

namespace parallelforge::threading {

/**
 * @brief Executes func(i) for all indices in [begin, end) partitioned across worker threads.
 */
template <typename Func>
void parallel_for(std::size_t begin, 
                  std::size_t end, 
                  Func func, 
                  std::size_t chunk_size = 0, 
                  unsigned int threads = 0) {
    if (begin >= end) {
        return;
    }
    const std::size_t count = end - begin;

    unsigned int worker_count = threads > 0 ? threads : std::thread::hardware_concurrency();
    if (worker_count == 0) worker_count = 4;
    worker_count = std::min<unsigned int>(worker_count, static_cast<unsigned int>(count));

    if (worker_count <= 1) {
        for (std::size_t i = begin; i < end; ++i) {
            func(i);
        }
        return;
    }

    if (chunk_size == 0) {
        chunk_size = (count + worker_count - 1) / worker_count;
    }

    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (unsigned int w = 0; w < worker_count; ++w) {
        const std::size_t chunk_start = begin + w * chunk_size;
        const std::size_t chunk_end = std::min(chunk_start + chunk_size, end);
        if (chunk_start >= end) {
            break;
        }

        workers.emplace_back([chunk_start, chunk_end, &func]() {
            for (std::size_t i = chunk_start; i < chunk_end; ++i) {
                func(i);
            }
        });
    }

    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
}

/**
 * @brief Executes parallel loop using a shared ThreadPool instance.
 */
template <typename Func>
void parallel_for(ThreadPool& pool, 
                  std::size_t begin, 
                  std::size_t end, 
                  Func func, 
                  std::size_t chunk_size = 0) {
    if (begin >= end) {
        return;
    }
    const std::size_t count = end - begin;
    const std::size_t workers = std::max<std::size_t>(1, pool.worker_count());

    if (chunk_size == 0) {
        chunk_size = std::max<std::size_t>(1, (count + workers - 1) / workers);
    }

    const std::size_t num_chunks = (count + chunk_size - 1) / chunk_size;
    std::vector<std::future<void>> futures;
    futures.reserve(num_chunks);

    for (std::size_t c = 0; c < num_chunks; ++c) {
        const std::size_t chunk_start = begin + c * chunk_size;
        const std::size_t chunk_end = std::min(chunk_start + chunk_size, end);

        futures.push_back(pool.submit([chunk_start, chunk_end, &func]() {
            for (std::size_t i = chunk_start; i < chunk_end; ++i) {
                func(i);
            }
        }));
    }

    for (auto& fut : futures) {
        fut.get();
    }
}

} // namespace parallelforge::threading
