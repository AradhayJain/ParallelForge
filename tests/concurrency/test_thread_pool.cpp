#include <gtest/gtest.h>
#include <parallelforge/threading/thread_pool.hpp>
#include <parallelforge/threading/parallel_for.hpp>
#include <parallelforge/core/error.hpp>

#include <vector>
#include <atomic>
#include <chrono>

using namespace parallelforge::threading;
using namespace parallelforge::core;

TEST(ThreadPoolTest, BasicTaskExecutionAndFuture) {
    ThreadPool pool(4);
    EXPECT_EQ(pool.worker_count(), 4);

    auto fut = pool.submit([](int a, int b) {
        return a * b;
    }, 6, 7);

    EXPECT_EQ(fut.get(), 42);
}

TEST(ThreadPoolTest, ExceptionPropagationThroughFuture) {
    ThreadPool pool(2);

    auto fut = pool.submit([]() -> int {
        throw std::invalid_argument("Worker error payload");
    });

    EXPECT_THROW({
        try {
            (void)fut.get();
        } catch (const std::invalid_argument& ex) {
            EXPECT_STREQ(ex.what(), "Worker error payload");
            throw;
        }
    }, std::invalid_argument);
}

TEST(ThreadPoolTest, GracefulShutdownDrainsTasks) {
    std::atomic<int> counter{0};
    {
        ThreadPool pool(4);
        for (int i = 0; i < 100; ++i) {
            pool.submit([&counter]() {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                counter++;
            });
        }
        // Destructor calls shutdown(), which drains all 100 tasks before joining
    }
    EXPECT_EQ(counter.load(), 100);
}

TEST(ThreadPoolTest, SubmitAfterShutdownThrows) {
    ThreadPool pool(2);
    pool.shutdown();
    EXPECT_TRUE(pool.is_shutdown());

    EXPECT_THROW(pool.submit([]() { return 1; }), QueueClosedException);
}

TEST(ThreadPoolTest, MultiProducerConcurrentSubmissions) {
    ThreadPool pool(4);
    const int num_producers = 4;
    const int tasks_per_producer = 500;
    std::atomic<uint64_t> completed_tasks{0};

    std::vector<std::thread> producers;
    producers.reserve(num_producers);

    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&pool, &completed_tasks, tasks_per_producer]() {
            std::vector<std::future<void>> local_futs;
            local_futs.reserve(tasks_per_producer);

            for (int i = 0; i < tasks_per_producer; ++i) {
                local_futs.push_back(pool.submit([&completed_tasks]() {
                    completed_tasks++;
                }));
            }

            for (auto& f : local_futs) {
                f.get();
            }
        });
    }

    for (auto& p : producers) {
        p.join();
    }

    EXPECT_EQ(completed_tasks.load(), static_cast<uint64_t>(num_producers * tasks_per_producer));
}

TEST(ThreadPoolTest, ParallelForUtility) {
    const std::size_t N = 10000;
    std::vector<int> data(N, 0);

    parallel_for(0, N, [&data](std::size_t i) {
        data[i] = static_cast<int>(i * 2);
    }, 100, 4);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_EQ(data[i], static_cast<int>(i * 2));
    }
}
