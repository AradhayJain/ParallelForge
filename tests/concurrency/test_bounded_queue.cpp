#include <gtest/gtest.h>
#include <parallelforge/threading/bounded_queue.hpp>
#include <parallelforge/core/error.hpp>

#include <thread>
#include <vector>
#include <atomic>
#include <numeric>
#include <future>
#include <chrono>

using namespace parallelforge::threading;
using namespace parallelforge::core;

TEST(BoundedQueueTest, CapacityAndTryPush) {
    BoundedQueue<int> q(2);
    EXPECT_EQ(q.capacity(), 2);
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0);

    EXPECT_TRUE(q.try_push(1));
    EXPECT_TRUE(q.try_push(2));
    EXPECT_FALSE(q.try_push(3)); // Queue is full!
    EXPECT_EQ(q.size(), 2);

    auto item = q.try_pop();
    ASSERT_TRUE(item.has_value());
    EXPECT_EQ(*item, 1);
    EXPECT_EQ(q.size(), 1);

    EXPECT_TRUE(q.try_push(4)); // Now space is available
    EXPECT_EQ(q.size(), 2);
}

TEST(BoundedQueueTest, ShutdownWhileIdle) {
    BoundedQueue<int> q(5);
    EXPECT_FALSE(q.is_closed());
    q.close();
    EXPECT_TRUE(q.is_closed());

    EXPECT_THROW(q.push(10), QueueClosedException);
    EXPECT_FALSE(q.try_push(20));

    auto val = q.pop();
    EXPECT_FALSE(val.has_value());
}

TEST(BoundedQueueTest, ShutdownWithPendingWorkDrainsCompletely) {
    BoundedQueue<int> q(5);
    q.push(1);
    q.push(2);
    q.push(3);

    q.close();
    EXPECT_TRUE(q.is_closed());
    EXPECT_EQ(q.size(), 3);

    // Consumers can still drain all pending items
    auto v1 = q.pop();
    ASSERT_TRUE(v1.has_value());
    EXPECT_EQ(*v1, 1);

    auto v2 = q.pop();
    ASSERT_TRUE(v2.has_value());
    EXPECT_EQ(*v2, 2);

    auto v3 = q.pop();
    ASSERT_TRUE(v3.has_value());
    EXPECT_EQ(*v3, 3);

    // Once drained and closed, returns nullopt
    auto v4 = q.pop();
    EXPECT_FALSE(v4.has_value());
}

TEST(BoundedQueueTest, MultiProducerMultiConsumerStressTest) {
    const unsigned int num_producers = 4;
    const unsigned int num_consumers = 4;
    const std::size_t items_per_producer = 10000;
    const std::size_t total_items = num_producers * items_per_producer;

    BoundedQueue<std::size_t> q(50); // Small bounded capacity to force frequent blocking & wakeups

    std::atomic<std::size_t> items_consumed{0};
    std::atomic<uint64_t> sum_consumed{0};

    // Consumers
    std::vector<std::thread> consumers;
    consumers.reserve(num_consumers);
    for (unsigned int c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&q, &items_consumed, &sum_consumed]() {
            while (auto val = q.pop()) {
                items_consumed++;
                sum_consumed += *val;
            }
        });
    }

    // Producers
    std::vector<std::thread> producers;
    producers.reserve(num_producers);
    for (unsigned int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&q, p, items_per_producer]() {
            for (std::size_t i = 0; i < items_per_producer; ++i) {
                q.push(p * items_per_producer + i);
            }
        });
    }

    for (auto& p : producers) {
        p.join();
    }
    q.close();

    for (auto& c : consumers) {
        c.join();
    }

    EXPECT_EQ(items_consumed.load(), total_items);
    // Sum of 0 to (total_items - 1) = (N - 1) * N / 2
    const uint64_t expected_sum = static_cast<uint64_t>(total_items - 1) * static_cast<uint64_t>(total_items) / 2;
    EXPECT_EQ(sum_consumed.load(), expected_sum);
}
