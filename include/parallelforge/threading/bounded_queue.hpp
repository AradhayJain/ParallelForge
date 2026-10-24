#pragma once

#include <parallelforge/core/error.hpp>

#include <cstddef>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <optional>
#include <chrono>

namespace parallelforge::threading {

/**
 * @brief Thread-safe bounded producer-consumer queue with clean closure semantics.
 * 
 * Guaranteeing:
 * 1. Bounded capacity: prevents memory exhaustion.
 * 2. No busy waiting: threads sleep on condition variables.
 * 3. No lost wakeups: separate cvs for not-empty and not-full conditions.
 * 4. Safe shutdown: close() unblocks all waiting threads deterministically.
 */
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity)
        : capacity_(capacity > 0 ? capacity : 1), closed_(false) {}

    ~BoundedQueue() {
        close();
    }

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;
    BoundedQueue(BoundedQueue&&) = delete;
    BoundedQueue& operator=(BoundedQueue&&) = delete;

    /**
     * @brief Pushes an item into the queue. Blocks if full until space is available.
     * @throws core::QueueClosedException if the queue is closed or closed while waiting.
     */
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_cv_.wait(lock, [this]() {
            return closed_ || queue_.size() < capacity_;
        });

        if (closed_) {
            throw core::QueueClosedException("Cannot push to closed queue");
        }

        queue_.push(std::move(item));
        not_empty_cv_.notify_one();
    }

    /**
     * @brief Attempts to push without blocking.
     * @return true if pushed, false if full or closed.
     */
    bool try_push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || queue_.size() >= capacity_) {
            return false;
        }
        queue_.push(std::move(item));
        not_empty_cv_.notify_one();
        return true;
    }

    /**
     * @brief Pops an item from the queue. Blocks if empty until an item is available.
     * @return std::optional containing the popped item, or std::nullopt if the queue is closed and empty.
     */
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_cv_.wait(lock, [this]() {
            return closed_ || !queue_.empty();
        });

        if (queue_.empty()) {
            return std::nullopt; // Closed and fully drained
        }

        T item = std::move(queue_.front());
        queue_.pop();
        not_full_cv_.notify_one();
        return item;
    }

    /**
     * @brief Attempts to pop an item without blocking.
     * @return std::optional containing the popped item, or std::nullopt if empty or closed.
     */
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_cv_.notify_one();
        return item;
    }

    /**
     * @brief Closes the queue, preventing new pushes and waking all waiting threads.
     * Pending items can still be drained by consumers.
     */
    void close() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            closed_ = true;
        }
        not_full_cv_.notify_all();
        not_empty_cv_.notify_all();
    }

    [[nodiscard]] bool is_closed() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return capacity_;
    }

private:
    const std::size_t capacity_;
    bool closed_{false};
    mutable std::mutex mutex_;
    std::condition_variable not_empty_cv_;
    std::condition_variable not_full_cv_;
    std::queue<T> queue_;
};

} // namespace parallelforge::threading
