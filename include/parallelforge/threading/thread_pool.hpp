#pragma once

#include <parallelforge/core/error.hpp>

#include <cstddef>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <type_traits>

namespace parallelforge::threading {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_workers = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    /**
     * @brief Submits a callable task for asynchronous execution.
     * @return std::future holding the eventual result or propagated exception.
     * @throws core::QueueClosedException if the pool has been shut down.
     */
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) 
        -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>> {
        using ReturnType = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;

        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            [func = std::forward<F>(f), 
             ...captured_args = std::forward<Args>(args)]() mutable -> ReturnType {
                return std::invoke(func, std::move(captured_args)...);
            }
        );

        std::future<ReturnType> fut = task->get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (shutdown_requested_) {
                throw core::QueueClosedException("Cannot submit task to stopped thread pool");
            }
            tasks_.emplace([task]() {
                (*task)();
            });
        }
        cv_.notify_one();
        return fut;
    }

    /**
     * @brief Drains all currently queued tasks before terminating worker threads.
     */
    void shutdown();

    /**
     * @brief Clears pending tasks and signals workers to terminate immediately after current tasks.
     */
    void stop();

    [[nodiscard]] std::size_t worker_count() const noexcept { return workers_.size(); }
    [[nodiscard]] std::size_t pending_tasks() const noexcept;
    [[nodiscard]] bool is_shutdown() const noexcept;

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool shutdown_requested_{false};
    bool stop_immediate_{false};

    void worker_loop();
};

} // namespace parallelforge::threading
