#include <parallelforge/threading/thread_pool.hpp>

#include <algorithm>

namespace parallelforge::threading {

ThreadPool::ThreadPool(std::size_t num_workers) {
    if (num_workers == 0) {
        const unsigned int hw = std::thread::hardware_concurrency();
        num_workers = hw > 0 ? hw : 4;
    }

    workers_.reserve(num_workers);
    for (std::size_t i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return stop_immediate_ || (shutdown_requested_ && tasks_.empty()) || !tasks_.empty();
            });

            if (stop_immediate_ || (shutdown_requested_ && tasks_.empty())) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        // Execute task outside the lock so exceptions inside task are contained within its packaged_task
        if (task) {
            try {
                task();
            } catch (...) {
                // packaged_task catches and stores exceptions into std::future automatically.
                // Any raw task exception is swallowed here to prevent worker thread abort.
            }
        }
    }
}

void ThreadPool::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_requested_) {
            return;
        }
        shutdown_requested_ = true;
    }
    cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_immediate_) {
            return;
        }
        stop_immediate_ = true;
        shutdown_requested_ = true;
        // Drain pending unstarted tasks
        std::queue<std::function<void()>> empty;
        std::swap(tasks_, empty);
    }
    cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

std::size_t ThreadPool::pending_tasks() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.size();
}

bool ThreadPool::is_shutdown() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_requested_;
}

} // namespace parallelforge::threading
