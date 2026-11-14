// 无模型回归：HTTP 预处理跨过 idle deadline 时不得释放模型资源。
#include "server-queue.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char ** argv) {
    using namespace std::chrono_literals;
    server_queue queue;
    std::atomic<int> sleeps{0};
    queue.on_new_task([](server_task &&, bool) { return true; });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([&](bool sleeping) { if (sleeping) ++sleeps; });
    const int idle_ms = argc > 1 ? std::stoi(argv[1]) : 50;
    std::thread loop([&] { queue.start_loop(idle_ms); });
    // 等待第一次休眠，模拟请求唤醒后耗时的模板处理和分词。
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!queue.is_sleeping() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(10ms);
    if (!queue.is_sleeping()) {
        queue.terminate();
        loop.join();
        std::cerr << "initial sleep deadline expired\n";
        return 1;
    }
#ifdef FIXED_REQUEST_GUARD
    queue.acquire_model_request();
    queue.acquire_model_request(); // 嵌套 response generator 不能提前放开保护。
    queue.release_model_request();
#else
    queue.wait_until_no_sleep();
#endif
    const int baseline = sleeps.load();
    std::this_thread::sleep_for(1300ms);
    const bool protected_request = sleeps.load() == baseline;
#ifdef FIXED_REQUEST_GUARD
    queue.release_model_request();
#endif
    const auto release_deadline = std::chrono::steady_clock::now() + 3s;
    while (!queue.is_sleeping() && std::chrono::steady_clock::now() < release_deadline) std::this_thread::sleep_for(10ms);
    const bool idle_released = queue.is_sleeping();
    queue.terminate();
    loop.join();
    if (!protected_request || !idle_released) {
        std::cerr << "protected_request=" << protected_request << " idle_released=" << idle_released << '\n';
        return 1;
    }
    std::cout << "request protected; nested guard released; idle sleep restored\n";
}
