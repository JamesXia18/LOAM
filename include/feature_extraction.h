#pragma once
#include <pcl/common/common.h>

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>


// ============ 线程安全队列模板 ============
template <typename T>
class SafeQueue {
public:
    // 将元素入队
    void push(const T& value) {
        std::unique_lock<std::mutex> lock(mtx_);
        q_.push(value);
        lock.unlock();
        cond_.notify_one();
    }

    // 将队列头部元素出队并赋值给 value
    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mtx_);
        cond_.wait(lock, [&] { return !q_.empty() || stop_; });
        if (q_.empty()) return false;
        value = std::move(q_.front());
        q_.pop();
        return true;
    }

    // 停止队列，唤醒所有等待的线程
    void stop() {
        std::unique_lock<std::mutex> lock(mtx_);
        stop_ = true;
        lock.unlock();
        cond_.notify_all();
    }

    // 判断队列是否为空
    bool empty() {
        std::lock_guard<std::mutex> lock(mtx_);
        return q_.empty();
    }

private:
    std::queue<T> q_; // 底层存储元素的队列
    std::mutex mtx_; // 互斥锁
    std::condition_variable cond_; // 条件变量，用于线程间的等待/通知机制
    bool stop_ = false; // 标记队列是否停止
};


struct FeatureCloud {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerSharp;      // 尖锐角点
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerLessSharp;  // 次尖锐角点
    pcl::PointCloud<pcl::PointXYZI>::Ptr surfFlat;         // 平面点
    pcl::PointCloud<pcl::PointXYZI>::Ptr surfLessFlat;     // 普通面点
};

std::vector<float> read_lidar_data(const std::string& lidar_data_path);

FeatureCloud extractFeatures(const pcl::PointCloud<pcl::PointXYZI>::Ptr& laserCloudIn);


// ============ FeatureExtractionThread 类 ============
class FeatureExtractionThread {
public:
    FeatureExtractionThread(SafeQueue<pcl::PointCloud<pcl::PointXYZI>::Ptr>& inputQueue,
        SafeQueue<FeatureCloud>& outputQueue);
    ~FeatureExtractionThread();

    void start();
    void stop();

private:
    void processLoop();

    SafeQueue<pcl::PointCloud<pcl::PointXYZI>::Ptr>& inputQueue_;
    SafeQueue<FeatureCloud>& outputQueue_;
    std::thread worker_;
    std::atomic<bool> running_{ false };
};
