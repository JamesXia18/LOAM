#pragma once
#include <pcl/common/common.h>

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

#include "common.hpp"

FeatureCloud extractFeatures(const pcl::PointCloud<pcl::PointXYZI>::Ptr& laserCloudIn);

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
