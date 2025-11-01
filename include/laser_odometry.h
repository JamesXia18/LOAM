#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "feature_extraction.h" // 包含 SafeQueue & FeatureCloud

// ================== 位姿结构体 ==================
struct OdomResult {
    Eigen::Matrix4f pose; // 世界坐标下的当前帧位姿
    pcl::PointCloud<pcl::PointXYZI>::Ptr registeredCloud; // 可视化/发布点云
};

// ================== LaserOdometryThread ==================
class LaserOdometryThread {
public:
    LaserOdometryThread(SafeQueue<FeatureCloud>& inputQueue,
        SafeQueue<OdomResult>& outputQueue);
    ~LaserOdometryThread();

    void start();
    void stop();

private:
    void processLoop();
    void matchAndOptimize(const FeatureCloud& currFrame);

    SafeQueue<FeatureCloud>& inputQueue_;
    SafeQueue<OdomResult>& outputQueue_;
    std::thread worker_;
    std::atomic<bool> running_{ false };

    // 上一帧特征
    pcl::PointCloud<pcl::PointXYZI>::Ptr lastCorner_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr lastSurf_;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCorner_;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurf_;

    Eigen::Matrix4f poseWorld_; // 当前累计位姿
};