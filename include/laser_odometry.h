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
    int cornerCount = 0;
    int planeCount = 0;
};

// ================== LaserOdometryThread ==================
class LaserOdometryThread {
public:
    LaserOdometryThread(SafeQueue<FeatureCloud>& inputQueue,
        SafeQueue<OdomResult>& outputQueue);
    ~LaserOdometryThread();

	void start(); // 启动线程
	void stop(); // 停止线程

private:
	void processLoop(); // 线程处理循环
    void matchAndOptimize(const FeatureCloud& currFrame);

    SafeQueue<FeatureCloud>& inputQueue_; // 特征提取输入队列
	SafeQueue<OdomResult>& outputQueue_; // 里程计位姿输出队列
    std::thread worker_;
    std::atomic<bool> running_{ false }; // 系统运行标志

    // 上一帧特征
	pcl::PointCloud<pcl::PointXYZI>::Ptr lastCorner_; // 上一帧角点特征
	pcl::PointCloud<pcl::PointXYZI>::Ptr lastSurf_; // 上一帧平面特征
	pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCorner_; // 上一帧角点kdtree
	pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurf_; // 上一帧平面点kdtree

    Eigen::Matrix4f poseWorld_; // 当前累计位姿
};