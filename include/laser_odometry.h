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

#include "feature_extraction.h" // ���� SafeQueue & FeatureCloud

struct OdomResult {
    Eigen::Matrix4f pose;

    int cornerCount = 0;
    int planeCount = 0;

    pcl::PointCloud<pcl::PointXYZI>::Ptr lastCorner_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr lastSurf_;

    OdomResult() {
        lastCorner_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        lastSurf_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    }
};

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

	pcl::PointCloud<pcl::PointXYZI>::Ptr lastCorner_;
	pcl::PointCloud<pcl::PointXYZI>::Ptr lastSurf_;
	pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCorner_;
	pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurf_;

    Eigen::Matrix4f poseWorld_;
};