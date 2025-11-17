#pragma once

#include <thread>
#include <atomic>
#include <mutex>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "common.hpp"
#include "laser_odometry.h"     // OdomResult

struct MapResult {
    Eigen::Matrix4f pose;
    pcl::PointCloud<pcl::PointXYZI>::Ptr mapCloud;
};

class LaserMappingThread {
public:
    LaserMappingThread(SafeQueue<OdomResult>& inputQueue,
        SafeQueue<MapResult>& outputQueue,
        float lineRes,
        float planeRes);
    ~LaserMappingThread();

    void start();
    void stop();

private:
    void processLoop();

    SafeQueue<OdomResult>& inputQueue_;
    SafeQueue<MapResult>& outputQueue_;

    std::thread worker_;
    std::atomic<bool> running_{ false };
    std::mutex mapMutex_;

    int laserCloudCenWidth = 10;
    int laserCloudCenHeight = 10;
    int laserCloudCenDepth = 5;
    static constexpr int laserCloudWidth = 21;
    static constexpr int laserCloudHeight = 21;
    static constexpr int laserCloudDepth = 11;
    static constexpr int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //4851

    float lineRes_;
    float planeRes_;

    double parameters_[7];// [0..3]: q(x,y,z,w), [4..6]: t
    Eigen::Map<Eigen::Quaterniond> q_w_curr_; // (x,y,z,w)
    Eigen::Map<Eigen::Vector3d>    t_w_curr_;
};
