#ifndef LASER_MAPPING_THREAD_H
#define LASER_MAPPING_THREAD_H

#include <thread>
#include <atomic>
#include <mutex>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "feature_extraction.h" // SafeQueue, FeatureCloud
#include "laser_odometry.h"     // OdomResult

// 输出结构：mapping 后的位姿与地图快照（可选）
struct MapResult {
    Eigen::Matrix4f pose; // 当前（累计/优化后）的位姿
    pcl::PointCloud<pcl::PointXYZI>::Ptr mapCloud; // 局部地图快照
};

class LaserMappingThread {
public:
    LaserMappingThread(SafeQueue<OdomResult>& inputQueue,
        SafeQueue<MapResult>& outputQueue,
        float voxel_size = 0.3f,
        size_t max_map_size = 500000);
    ~LaserMappingThread();

    void start();
    void stop();

private:
    void processLoop();
    void integrateFrame(const OdomResult& odom);

    SafeQueue<OdomResult>& inputQueue_;
    SafeQueue<MapResult>& outputQueue_;
    std::thread worker_;
    std::atomic<bool> running_{ false };

    // 地图维护
    pcl::PointCloud<pcl::PointXYZI>::Ptr localMap_; // 局部/全局地图
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr mapKdTree_;
    std::mutex mapMutex_;

    // 下采样参数与策略
    float voxelSize_;
    size_t maxMapSize_; // 地图最大点数，超过则裁剪或下采样
    int frameCount_;
};

#endif
