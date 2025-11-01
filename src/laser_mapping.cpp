#include "laser_mapping.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <iostream>
#include <chrono>

LaserMappingThread::LaserMappingThread(SafeQueue<OdomResult>& inputQueue,
    SafeQueue<MapResult>& outputQueue,
    float voxel_size,
    size_t max_map_size)
    : inputQueue_(inputQueue),
    outputQueue_(outputQueue),
    voxelSize_(voxel_size),
    maxMapSize_(max_map_size),
    frameCount_(0)
{
    localMap_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    mapKdTree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
}

LaserMappingThread::~LaserMappingThread() {
    stop();
}

void LaserMappingThread::start() {
    running_ = true;
    worker_ = std::thread(&LaserMappingThread::processLoop, this);
}

void LaserMappingThread::stop() {
    running_ = false;
    inputQueue_.stop();
    if (worker_.joinable()) worker_.join();
}

void LaserMappingThread::processLoop() {
    std::cout << "[LaserMappingThread] Started." << std::endl;

    while (running_) {
        OdomResult odom;
        if (!inputQueue_.pop(odom)) {
            if (!running_) break;
            continue;
        }

        auto t0 = std::chrono::steady_clock::now();
        integrateFrame(odom);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "[LaserMappingThread] Integrated frame in " << ms << " ms. Map size: " << localMap_->size() << std::endl;

        // 可选：每若干帧输出一个地图快照
        if (++frameCount_ % 5 == 0) {
            MapResult mr;
            mr.pose = odom.pose;
            {
                std::lock_guard<std::mutex> lock(mapMutex_);
                mr.mapCloud.reset(new pcl::PointCloud<pcl::PointXYZI>(*localMap_));
            }
            outputQueue_.push(mr);
        }
    }

    std::cout << "[LaserMappingThread] Stopped." << std::endl;
}

void LaserMappingThread::integrateFrame(const OdomResult& odom) {
    // 把当前帧点云从传入坐标系（假设是map坐标系或里程计里程计估计基准）变换到地图坐标系
    // 这里假设 odom.registeredCloud 已是当前帧在里程计估计下的点云（未变换到世界系）
    // 我们把点云按 odom.pose (4x4) 变换到地图坐标系：p_map = pose * p_local
    if (!odom.registeredCloud || odom.registeredCloud->empty()) return;

    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Matrix4f T = odom.pose; // pose 把点从局部（sensor/odom）变换到世界/地图
    pcl::transformPointCloud(*odom.registeredCloud, *transformed, T);

    // 将变换后的点并入 localMap
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        // 直接 append（注意内存增长）：可根据实际需求改成基于 voxel map 或八叉树的管理
        *localMap_ += *transformed;
    }

    // 下采样以控制地图大小与去噪
    if (localMap_->size() > maxMapSize_ || (frameCount_ % 10 == 0)) {
        std::lock_guard<std::mutex> lock(mapMutex_);
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::VoxelGrid<pcl::PointXYZI> vg;
        vg.setInputCloud(localMap_);
        vg.setLeafSize(voxelSize_, voxelSize_, voxelSize_);
        vg.filter(*filtered);

        localMap_.swap(filtered);
        // 更新 kd-tree
        mapKdTree_->setInputCloud(localMap_);
    }
    else {
        // 如果未下采样，也可以逐渐插入 kd-tree 的点（这里我们每次都更新，简单但开销略大）
        std::lock_guard<std::mutex> lock(mapMutex_);
        mapKdTree_->setInputCloud(localMap_);
    }

    // （可扩展）若要执行 scan-to-map 优化，可以：
    // 1) 使用当前帧的 corner/surf 特征在 mapKdTree_ 中搜索对应点
    // 2) 构建残差并优化当前帧 pose（像 LaserOdometry 的 matchAndOptimize）
    // 3) 用优化后的 pose 将点云变换再加入地图（或用滑窗策略）
}
