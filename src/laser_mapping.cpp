#include "laser_mapping.h"
#include "lidarFactor.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/ceres.h>
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
        if (++frameCount_ % 2 == 0) {
            MapResult mr;
            mr.pose = odom.pose;
            {
                std::lock_guard<std::mutex> lock(mapMutex_);
                mr.mapCloud.reset(new pcl::PointCloud<pcl::PointXYZI>(*localMap_));
            }
            std::cout << "[LaserMappingThread] pushing MapResult, map size: " << mr.mapCloud->size() << std::endl;
            outputQueue_.push(mr);
        }
    }

    std::cout << "[LaserMappingThread] Stopped." << std::endl;
}

static inline void eigenPoseFromMatrix4f(const Eigen::Matrix4f& T, double* q, double* t) {
    Eigen::Matrix3f R = T.block<3,3>(0,0);
    Eigen::Quaternionf qf(R);
    // ceres expects (x,y,z,w) order in this codebase
    q[0] = qf.x(); q[1] = qf.y(); q[2] = qf.z(); q[3] = qf.w();
    t[0] = T(0,3); t[1] = T(1,3); t[2] = T(2,3);
}

void LaserMappingThread::integrateFrame(const OdomResult& odom) {
    // 如果没有点云，直接返回
    if (!odom.registeredCloud || odom.registeredCloud->empty()) return;

    // 如果地图为空，直接将当前帧变换后加入地图并建立 kd-tree
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        if (localMap_->empty()) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZI>());
            Eigen::Matrix4f T = odom.pose;
            pcl::transformPointCloud(*odom.registeredCloud, *transformed, T);
            *localMap_ += *transformed;
            mapKdTree_->setInputCloud(localMap_);
            return;
        }
    }

    // 扫描到地图的配准：以 odom.pose 作为初值，在 localMap_ 上进行特征级匹配并用 Ceres 优化位姿
    // 准备 Ceres 参数：四元数 (x,y,z,w) 与 平移 t
    double q_init[4];
    double t_init[3];
    eigenPoseFromMatrix4f(odom.pose, q_init, t_init);

    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold();
    problem.AddParameterBlock(q_init, 4, quaternion_manifold);
    problem.AddParameterBlock(t_init, 3);

    // Build KD-tree copy under lock
    pcl::PointCloud<pcl::PointXYZI>::Ptr mapCopy(new pcl::PointCloud<pcl::PointXYZI>());
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        *mapCopy = *localMap_;
    }
    pcl::KdTreeFLANN<pcl::PointXYZI> mapTree;
    mapTree.setInputCloud(mapCopy);

    // For each point in current registeredCloud, search neighbors in map and add residuals
    const int kSearch = 5;
    int added_residuals = 0;
    const int max_residuals = 2000; // limit to control solve time

    for (const auto& p : odom.registeredCloud->points) {
        if (added_residuals >= max_residuals) break;

        std::vector<int> nnIdx(kSearch);
        std::vector<float> nnDist(kSearch);
        int found = mapTree.nearestKSearch(p, kSearch, nnIdx, nnDist);
        if (found < 2) continue;
        if (nnDist[0] > 25.0f) continue; // skip far points

        // Try plane first if at least 3 neighbors
        if (found >= 3) {
            Eigen::Vector3d a(mapCopy->points[nnIdx[0]].x, mapCopy->points[nnIdx[0]].y, mapCopy->points[nnIdx[0]].z);
            Eigen::Vector3d b(mapCopy->points[nnIdx[1]].x, mapCopy->points[nnIdx[1]].y, mapCopy->points[nnIdx[1]].z);
            Eigen::Vector3d c(mapCopy->points[nnIdx[2]].x, mapCopy->points[nnIdx[2]].y, mapCopy->points[nnIdx[2]].z);
            // check non-collinear
            Eigen::Vector3d normal = (b - a).cross(c - a);
            double norm = normal.norm();
            if (norm > 1e-3) {
                Eigen::Vector3d curr(p.x, p.y, p.z);
                ceres::CostFunction* cf = LidarPlaneFactor::Create(curr, a, b, c);
                problem.AddResidualBlock(cf, loss_function, q_init, t_init);
                added_residuals++;
                continue;
            }
        }

        // Fallback to edge with two nearest neighbors
        if (found >= 2) {
            Eigen::Vector3d a(mapCopy->points[nnIdx[0]].x, mapCopy->points[nnIdx[0]].y, mapCopy->points[nnIdx[0]].z);
            Eigen::Vector3d b(mapCopy->points[nnIdx[1]].x, mapCopy->points[nnIdx[1]].y, mapCopy->points[nnIdx[1]].z);
            Eigen::Vector3d curr(p.x, p.y, p.z);
            ceres::CostFunction* cf = LidarEdgeFactor::Create(curr, a, b);
            problem.AddResidualBlock(cf, loss_function, q_init, t_init);
            added_residuals++;
        }
    }

    if (added_residuals == 0) {
        // 没有添加约束，直接把 odom.pose 的变换结果加入地图
        pcl::PointCloud<pcl::PointXYZI>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*odom.registeredCloud, *transformed, odom.pose);
        std::lock_guard<std::mutex> lock(mapMutex_);
        *localMap_ += *transformed;
        mapKdTree_->setInputCloud(localMap_);
        return;
    }

    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 20;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Convert solution to Eigen::Matrix4f
    Eigen::Quaterniond q_sol(q_init[3], q_init[0], q_init[1], q_init[2]);
    Eigen::Matrix3d R_d = q_sol.toRotationMatrix();
    Eigen::Matrix4f T_sol = Eigen::Matrix4f::Identity();
    T_sol.block<3,3>(0,0) = R_d.cast<float>();
    T_sol(0,3) = static_cast<float>(t_init[0]);
    T_sol(1,3) = static_cast<float>(t_init[1]);
    T_sol(2,3) = static_cast<float>(t_init[2]);

    // Transform current scan by optimized pose and insert into map
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*odom.registeredCloud, *transformed, T_sol);

    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        *localMap_ += *transformed;
        // 下采样或限制地图大小
        if (localMap_->size() > maxMapSize_) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::VoxelGrid<pcl::PointXYZI> vg;
            vg.setInputCloud(localMap_);
            vg.setLeafSize(voxelSize_, voxelSize_, voxelSize_);
            vg.filter(*filtered);
            localMap_.swap(filtered);
        }
        mapKdTree_->setInputCloud(localMap_);
    }
}
