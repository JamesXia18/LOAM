#include "laser_odometry.h"

#include <iostream>
#include <chrono>
#include <cmath>

LaserOdometryThread::LaserOdometryThread(
    SafeQueue<FeatureCloud>& inputQueue,
    SafeQueue<OdomResult>& outputQueue)
    : inputQueue_(inputQueue), outputQueue_(outputQueue)
{
    lastCorner_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    lastSurf_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    kdtreeCorner_.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    kdtreeSurf_.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    poseWorld_.setIdentity();
}

LaserOdometryThread::~LaserOdometryThread() {
    stop();
}

void LaserOdometryThread::start() {
    running_ = true;
    worker_ = std::thread(&LaserOdometryThread::processLoop, this);
}

void LaserOdometryThread::stop() {
    running_ = false;
    inputQueue_.stop();
    if (worker_.joinable())
        worker_.join();
}

void LaserOdometryThread::processLoop() {
    std::cout << "[LaserOdometryThread] Started." << std::endl;

    while (running_) {
        FeatureCloud currFeatures;
        if (!inputQueue_.pop(currFeatures)) {
            if (!running_) break;
            continue;
        }

        if (currFeatures.cornerLessSharp->empty() || currFeatures.surfLessFlat->empty()) {
            std::cerr << "[LaserOdometryThread] Empty feature cloud, skip frame." << std::endl;
            continue;
        }

        auto start = std::chrono::steady_clock::now();

        matchAndOptimize(currFeatures);

        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[LaserOdometryThread] Frame processed in " << ms << " ms." << std::endl;
    }

    std::cout << "[LaserOdometryThread] Stopped." << std::endl;
}

// ================== 核心：帧间匹配与位姿优化 ==================
void LaserOdometryThread::matchAndOptimize(const FeatureCloud& currFrame)
{
    // 初帧直接建立KD-Tree
    if (lastCorner_->empty()) {
        *lastCorner_ = *currFrame.cornerLessSharp;
        *lastSurf_ = *currFrame.surfLessFlat;
        kdtreeCorner_->setInputCloud(lastCorner_);
        kdtreeSurf_->setInputCloud(lastSurf_);
        return;
    }

    // 构建kd-tree以进行最近邻搜索
    kdtreeCorner_->setInputCloud(lastCorner_);
    kdtreeSurf_->setInputCloud(lastSurf_);

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

    // 简化版：使用最近邻ICP匹配
    int iter = 0;
    for (; iter < 5; ++iter) {
        std::vector<Eigen::Vector3f> srcPts;
        std::vector<Eigen::Vector3f> tgtPts;

        // 角点匹配
        for (auto& p : currFrame.cornerLessSharp->points) {
            std::vector<int> idx;
            std::vector<float> dist;
            if (kdtreeCorner_->nearestKSearch(p, 1, idx, dist) > 0 && dist[0] < 0.5) {
                Eigen::Vector3f ps(p.x, p.y, p.z);
                Eigen::Vector3f pt(lastCorner_->points[idx[0]].x, lastCorner_->points[idx[0]].y, lastCorner_->points[idx[0]].z);
                srcPts.push_back(ps);
                tgtPts.push_back(pt);
            }
        }

        // 面点匹配
        for (auto& p : currFrame.surfLessFlat->points) {
            std::vector<int> idx;
            std::vector<float> dist;
            if (kdtreeSurf_->nearestKSearch(p, 1, idx, dist) > 0 && dist[0] < 0.5) {
                Eigen::Vector3f ps(p.x, p.y, p.z);
                Eigen::Vector3f pt(lastSurf_->points[idx[0]].x, lastSurf_->points[idx[0]].y, lastSurf_->points[idx[0]].z);
                srcPts.push_back(ps);
                tgtPts.push_back(pt);
            }
        }

        if (srcPts.size() < 10) {
            std::cout << "[LaserOdometry] Not enough correspondences, skip frame." << std::endl;
            return;
        }

        // 计算质心
        Eigen::Vector3f srcMean = Eigen::Vector3f::Zero();
        Eigen::Vector3f tgtMean = Eigen::Vector3f::Zero();
        for (size_t i = 0; i < srcPts.size(); ++i) {
            srcMean += srcPts[i];
            tgtMean += tgtPts[i];
        }
        srcMean /= srcPts.size();
        tgtMean /= tgtPts.size();

        // 去均值
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        for (size_t i = 0; i < srcPts.size(); ++i) {
            H += (srcPts[i] - srcMean) * (tgtPts[i] - tgtMean).transpose();
        }

        // SVD 解旋转矩阵
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
        if (R.determinant() < 0) {
            Eigen::Matrix3f V = svd.matrixV();
            V.col(2) *= -1;
            R = V * svd.matrixU().transpose();
        }
        Eigen::Vector3f t = tgtMean - R * srcMean;

        // 更新增量
        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3, 3>(0, 0) = R;
        delta.block<3, 1>(0, 3) = t;

        T = delta * T;

        // 判断收敛
        if (t.norm() < 0.001 && (Eigen::AngleAxisf(R)).angle() < 0.001)
            break;
    }

    poseWorld_ = poseWorld_ * T;

    // 保存当前帧作为下一帧参考
    *lastCorner_ = *currFrame.cornerLessSharp;
    *lastSurf_ = *currFrame.surfLessFlat;

    // 输出
    OdomResult result;
    result.pose = poseWorld_;
    result.registeredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    *result.registeredCloud = *currFrame.surfLessFlat + *currFrame.cornerLessSharp;

    outputQueue_.push(result);

    std::cout << "[LaserOdometry] Iter=" << iter
        << ", pose: \n" << poseWorld_ << std::endl;
}
