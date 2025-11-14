#include "laser_odometry.h"
#include "lidarFactor.hpp"

#include <pcl/common/distances.h>

#include <iostream>
#include <chrono>
#include <cmath>

constexpr double DISTANCE_SQ_THRESHOLD = 25; //距离平方阈值
constexpr double NEARBY_SCAN = 2.5; // 认为相邻2个线束才是相邻线束

double para_q[4] = { 0, 0, 0, 1 }; // 四元数表示旋转 (x,y,z,w)
double para_t[3] = { 0, 0, 0 }; // 平移向量

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

void LaserOdometryThread::matchAndOptimize(const FeatureCloud& currFrame)
{
    // --- Step 0: 初始化上一帧 ---
    if (lastCorner_->empty()) {
        *lastCorner_ = *currFrame.cornerSharp;
        *lastSurf_ = *currFrame.surfFlat;
        kdtreeCorner_->setInputCloud(lastCorner_);
        kdtreeSurf_->setInputCloud(lastSurf_);
        std::cout << "[LaserOdometryThread] First frame initialized." << std::endl;
        return;
    }

    // --- Step 1: 建立 KD-Tree ---
    kdtreeCorner_->setInputCloud(lastCorner_);
    kdtreeSurf_->setInputCloud(lastSurf_);

    int cornerPointsSharpNum = static_cast<int>(currFrame.cornerSharp->size());
    int surfPointsFlatNum    = static_cast<int>(currFrame.surfFlat->size());
    int corner_correspondence = 0;
    int plane_correspondence  = 0;

    // --- Step 2: 迭代优化 ---
    for (int iter = 0; iter < 5; ++iter) {
        corner_correspondence = 0;
        plane_correspondence = 0;

        ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
        ceres::Manifold* q_parameterization = new ceres::EigenQuaternionManifold();
        ceres::Problem problem;

        problem.AddParameterBlock(para_q, 4, q_parameterization);
        problem.AddParameterBlock(para_t, 3);

        pcl::PointXYZI pointSel;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // --- Step 2.1 Corner 特征匹配 ---
        for (int i = 0; i < cornerPointsSharpNum; ++i) {
            pointSel = currFrame.cornerSharp->points[i];
            int found = kdtreeCorner_->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
            if (found <= 0) continue;
            if (pointSearchSqDis.empty()) continue;

            if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
                int closestPointInd = pointSearchInd[0];
                int minPointInd2 = -1;
                int closestPointScanID = static_cast<int>(lastCorner_->points[closestPointInd].intensity);
                double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;

                // 向前搜索
                for (int j = closestPointInd + 1; j < static_cast<int>(lastCorner_->points.size()); ++j) {
                    int scanID = static_cast<int>(lastCorner_->points[j].intensity);
                    if (scanID <= closestPointScanID) continue;
                    if (scanID > closestPointScanID + NEARBY_SCAN) break;

                    double dist = pcl::euclideanDistance(lastCorner_->points[j], pointSel);
                    if (dist * dist < minPointSqDis2) {
                        minPointSqDis2 = dist * dist;
                        minPointInd2 = j;
                    }
                }

                // 向后搜索
                for (int j = closestPointInd - 1; j >= 0; --j) {
                    int scanID = static_cast<int>(lastCorner_->points[j].intensity);
                    if (scanID >= closestPointScanID) continue;
                    if (scanID < closestPointScanID - NEARBY_SCAN) break;

                    double dist = pcl::euclideanDistance(lastCorner_->points[j], pointSel);
                    if (dist * dist < minPointSqDis2) {
                        minPointSqDis2 = dist * dist;
                        minPointInd2 = j;
                    }
                }

                // 构建残差项
                if (minPointInd2 >= 0) {
                    Eigen::Vector3d curr_point(pointSel.x, pointSel.y, pointSel.z);
                    Eigen::Vector3d last_point_a(
                        lastCorner_->points[closestPointInd].x,
                        lastCorner_->points[closestPointInd].y,
                        lastCorner_->points[closestPointInd].z);
                    Eigen::Vector3d last_point_b(
                        lastCorner_->points[minPointInd2].x,
                        lastCorner_->points[minPointInd2].y,
                        lastCorner_->points[minPointInd2].z);

                    ceres::CostFunction* cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b);
                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    corner_correspondence++;
                }
            }
        }

        // --- Step 2.2 Plane 特征匹配 ---
        for (int i = 0; i < surfPointsFlatNum; ++i) {
            pointSel = currFrame.surfFlat->points[i];
            int found = kdtreeSurf_->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
            if (found <= 0) continue;
            if (pointSearchSqDis.empty()) continue;

            if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
                int closestPointInd = pointSearchInd[0];
                int closestPointScanID = static_cast<int>(lastSurf_->points[closestPointInd].intensity);
                int minPointInd2 = -1, minPointInd3 = -1;
                double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                // 前向搜索
                for (int j = closestPointInd + 1; j < static_cast<int>(lastSurf_->points.size()); ++j) {
                    int scanID = static_cast<int>(lastSurf_->points[j].intensity);
                    if (scanID > closestPointScanID + NEARBY_SCAN) break;

                    double dist = pcl::euclideanDistance(lastSurf_->points[j], pointSel);
                    if (scanID <= closestPointScanID && dist * dist < minPointSqDis2) {
                        minPointSqDis2 = dist * dist;
                        minPointInd2 = j;
                    }
                    else if (scanID > closestPointScanID && dist * dist < minPointSqDis3) {
                        minPointSqDis3 = dist * dist;
                        minPointInd3 = j;
                    }
                }

                // 后向搜索
                for (int j = closestPointInd - 1; j >= 0; --j) {
                    int scanID = static_cast<int>(lastSurf_->points[j].intensity);
                    if (scanID < closestPointScanID - NEARBY_SCAN) break;

                    double dist = pcl::euclideanDistance(lastSurf_->points[j], pointSel);
                    if (scanID >= closestPointScanID && dist * dist < minPointSqDis2) {
                        minPointSqDis2 = dist * dist;
                        minPointInd2 = j;
                    }
                    else if (scanID < closestPointScanID && dist * dist < minPointSqDis3) {
                        minPointSqDis3 = dist * dist;
                        minPointInd3 = j;
                    }
                }

                if (minPointInd2 >= 0 && minPointInd3 >= 0) {
                    Eigen::Vector3d curr_point(pointSel.x, pointSel.y, pointSel.z);
                    Eigen::Vector3d last_point_a(
                        lastSurf_->points[closestPointInd].x,
                        lastSurf_->points[closestPointInd].y,
                        lastSurf_->points[closestPointInd].z);
                    Eigen::Vector3d last_point_b(
                        lastSurf_->points[minPointInd2].x,
                        lastSurf_->points[minPointInd2].y,
                        lastSurf_->points[minPointInd2].z);
                    Eigen::Vector3d last_point_c(
                        lastSurf_->points[minPointInd3].x,
                        lastSurf_->points[minPointInd3].y,
                        lastSurf_->points[minPointInd3].z);

                    ceres::CostFunction* cost_function =
                        LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c);
                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    plane_correspondence++;
                }
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 4;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    // --- Step 3: 位姿更新 ---
    // Ceres 参数是 double，poseWorld_ 使用 float (Eigen::Matrix4f)，因此把求解结果转换为 float
    Eigen::Quaterniond q_last_curr = Eigen::Map<Eigen::Quaterniond>(para_q); // 内存布局: (x,y,z,w)
    Eigen::Vector3d t_last_curr_d = Eigen::Map<Eigen::Vector3d>(para_t);

    Eigen::Matrix3f R = q_last_curr.toRotationMatrix().cast<float>();
    Eigen::Vector3f t = t_last_curr_d.cast<float>();

    Eigen::Matrix4f T_last_curr = Eigen::Matrix4f::Identity();
    T_last_curr.block<3, 3>(0, 0) = R;
    T_last_curr.block<3, 1>(0, 3) = t;

    poseWorld_ = poseWorld_ * T_last_curr; // 当前帧在世界坐标系下的位姿

    // --- Step 4: 输出结果 ---
    OdomResult result;
    result.pose = poseWorld_;
    result.cornerCount = corner_correspondence;
    result.planeCount = plane_correspondence;
    result.registeredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    *result.registeredCloud += *currFrame.cornerSharp;
    *result.registeredCloud += *currFrame.surfFlat;

    outputQueue_.push(result);

    // --- Step 5: 更新上一帧 ---
    *lastCorner_ = *currFrame.cornerSharp;
    *lastSurf_ = *currFrame.surfFlat;
}
