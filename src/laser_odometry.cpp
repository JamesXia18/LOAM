#include "laser_odometry.h"
#include "lidarFactor.hpp"

#include <pcl/common/distances.h>

#include <iostream>
#include <chrono>
#include <cmath>

constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

double para_q[4] = { 0, 0, 0, 1 };
double para_t[3] = { 0, 0, 0 };

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
    para_q[0] = para_q[1] = para_q[2] = 0.0; para_q[3] = 1.0;
    para_t[0] = para_t[1] = para_t[2] = 0.0;
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
    if (lastCorner_->empty()) {
        if (currFrame.cornerLessSharp->empty() || currFrame.surfLessFlat->empty()) {
            std::cerr << "[LaserOdometryThread] First frame has no less features, skip." << std::endl;
            return;
        }

        *lastCorner_ = *currFrame.cornerLessSharp;  // laserCloudCornerLast
        *lastSurf_ = *currFrame.surfLessFlat;     // laserCloudSurfLast

        kdtreeCorner_->setInputCloud(lastCorner_);
        kdtreeSurf_->setInputCloud(lastSurf_);

        poseWorld_.setIdentity();
        std::cout << "[LaserOdometryThread] First frame initialized." << std::endl;
        return;
    }

    if (currFrame.cornerSharp->empty() || currFrame.surfFlat->empty()) {
        std::cerr << "[LaserOdometryThread] Empty sharp/flat feature, skip frame." << std::endl;
        return;
    }

    kdtreeCorner_->setInputCloud(lastCorner_);
    kdtreeSurf_->setInputCloud(lastSurf_);

    const int cornerPointsSharpNum = static_cast<int>(currFrame.cornerSharp->size());
    const int surfPointsFlatNum = static_cast<int>(currFrame.surfFlat->size());

    int corner_correspondence = 0;
    int plane_correspondence = 0;

    for (int opti = 0; opti < 2; ++opti) {

        corner_correspondence = 0;
        plane_correspondence = 0;

        ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
        ceres::Manifold* q_parameterization = new ceres::EigenQuaternionManifold();
        ceres::Problem         problem;

        problem.AddParameterBlock(para_q, 4, q_parameterization);
        problem.AddParameterBlock(para_t, 3);

        Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);    // (x,y,z,w)
        Eigen::Map<Eigen::Vector3d>    t_last_curr(para_t);

        auto transformToStart = [&](pcl::PointXYZI const& pi, pcl::PointXYZI& po)
            {
                double intensity = pi.intensity;
                double s = intensity - std::floor(intensity);

                s = 1.0; // Kitti has done ...
                Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
                Eigen::Vector3d    t_point_last = s * t_last_curr;

                Eigen::Vector3d point(pi.x, pi.y, pi.z);
                Eigen::Vector3d un_point = q_point_last * point + t_point_last; // ���Ƶ���һ֡��ʼ

                po.x = static_cast<float>(un_point.x());
                po.y = static_cast<float>(un_point.y());
                po.z = static_cast<float>(un_point.z());
                po.intensity = pi.intensity;
            };

        pcl::PointXYZI pointSel;
        std::vector<int>   pointSearchInd;
        std::vector<float> pointSearchSqDis;

        for (int i = 0; i < cornerPointsSharpNum; ++i) {
            transformToStart(currFrame.cornerSharp->points[i], pointSel);

            if (kdtreeCorner_->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis) <= 0)
                continue;
            if (pointSearchSqDis.empty() || pointSearchSqDis[0] >= DISTANCE_SQ_THRESHOLD)
                continue;

            int   closestPointInd = pointSearchInd[0];
            int   closestScanID = static_cast<int>(lastCorner_->points[closestPointInd].intensity);
            int   minPointInd2 = -1;
            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;

            for (int j = closestPointInd + 1; j < static_cast<int>(lastCorner_->points.size()); ++j) {
                int scanID = static_cast<int>(lastCorner_->points[j].intensity);
                if (scanID <= closestScanID) continue;
                if (scanID > closestScanID + NEARBY_SCAN) break;

                double dx = lastCorner_->points[j].x - pointSel.x;
                double dy = lastCorner_->points[j].y - pointSel.y;
                double dz = lastCorner_->points[j].z - pointSel.z;
                double dist2 = dx * dx + dy * dy + dz * dz;

                if (dist2 < minPointSqDis2) {
                    minPointSqDis2 = dist2;
                    minPointInd2 = j;
                }
            }

            for (int j = closestPointInd - 1; j >= 0; --j) {
                int scanID = static_cast<int>(lastCorner_->points[j].intensity);
                if (scanID >= closestScanID) continue;
                if (scanID < closestScanID - NEARBY_SCAN) break;

                double dx = lastCorner_->points[j].x - pointSel.x;
                double dy = lastCorner_->points[j].y - pointSel.y;
                double dz = lastCorner_->points[j].z - pointSel.z;
                double dist2 = dx * dx + dy * dy + dz * dz;

                if (dist2 < minPointSqDis2) {
                    minPointSqDis2 = dist2;
                    minPointInd2 = j;
                }
            }

            if (minPointInd2 >= 0) {
                Eigen::Vector3d curr_point(
                    currFrame.cornerSharp->points[i].x,
                    currFrame.cornerSharp->points[i].y,
                    currFrame.cornerSharp->points[i].z);
                Eigen::Vector3d last_point_a(
                    lastCorner_->points[closestPointInd].x,
                    lastCorner_->points[closestPointInd].y,
                    lastCorner_->points[closestPointInd].z);
                Eigen::Vector3d last_point_b(
                    lastCorner_->points[minPointInd2].x,
                    lastCorner_->points[minPointInd2].y,
                    lastCorner_->points[minPointInd2].z);

                if ((last_point_a - last_point_b).squaredNorm() < 1e-6)
                    continue;

                ceres::CostFunction* cost_function =
                    LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b);
                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                ++corner_correspondence;
            }
        }

        for (int i = 0; i < surfPointsFlatNum; ++i) {
            transformToStart(currFrame.surfFlat->points[i], pointSel);

            if (kdtreeSurf_->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis) <= 0)
                continue;
            if (pointSearchSqDis.empty() || pointSearchSqDis[0] >= DISTANCE_SQ_THRESHOLD)
                continue;

            int closestPointInd = pointSearchInd[0];
            int closestScanID = static_cast<int>(lastSurf_->points[closestPointInd].intensity);

            int    minPointInd2 = -1, minPointInd3 = -1;
            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
            double minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

            for (int j = closestPointInd + 1; j < static_cast<int>(lastSurf_->points.size()); ++j) {
                int scanID = static_cast<int>(lastSurf_->points[j].intensity);
                if (scanID > closestScanID + NEARBY_SCAN) break;

                double dx = lastSurf_->points[j].x - pointSel.x;
                double dy = lastSurf_->points[j].y - pointSel.y;
                double dz = lastSurf_->points[j].z - pointSel.z;
                double dist2 = dx * dx + dy * dy + dz * dz;

                if (scanID <= closestScanID) {
                    if (dist2 < minPointSqDis2) {
                        minPointSqDis2 = dist2;
                        minPointInd2 = j;
                    }
                }
                else {
                    if (dist2 < minPointSqDis3) {
                        minPointSqDis3 = dist2;
                        minPointInd3 = j;
                    }
                }
            }

            for (int j = closestPointInd - 1; j >= 0; --j) {
                int scanID = static_cast<int>(lastSurf_->points[j].intensity);
                if (scanID < closestScanID - NEARBY_SCAN) break;

                double dx = lastSurf_->points[j].x - pointSel.x;
                double dy = lastSurf_->points[j].y - pointSel.y;
                double dz = lastSurf_->points[j].z - pointSel.z;
                double dist2 = dx * dx + dy * dy + dz * dz;

                if (scanID >= closestScanID) {
                    if (dist2 < minPointSqDis2) {
                        minPointSqDis2 = dist2;
                        minPointInd2 = j;
                    }
                }
                else {
                    if (dist2 < minPointSqDis3) {
                        minPointSqDis3 = dist2;
                        minPointInd3 = j;
                    }
                }
            }

            if (minPointInd2 >= 0 && minPointInd3 >= 0) {
                Eigen::Vector3d last_a(
                    lastSurf_->points[closestPointInd].x,
                    lastSurf_->points[closestPointInd].y,
                    lastSurf_->points[closestPointInd].z);
                Eigen::Vector3d last_b(
                    lastSurf_->points[minPointInd2].x,
                    lastSurf_->points[minPointInd2].y,
                    lastSurf_->points[minPointInd2].z);
                Eigen::Vector3d last_c(
                    lastSurf_->points[minPointInd3].x,
                    lastSurf_->points[minPointInd3].y,
                    lastSurf_->points[minPointInd3].z);

                if (((last_b - last_a).cross(last_c - last_a)).squaredNorm() < 1e-8)
                    continue;

                Eigen::Vector3d curr_point(
                    currFrame.surfFlat->points[i].x,
                    currFrame.surfFlat->points[i].y,
                    currFrame.surfFlat->points[i].z);

                ceres::CostFunction* cost_function =
                    LidarPlaneFactor::Create(curr_point, last_a, last_b, last_c);
                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                ++plane_correspondence;
            }
        }

        if (corner_correspondence + plane_correspondence < 10) {
            std::cout << "[LaserOdometryThread] Too few correspondences: "
                << corner_correspondence << " corners, "
                << plane_correspondence << " planes." << std::endl;
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 4;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    Eigen::Quaterniond q_last_curr_res(para_q[3], para_q[0], para_q[1], para_q[2]); // w,x,y,z
    Eigen::Vector3d    t_last_curr_res(para_t[0], para_t[1], para_t[2]);

    Eigen::Matrix4f T_last_curr = Eigen::Matrix4f::Identity();
    T_last_curr.block<3, 3>(0, 0) = q_last_curr_res.toRotationMatrix().cast<float>();
    T_last_curr.block<3, 1>(0, 3) = t_last_curr_res.cast<float>();

    poseWorld_ = poseWorld_ * T_last_curr;

    OdomResult result;
    result.pose = poseWorld_;
    result.cornerCount = corner_correspondence;
    result.planeCount = plane_correspondence;

    *result.lastCorner_ = *currFrame.cornerSharp;
    *result.lastSurf_ = *currFrame.surfFlat;

    outputQueue_.push(result);

    lastCorner_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    lastSurf_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    *lastCorner_ = *currFrame.cornerLessSharp;
    *lastSurf_ = *currFrame.surfLessFlat;

}