#include "laser_mapping.h"

#include <pcl/filters/voxel_grid.h>
#include "lidarFactor.hpp"
#include "laser_odometry.h"

LaserMappingThread::LaserMappingThread(SafeQueue<OdomResult> &inputQueue,
                                       SafeQueue<MapResult> &outputQueue,
                                       float lineRes = 0.4, float planeRes = 0.8
)
    : inputQueue_(inputQueue),
      outputQueue_(outputQueue),
      lineRes_(lineRes), planeRes_(planeRes),
      q_w_curr_(parameters_),
      t_w_curr_(parameters_ + 4) {
    parameters_[0] = 0.0; // qx
    parameters_[1] = 0.0; // qy
    parameters_[2] = 0.0; // qz
    parameters_[3] = 1.0; // qw
    parameters_[4] = 0.0; // tx
    parameters_[5] = 0.0; // ty
    parameters_[6] = 0.0; // tz
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
    if (worker_.joinable())
        worker_.join();
}

void LaserMappingThread::processLoop() {
    std::cout << "[LaserMappingThread] Started." << std::endl;

    int frameCount = 0;

    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurround(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterCorner;
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterSurf;
    downSizeFilterCorner.setLeafSize(lineRes_, lineRes_, lineRes_);
    downSizeFilterSurf.setLeafSize(planeRes_, planeRes_, planeRes_);

    OdomResult odomResult;
    MapResult mapResult;

    Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
    Eigen::Vector3d t_wmap_wodom(0, 0, 0);
    Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
    Eigen::Vector3d t_wodom_curr(0, 0, 0);

    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerArray[laserCloudNum];
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfArray[laserCloudNum];
    for (int idx = 0; idx < laserCloudNum; ++idx) {
        laserCloudCornerArray[idx].reset(new pcl::PointCloud<pcl::PointXYZI>());
        laserCloudSurfArray[idx].reset(new pcl::PointCloud<pcl::PointXYZI>());
    }

    int laserCloudValidInd[125];
    int laserCloudSurroundInd[125];

    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<pcl::PointXYZI>());

    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<pcl::PointXYZI>());

    // match the current frame with the existing map
    auto transformAssociateToMap = [&]() {
        q_w_curr_ = q_wmap_wodom * q_wodom_curr;
        t_w_curr_ = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
    };

    auto transformUpdate = [&] {
        q_wmap_wodom = q_w_curr_ * q_wodom_curr.inverse();
        t_wmap_wodom = t_w_curr_ - q_wmap_wodom * t_wodom_curr;
    };

    auto pointAssociateToMap = [&](const pcl::PointXYZI *const pi, pcl::PointXYZI *const po) {
        Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
        Eigen::Vector3d point_w = q_w_curr_ * point_curr + t_w_curr_;
        po->x = point_w.x();
        po->y = point_w.y();
        po->z = point_w.z();
        po->intensity = pi->intensity;
    };

    while (running_) {
        if (!inputQueue_.pop(odomResult)) {
            if (!running_) break;
            continue;
        }

        if (odomResult.lastCorner_->empty() || odomResult.lastSurf_->empty()) {
            std::cerr << "[LaserMappingThread] Empty feature cloud, skip frame." << std::endl;
            continue;
        }
        auto start = std::chrono::steady_clock::now();


        auto t = odomResult.pose.block<3,1>(0,3).eval().cast<double>();
        auto R = odomResult.pose.block<3,3>(0,0).eval().cast<double>();

        q_wodom_curr = Eigen::Quaterniond(R);
        t_wodom_curr = t;

        // process
        transformAssociateToMap();

        int centerCubeI = int((t_w_curr_.x() + 25.0) / 50.0) + laserCloudCenWidth;
        int centerCubeJ = int((t_w_curr_.y() + 25.0) / 50.0) + laserCloudCenHeight;
        int centerCubeK = int((t_w_curr_.z() + 25.0) / 50.0) + laserCloudCenDepth;

        if (t_w_curr_.x() + 25.0 < 0)
            centerCubeI--;
        if (t_w_curr_.y() + 25.0 < 0)
            centerCubeJ--;
        if (t_w_curr_.z() + 25.0 < 0)
            centerCubeK--;

        while (centerCubeI < 3) {
            for (int j = 0; j < laserCloudHeight; j++) {
                for (int k = 0; k < laserCloudDepth; k++) {
                    int i = laserCloudWidth - 1;
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer =
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer =
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; i >= 1; i--) {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCornerArray[
                                    i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudSurfArray[
                                    i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }
            centerCubeI++;
            laserCloudCenWidth++;
        }

        while (centerCubeI >= laserCloudWidth - 3) {
            for (int j = 0; j < laserCloudHeight; j++) {
                for (int k = 0; k < laserCloudDepth; k++) {
                    int i = 0;
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer =
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer =
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; i < laserCloudWidth - 1; i++) {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCornerArray[
                                    i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudSurfArray[
                                    i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeI--;
            laserCloudCenWidth--;
        }

        while (centerCubeJ < 3) {
            for (int i = 0; i < laserCloudWidth; i++) {
                for (int k = 0; k < laserCloudDepth; k++) {
                    int j = laserCloudHeight - 1;
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer =
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer =
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; j >= 1; j--) {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCornerArray[
                                    i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudSurfArray[
                                    i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeJ++;
            laserCloudCenHeight++;
        }

        while (centerCubeJ >= laserCloudHeight - 3) {
            for (int i = 0; i < laserCloudWidth; i++) {
                for (int k = 0; k < laserCloudDepth; k++) {
                    int j = 0;
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer =
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer =
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; j < laserCloudHeight - 1; j++) {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCornerArray[
                                    i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudSurfArray[
                                    i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeJ--;
            laserCloudCenHeight--;
        }

        while (centerCubeK < 3) {
            for (int i = 0; i < laserCloudWidth; i++) {
                for (int j = 0; j < laserCloudHeight; j++) {
                    int k = laserCloudDepth - 1;
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer =
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer =
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; k >= 1; k--) {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCornerArray[
                                    i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudSurfArray[
                                    i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeK++;
            laserCloudCenDepth++;
        }

        while (centerCubeK >= laserCloudDepth - 3) {
            for (int i = 0; i < laserCloudWidth; i++) {
                for (int j = 0; j < laserCloudHeight; j++) {
                    int k = 0;
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer =
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer =
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    for (; k < laserCloudDepth - 1; k++) {
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCornerArray[
                                    i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudSurfArray[
                                    i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
                    }
                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeCornerPointer;
                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            laserCloudCubeSurfPointer;
                    laserCloudCubeCornerPointer->clear();
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeK--;
            laserCloudCenDepth--;
        }

        int laserCloudValidNum = 0;
        int laserCloudSurroundNum = 0;

        for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
            for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
                for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++) {
                    // 确保不越界
                    if (i >= 0 && i < laserCloudWidth &&
                        j >= 0 && j < laserCloudHeight &&
                        k >= 0 && k < laserCloudDepth) {
                        laserCloudValidInd[laserCloudValidNum] =
                                i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                        laserCloudValidNum++;
                        laserCloudSurroundInd[laserCloudSurroundNum] =
                                i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                        laserCloudSurroundNum++;
                    }
                }
            }
        }

        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        for (int i = 0; i < laserCloudValidNum; i++) {
            *laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
            *laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
        }
        int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
        int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();


        pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerStack(new pcl::PointCloud<pcl::PointXYZI>());
        downSizeFilterCorner.setInputCloud(odomResult.lastCorner_);
        downSizeFilterCorner.filter(*laserCloudCornerStack);
        int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

        pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfStack(new pcl::PointCloud<pcl::PointXYZI>());
        downSizeFilterSurf.setInputCloud(odomResult.lastSurf_);
        downSizeFilterSurf.filter(*laserCloudSurfStack);
        int laserCloudSurfStackNum = laserCloudSurfStack->points.size();
        if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50) {
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
            for (int iterCount = 0; iterCount < 2; iterCount++) {
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::Manifold *q_parameterization =
                        new ceres::EigenQuaternionManifold();
                ceres::Problem::Options problem_options;

                ceres::Problem problem(problem_options);
                problem.AddParameterBlock(parameters_, 4, q_parameterization);
                problem.AddParameterBlock(parameters_ + 4, 3);

                pcl::PointXYZI pointOri, pointSel;
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;

                int corner_num = 0;
                for (int i = 0; i < laserCloudCornerStackNum; i++) {
                    pointOri = laserCloudCornerStack->points[i];

                    pointAssociateToMap(&pointOri, &pointSel);
                    kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                    if (pointSearchSqDis[4] < 1.0) {
                        std::vector<Eigen::Vector3d> nearCorners;
                        Eigen::Vector3d center(0, 0, 0);
                        for (int j = 0; j < 5; j++) {
                            Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
                                                laserCloudCornerFromMap->points[pointSearchInd[j]].y,
                                                laserCloudCornerFromMap->points[pointSearchInd[j]].z);
                            center = center + tmp;
                            nearCorners.push_back(tmp);
                        }
                        center = center / 5.0;

                        Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                        for (int j = 0; j < 5; j++) {
                            Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                            covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                        }

                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                        Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                        Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                        if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
                            Eigen::Vector3d point_on_line = center;
                            Eigen::Vector3d point_a, point_b;
                            point_a = 0.1 * unit_direction + point_on_line;
                            point_b = -0.1 * unit_direction + point_on_line;
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b);
                            problem.AddResidualBlock(cost_function, loss_function, parameters_, parameters_ + 4);
                            corner_num++;
                        }
                    }
                }
                int surf_num = 0;
                for (int i = 0; i < laserCloudSurfStackNum; i++) {
                    pointOri = laserCloudSurfStack->points[i];
                    pointAssociateToMap(&pointOri, &pointSel);
                    kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                    Eigen::Matrix<double, 5, 3> matA0;
                    Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
                    if (pointSearchSqDis[4] < 1.0) {
                        for (int j = 0; j < 5; j++) {
                            matA0(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
                            matA0(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
                            matA0(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
                        }
                        Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                        double negative_OA_dot_norm = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++) {
                            if (fabs(norm(0) * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
                                     norm(1) * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
                                     norm(2) * laserCloudSurfFromMap->points[pointSearchInd[j]].z +
                                     negative_OA_dot_norm) > 0.2) {
                                planeValid = false;
                                break;
                            }
                        }
                        Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                        if (planeValid) {
                            ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(
                                curr_point, norm, negative_OA_dot_norm);
                            problem.AddResidualBlock(cost_function, loss_function, parameters_, parameters_ + 4);
                            surf_num++;
                        }
                    }
                }
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 4;
                options.minimizer_progress_to_stdout = false;
                options.check_gradients = false;
                options.gradient_check_relative_precision = 1e-4;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
            }
        }

        transformUpdate();
        pcl::PointXYZI pointSel;

        for (int i = 0; i < laserCloudCornerStackNum; i++)
        {
            pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);

            int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
            int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
            int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

            if (pointSel.x + 25.0 < 0)
                cubeI--;
            if (pointSel.y + 25.0 < 0)
                cubeJ--;
            if (pointSel.z + 25.0 < 0)
                cubeK--;

            if (cubeI >= 0 && cubeI < laserCloudWidth &&
                cubeJ >= 0 && cubeJ < laserCloudHeight &&
                cubeK >= 0 && cubeK < laserCloudDepth)
            {
                int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                laserCloudCornerArray[cubeInd]->push_back(pointSel);
            }
        }

        for (int i = 0; i < laserCloudSurfStackNum; i++)
        {
            pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

            int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
            int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
            int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

            if (pointSel.x + 25.0 < 0)
                cubeI--;
            if (pointSel.y + 25.0 < 0)
                cubeJ--;
            if (pointSel.z + 25.0 < 0)
                cubeK--;

            if (cubeI >= 0 && cubeI < laserCloudWidth &&
                cubeJ >= 0 && cubeJ < laserCloudHeight &&
                cubeK >= 0 && cubeK < laserCloudDepth)
            {
                int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                laserCloudSurfArray[cubeInd]->push_back(pointSel);
            }
        }

        for (int i = 0; i < laserCloudValidNum; i++)
        {
            int ind = laserCloudValidInd[i];

            pcl::PointCloud<pcl::PointXYZI>::Ptr tmpCorner(new pcl::PointCloud<pcl::PointXYZI>());
            downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
            downSizeFilterCorner.filter(*tmpCorner);
            laserCloudCornerArray[ind] = tmpCorner;

            pcl::PointCloud<pcl::PointXYZI>::Ptr tmpSurf(new pcl::PointCloud<pcl::PointXYZI>());
            downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
            downSizeFilterSurf.filter(*tmpSurf);
            laserCloudSurfArray[ind] = tmpSurf;
        }

        if (frameCount % 5 == 0) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr surround(new pcl::PointCloud<pcl::PointXYZI>());
            for (int i = 0; i < laserCloudSurroundNum; i++) {
                int ind = laserCloudSurroundInd[i];
                *surround += *laserCloudCornerArray[ind];
                *surround += *laserCloudSurfArray[ind];
            }

            mapResult.pose = Eigen::Matrix4f::Identity();
            mapResult.pose.block<3,3>(0,0) = q_w_curr_.toRotationMatrix().transpose().cast<float>();
            mapResult.pose.block<3,1>(0,3) = t_w_curr_.cast<float>();

            pcl::PointCloud<pcl::PointXYZI>::Ptr mapCloudCopy(new pcl::PointCloud<pcl::PointXYZI>(*surround));
            mapResult.mapCloud = mapCloudCopy;

            std::cout<<"[laserMappingThread]"<<mapResult.pose<<std::endl;

            outputQueue_.push(mapResult);
        }

        frameCount++;

        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[LaserMappingThread] Frame processed in " << ms << " ms." << std::endl;
    }
    std::cout << "[LaserMappingThread] Stopped." << std::endl;
}
