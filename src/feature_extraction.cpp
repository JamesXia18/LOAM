#include "feature_extraction.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>

constexpr int N_SCANS = 64; // 适用Velodyne老式激光雷达：可以是16、32、64线
constexpr double scanPeriod = 0.1; // 10Hz的扫描频率

std::vector<float> read_lidar_data(const std::string& lidar_data_path) {
	// Velodyne的点云为二进制文件
	std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
	if (!lidar_data_file) {
		std::cerr << "Failed to open lidar data file: " << lidar_data_path << std::endl;
		return {};
	}
	lidar_data_file.seekg(0, std::ios::end);
	auto file_size = lidar_data_file.tellg();
	if (file_size <= 0) return {};
	const size_t num_elements = static_cast<size_t>(file_size) / sizeof(float);
	lidar_data_file.seekg(0, std::ios::beg);

	std::vector<float> lidar_data_buffer(num_elements);
	if (num_elements > 0)
		lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements * sizeof(float));
	return lidar_data_buffer;
}

FeatureCloud extractFeatures(const pcl::PointCloud<pcl::PointXYZI>::Ptr& laserCloudIn)
{
    FeatureCloud output;
    output.cornerSharp.reset(new pcl::PointCloud<pcl::PointXYZI>());
    output.cornerLessSharp.reset(new pcl::PointCloud<pcl::PointXYZI>());
    output.surfFlat.reset(new pcl::PointCloud<pcl::PointXYZI>());
    output.surfLessFlat.reset(new pcl::PointCloud<pcl::PointXYZI>());

    pcl::PointCloud<pcl::PointXYZI> laserCloud = *laserCloudIn;
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(laserCloud, laserCloud, indices);

    int laserCloudSize = static_cast<int>(laserCloud.points.size());
    if (laserCloudSize < 100) {
        std::cerr << "Too few points for feature extraction!" << std::endl;
        return output;
    }
    float startOri = -std::atan2(laserCloud.points[0].y, laserCloud.points[0].x);
    float endOri = -std::atan2(laserCloud.points[laserCloudSize - 1].y, laserCloud.points[laserCloudSize - 1].x) + 2 * M_PI;

    if (endOri - startOri > 3 * M_PI) // 两者差非常小! 例如起始在179°，结束在180°
        endOri -= 2 * M_PI;
    else if (endOri - startOri < M_PI) // 两者差非常大！例如起始在-179°，结束在359°
        endOri += 2 * M_PI;

    std::vector<pcl::PointCloud<pcl::PointXYZI>> laserCloudScans(N_SCANS);
    bool halfPassed = false; // 处理一半标志位

    // 按线束号区分不同的点云
    for (int i = 0; i < laserCloudSize; ++i) {
        pcl::PointXYZI point = laserCloud.points[i];
        // 计算高度角
        float angle = std::atan(point.z / std::sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;

        // 根据高度角计算线束号
        if (N_SCANS == 16) {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID < 0 || scanID >= 16) continue;
        }
        else if (N_SCANS == 32) {
            scanID = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
            if (scanID < 0 || scanID >= 32) continue;
        }
        else if (N_SCANS == 64) {
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
                continue;
        }

        float ori = -std::atan2(point.y, point.x);
        if (!halfPassed) {
            if (ori < startOri - M_PI / 2) ori += 2 * M_PI;
            else if (ori > startOri + M_PI * 3 / 2) ori -= 2 * M_PI;
            if (ori - startOri > M_PI) halfPassed = true;
        }
        else {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2) ori += 2 * M_PI;
            else if (ori > endOri + M_PI / 2) ori -= 2 * M_PI;
        }

        float relTime = (ori - startOri) / (endOri - startOri);
        // 整数部分是线束号索引，小数部分是相对起始部分时间
        point.intensity = scanID + scanPeriod * relTime;
        // 根据不同的线束号送到不同的点云集合里
        laserCloudScans[scanID].push_back(point);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr fullCloud(new pcl::PointCloud<pcl::PointXYZI>());
    std::vector<int> scanStartInd(N_SCANS, 0); // 记录i线束对应全部点云中的起始序号
    std::vector<int> scanEndInd(N_SCANS, 0); // 记录i线束对应全部点云中的终止序号
    // 将不同线束的点云"拼"起来
    for (int i = 0; i < N_SCANS; ++i) {
        scanStartInd[i] = static_cast<int>(fullCloud->size()) + 5;
        *fullCloud += laserCloudScans[i];
        scanEndInd[i] = static_cast<int>(fullCloud->size()) - 6;
    }

    int totalPoints = static_cast<int>(fullCloud->points.size());
    if (totalPoints < 100) {
        std::cerr << "Too few valid points after scan separation." << std::endl;
        return output;
    }

    // 使用 fullCloud 大小作为后续计算基准，避免越界
    int fullCloudSize = totalPoints;

    // 初始化并确保每个索引都有合理初始值
    std::vector<float> cloudCurvature(fullCloudSize, 0.0f); // 每个点的曲率
    std::vector<int> cloudSortInd(fullCloudSize, 0); // 记录每个点在总点云中的序列号
    std::vector<int> cloudNeighborPicked(fullCloudSize, 0); // 每个点邻居点是否被选择
    std::vector<int> cloudLabel(fullCloudSize, 0); // 当前点的状态位 0: 普通点 2:尖锐的角点 1:角点 -1:平面点
    for (int i = 0; i < fullCloudSize; ++i) cloudSortInd[i] = i;

	// 计算每个点的曲率（有效范围 5 .. size-5）
    for (int i = 5; i < fullCloudSize - 5; ++i) {
        float diffX = 0.0f, diffY = 0.0f, diffZ = 0.0f;
        for (int j = -5; j <= 5; ++j) {
            if (j != 0) {
                diffX += fullCloud->points[i + j].x;
                diffY += fullCloud->points[i + j].y;
                diffZ += fullCloud->points[i + j].z;
            }
        }
        diffX -= 10.0f * fullCloud->points[i].x;
        diffY -= 10.0f * fullCloud->points[i].y;
        diffZ -= 10.0f * fullCloud->points[i].z;

        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        // cloudSortInd[i] 已经初始化为 i
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
    }

	// 特征点提取
    for (int i = 0; i < N_SCANS; ++i) {
        if (scanEndInd[i] - scanStartInd[i] < 6)
            continue;

        pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<pcl::PointXYZI>);

        // 每个线束分成6个区段，均匀选点
        for (int j = 0; j < 6; ++j) {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;
            if (ep <= sp)
                continue;
            if (sp < 0) sp = 0;
            if (ep >= fullCloudSize) ep = fullCloudSize - 1;

            // 按曲率升序排序
            std::sort(cloudSortInd.begin() + sp, cloudSortInd.begin() + ep + 1,
                [&](int a, int b) { return cloudCurvature[a] < cloudCurvature[b]; });

            // ----- (1) 选择角点 -----
            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--) {
                int ind = cloudSortInd[k];
                if (cloudNeighborPicked[ind]) continue;
                if (cloudCurvature[ind] < 0.1f) break;

                largestPickedNum++;
                if (largestPickedNum <= 2) {
                    cloudLabel[ind] = 2; // sharp
                    output.cornerSharp->push_back(fullCloud->points[ind]);
                    output.cornerLessSharp->push_back(fullCloud->points[ind]);
                }
                else if (largestPickedNum <= 20) {
                    cloudLabel[ind] = 1; // less sharp
                    output.cornerLessSharp->push_back(fullCloud->points[ind]);
                }
                else break;

                // 抑制邻域
                cloudNeighborPicked[ind] = 1;
                for (int l = 1; l <= 5; ++l) {
                    if (ind + l >= fullCloudSize) break;
                    float diffX = fullCloud->points[ind + l].x - fullCloud->points[ind + l - 1].x;
                    float diffY = fullCloud->points[ind + l].y - fullCloud->points[ind + l - 1].y;
                    float diffZ = fullCloud->points[ind + l].z - fullCloud->points[ind + l - 1].z;
                    if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05f)
                        break;
                    cloudNeighborPicked[ind + l] = 1;
                }
                for (int l = -1; l >= -5; --l) {
                    if (ind + l < 0) break;
                    float diffX = fullCloud->points[ind + l].x - fullCloud->points[ind + l + 1].x;
                    float diffY = fullCloud->points[ind + l].y - fullCloud->points[ind + l + 1].y;
                    float diffZ = fullCloud->points[ind + l].z - fullCloud->points[ind + l + 1].z;
                    if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05f)
                        break;
                    cloudNeighborPicked[ind + l] = 1;
                }
            }

            // ----- (2) 选择平面点 -----
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; ++k) {
                int ind = cloudSortInd[k];
                if (cloudNeighborPicked[ind]) continue;
                if (cloudCurvature[ind] > 0.1f) break;

                smallestPickedNum++;
                if (smallestPickedNum <= 4) {
                    cloudLabel[ind] = -1; // flat
                    output.surfFlat->push_back(fullCloud->points[ind]);
                }
                else break;

                cloudNeighborPicked[ind] = 1;
                for (int l = 1; l <= 5; ++l) {
                    if (ind + l >= fullCloudSize) break;
                    float diffX = fullCloud->points[ind + l].x - fullCloud->points[ind + l - 1].x;
                    float diffY = fullCloud->points[ind + l].y - fullCloud->points[ind + l - 1].y;
                    float diffZ = fullCloud->points[ind + l].z - fullCloud->points[ind + l - 1].z;
                    if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05f)
                        break;
                    cloudNeighborPicked[ind + l] = 1;
                }
                for (int l = -1; l >= -5; --l) {
                    if (ind + l < 0) break;
                    float diffX = fullCloud->points[ind + l].x - fullCloud->points[ind + l + 1].x;
                    float diffY = fullCloud->points[ind + l].y - fullCloud->points[ind + l + 1].y;
                    float diffZ = fullCloud->points[ind + l].z - fullCloud->points[ind + l + 1].z;
                    if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05f)
                        break;
                    cloudNeighborPicked[ind + l] = 1;
                }
            }

            // ----- (3) 其他点归入 less flat -----
            for (int k = sp; k <= ep; ++k) {
                if (cloudLabel[k] <= 0)
                    surfPointsLessFlatScan->push_back(fullCloud->points[k]);
            }
        }

        // 对每个线束的普通面点下采样
        pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2f, 0.2f, 0.2f);
        pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlatScanDS(new pcl::PointCloud<pcl::PointXYZI>());
        downSizeFilter.filter(*surfPointsLessFlatScanDS);
        *output.surfLessFlat += *surfPointsLessFlatScanDS;
    }   
    return output;
}


FeatureExtractionThread::FeatureExtractionThread(
    SafeQueue<pcl::PointCloud<pcl::PointXYZI>::Ptr>& inputQueue,
    SafeQueue<FeatureCloud>& outputQueue)
    : inputQueue_(inputQueue), outputQueue_(outputQueue) {
}

FeatureExtractionThread::~FeatureExtractionThread() {
    stop();
}

void FeatureExtractionThread::start() {
    running_ = true;
    worker_ = std::thread(&FeatureExtractionThread::processLoop, this);
}

void FeatureExtractionThread::stop() {
    running_ = false;
    inputQueue_.stop();
    if (worker_.joinable())
        worker_.join();
}

void FeatureExtractionThread::processLoop() {
    std::cout << "[FeatureExtractionThread] Started." << std::endl;

    while (running_) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr rawCloud;
        if (!inputQueue_.pop(rawCloud)) {
            if (!running_) break;
            continue;
        }

        if (!rawCloud || rawCloud->empty()) {
            std::cerr << "[FeatureExtractionThread] Received empty cloud." << std::endl;
            continue;
        }

        auto start = std::chrono::steady_clock::now();
        FeatureCloud features = extractFeatures(rawCloud);
        auto end = std::chrono::steady_clock::now();

        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[FeatureExtractionThread] Processed 1 frame (" << rawCloud->size()
            << " pts) in " << ms << " ms." << std::endl;

        outputQueue_.push(features);
    }

    std::cout << "[FeatureExtractionThread] Stopped." << std::endl;
}