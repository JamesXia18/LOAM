#pragma once
#include <pcl/common/common.h>
#include <vector>
#include <string>

struct FeatureCloud {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerSharp;      // 尖锐角点
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerLessSharp;  // 次尖锐角点
    pcl::PointCloud<pcl::PointXYZI>::Ptr surfFlat;         // 平面点
    pcl::PointCloud<pcl::PointXYZI>::Ptr surfLessFlat;     // 普通面点
};

std::vector<float> read_lidar_data(const std::string& lidar_data_path);

FeatureCloud extractFeatures(const pcl::PointCloud<pcl::PointXYZI>::Ptr& laserCloudIn);
