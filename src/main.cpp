#include "feature_extraction.h"
#include "laser_odometry.h"
#include "laser_mapping.h"

#include <iostream>
#include <chrono>
#include <thread>


int main() {
    SafeQueue<pcl::PointCloud<pcl::PointXYZI>::Ptr> rawCloudQueue;
    SafeQueue<FeatureCloud> featureQueue;
    SafeQueue<OdomResult> odomQueue;
    SafeQueue<MapResult> mapQueue;

    FeatureExtractionThread featureThread(rawCloudQueue, featureQueue);
    LaserOdometryThread odomThread(featureQueue, odomQueue);
    LaserMappingThread mappingThread(odomQueue, mapQueue, 0.3f, 300000);

    featureThread.start();
    odomThread.start();
    mappingThread.start();

    // 模拟输入多帧点云

    std::string folder = "F:\\桌面\\sequences\\00\\velodyne\\";
    for (int i = 0; i < 10; ++i) {
        std::string filename = folder + (std::to_string(i).insert(0, 6 - std::to_string(i).length(), '0')) + ".bin";
        std::vector<float> lidar_data = read_lidar_data(filename);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
        for (size_t k = 0; k < lidar_data.size(); k += 4) {
            pcl::PointXYZI p;
            p.x = lidar_data[k];
            p.y = lidar_data[k + 1];
            p.z = lidar_data[k + 2];
            p.intensity = lidar_data[k + 3];
            cloud->push_back(p);
        }

        rawCloudQueue.push(cloud);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    featureThread.stop();

    // 等待处理
    std::this_thread::sleep_for(std::chrono::seconds(10));

    // 停止线程
    featureThread.stop();
    odomThread.stop();
    mappingThread.stop();

    // 读取一些 map 快照
    while (!mapQueue.empty()) {
        MapResult mr;
        if (mapQueue.pop(mr)) {
            std::cout << "Map snapshot: pose:\n" << mr.pose << " map size: " << mr.mapCloud->size() << std::endl;
        }
    }

    return 0;
}
