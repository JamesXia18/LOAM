#include "feature_extraction.h"
#include "laser_odometry.h"
#include "laser_mapping.h"

#include <pangolin/pangolin.h>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <thread>
#include <mutex>
#include <vector>

int main() {

    // kitti 00.carlib
    Eigen::Matrix4f Tr;
    Tr << 4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
        -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
        9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
        0, 0, 0, 1;

    SafeQueue<pcl::PointCloud<pcl::PointXYZI>::Ptr> rawCloudQueue;
    SafeQueue<FeatureCloud> featureQueue;
    SafeQueue<OdomResult> odomQueue;
    SafeQueue<MapResult> mapQueue;

    FeatureExtractionThread featureThread(rawCloudQueue, featureQueue);
    LaserOdometryThread odomThread(featureQueue, odomQueue);
    LaserMappingThread mappingThread(odomQueue, mapQueue, 0.4, 0.8);

    featureThread.start();
    odomThread.start();
    mappingThread.start();

    std::vector<Eigen::Matrix4f> poses;
    pcl::PointCloud<pcl::PointXYZI>::Ptr mapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    std::mutex dataMutex;

    std::string folder = R"(F:\桌面\sequences\00\velodyne\)";
    auto files = list_bin_files(folder);
    if (files.empty()) {
        std::cerr << "[Error] No .bin files found in " << folder << std::endl;
        return -1;
    }
    std::cout << "[Main] found " << files.size() << " velodyne files.\n";

    // feeder thread: push all files (iterate whole sequence)
    std::thread feeder([&rawCloudQueue, files]() {
        for (size_t idx = 0; idx < files.size(); ++idx) {
            const auto& p = files[idx];
            std::string filename = p.string();
            // 你原有的 read_lidar_data 函数（返回 float vector）
            std::vector<float> lidar_data = read_lidar_data(filename);

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
            for (size_t k = 0; k + 3 < lidar_data.size(); k += 4) {
                pcl::PointXYZI pt;
                pt.x = lidar_data[k];
                pt.y = lidar_data[k + 1];
                pt.z = lidar_data[k + 2];
                pt.intensity = lidar_data[k + 3];
                cloud->push_back(pt);
            }

            rawCloudQueue.push(cloud);

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "[Feeder] finished pushing all files.\n";
        });

    // ---- Pangolin on main thread ----
    pangolin::CreateWindowAndBind("ALOAM Viewer", 1280, 720);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1280, 720, 500, 500, 640, 360, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -10, -10, 0, 0, 0, 0, -1, 0));

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1280.0f / 720.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    // 绘制坐标轴
    auto glDrawAxis = [](const Eigen::Matrix4f& T, float len = 1.0f) {
        Eigen::Vector3f origin = T.block<3,1>(0,3);
        Eigen::Vector3f x = T.block<3,1>(0,0).normalized() * len + origin;
        Eigen::Vector3f y = T.block<3,1>(0,1).normalized() * len + origin;
        Eigen::Vector3f z = T.block<3,1>(0,2).normalized() * len + origin;
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        // X - red
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(origin.x(), origin.y(), origin.z());
        glVertex3f(x.x(), x.y(), x.z());
        // Y - green
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(origin.x(), origin.y(), origin.z());
        glVertex3f(y.x(), y.y(), y.z());
        // Z - blue
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(origin.x(), origin.y(), origin.z());
        glVertex3f(z.x(), z.y(), z.z());
        glEnd();
    };

    // 绘制轨迹
    auto drawTrajectory = [&]() {
        std::lock_guard<std::mutex> lock(dataMutex);
        if (poses.empty()) return;
        glLineWidth(2);
        glColor3f(1.0, 0.0, 0.0);
        glBegin(GL_LINE_STRIP);
        for (const auto& pose : poses) {
            glVertex3f(pose(0,3), pose(1,3), pose(2,3));
        }
        glEnd();
    };

    // main render loop (runs in main thread)
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // non-blocking pull from mapQueue using try_pop
        MapResult mr;
        int got = 0;
        while (mapQueue.try_pop(mr)) {
            std::lock_guard<std::mutex> lock(dataMutex);
            poses.push_back(Tr*mr.pose);
            if (mr.mapCloud) {
                *mapCloud += *mr.mapCloud;
            }
            ++got;
        }
        if (got > 0) {
            std::lock_guard<std::mutex> lock(dataMutex);
            std::cout << "[MainViewer] pulled " << got << " MapResult(s). poses=" << poses.size()
                      << " mapPoints=" << mapCloud->size() << std::endl;
        }

        d_cam.Activate(s_cam);

        // origin axis (always visible)
        glDisable(GL_DEPTH_TEST);
        Eigen::Matrix4f originT = Eigen::Matrix4f::Identity();
        glDrawAxis(originT, 2.0f);
        glEnable(GL_DEPTH_TEST);

        drawTrajectory();

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // window closed -> stop threads
    featureThread.stop();
    odomThread.stop();
    mappingThread.stop();

    if (feeder.joinable()) feeder.join();

    std::string out_path = R"(F:\桌面\KITTI_00_est.txt)";
    {
        std::lock_guard<std::mutex> lock(dataMutex);
        bool ok = save_poses_kitti(out_path, poses);
        if (ok) {
            std::cout << "[Main] saved " << poses.size() << " poses to " << out_path << std::endl;
        }
        else {
            std::cerr << "[Main] failed to save poses to " << out_path << std::endl;
        }
    }

    return 0;
}