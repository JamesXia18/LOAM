#include "feature_extraction.h"
#include "laser_odometry.h"
#include "laser_mapping.h"

#include <pangolin/pangolin.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>

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

    // 数据结构用于可视化（在主线程）
    std::vector<Eigen::Matrix4f> poses;
    pcl::PointCloud<pcl::PointXYZI>::Ptr mapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    std::mutex dataMutex;

    // feeder thread: simulate input and push into rawCloudQueue
    std::thread feeder([&rawCloudQueue]() {
        // 使用原始字符串字面量，避免转义错误
        std::string folder = R"(F:\桌面\sequences\00\velodyne\)";
        for (int i = 0; i < 100; ++i) {
            std::string filename = folder + (std::to_string(i).insert(0, 6 - std::to_string(i).length(), '0')) + ".bin";
            std::vector<float> lidar_data = read_lidar_data(filename);

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
            for (size_t k = 0; k + 3 < lidar_data.size(); k += 4) {
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

    auto drawTrajectory = [&](void) {
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

    auto drawPointCloud = [&](void) {
        std::lock_guard<std::mutex> lock(dataMutex);
        if (!mapCloud || mapCloud->empty()) return;
        glPointSize(2);
        glBegin(GL_POINTS);
        // draw at most a subset if too many points
        const size_t max_draw = 200000;
        size_t step = std::max<size_t>(1, mapCloud->size() / max_draw);
        for (size_t i = 0; i < mapCloud->size(); i += step) {
            const auto &p = mapCloud->points[i];
            float c = std::min(1.0f, std::max(0.0f, (p.z + 5.0f) / 10.0f));
            glColor3f(0.3f * c + 0.1f, 0.8f * (1.0f - c) + 0.1f, 1.0f * c + 0.1f);
            glVertex3f(p.x, p.y, p.z);
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
            poses.push_back(mr.pose);
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

        // draw small axes for last few poses
        {
            std::lock_guard<std::mutex> lock(dataMutex);
            int N = std::min<int>((int)poses.size(), 8);
            for (int i = (int)poses.size() - N; i < (int)poses.size(); ++i) {
                if (i >= 0) glDrawAxis(poses[i], 0.5f);
            }
        }

        drawTrajectory();
        drawPointCloud();

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // window closed -> stop threads
    featureThread.stop();
    odomThread.stop();
    mappingThread.stop();

    if (feeder.joinable()) feeder.join();

    return 0;
}
