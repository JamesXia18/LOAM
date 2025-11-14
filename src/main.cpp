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

static std::vector<std::filesystem::path> list_bin_files(const std::string& folder) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    for (auto& p : fs::directory_iterator(folder)) {
        if (!p.is_regular_file()) continue;
        if (p.path().extension() == ".bin") files.push_back(p.path());
    }
    // sort by filename (natural lexicographic works for KITTI zero-padded names)
    std::sort(files.begin(), files.end());
    return files;
}


static bool save_poses_kitti(const std::string& out_file, const std::vector<Eigen::Matrix4f>& poses) {
    std::ofstream ofs(out_file);
    if (!ofs.is_open()) return false;
    ofs << std::fixed << std::setprecision(9);
    for (const auto& T : poses) {
        // write row-major 3x4 as doubles
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                double v = static_cast<double>(T(r, c));
                ofs << v;
                if (!(r == 2 && c == 3)) ofs << " ";
            }
        }
        ofs << "\n";
    }
    ofs.close();
    return true;
}

int main() {
    Eigen::Matrix4f Tr;
    Tr << 4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
        -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
        9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
        0, 0, 0, 1;

    SafeQueue<pcl::PointCloud<pcl::PointXYZI>::Ptr> rawCloudQueue;
    SafeQueue<FeatureCloud> featureQueue;
    SafeQueue<OdomResult> odomQueue;

    FeatureExtractionThread featureThread(rawCloudQueue, featureQueue);
    LaserOdometryThread odomThread(featureQueue, odomQueue);

    featureThread.start();
    odomThread.start();

    std::vector<Eigen::Matrix4f> poses;
    std::mutex dataMutex;

    std::string folder = R"(F:\桌面\sequences\00\velodyne\)";
    auto files = list_bin_files(folder);
    if (files.empty()) {
        std::cerr << "[Error] No .bin files found in " << folder << std::endl;
        return -1;
    }
    std::cout << "[Main] found " << files.size() << " velodyne files.\n";

    // feeder thread
    std::thread feeder([&rawCloudQueue, files]() {
        for (size_t idx = 0; idx < files.size(); ++idx) {
            std::string filename = files[idx].string();
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

    // pangolin setup
    pangolin::CreateWindowAndBind("ALOAM Odometry Viewer", 1280, 720);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1280, 720, 500, 500, 640, 360, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -10, -10, 0, 0, 0, 0, -1, 0));
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1280.0f / 720.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    auto glDrawAxis = [](const Eigen::Matrix4f& T, float len = 1.0f) {
        Eigen::Vector3f o = T.block<3, 1>(0, 3);
        Eigen::Vector3f x = o + T.block<3, 1>(0, 0).normalized() * len;
        Eigen::Vector3f y = o + T.block<3, 1>(0, 1).normalized() * len;
        Eigen::Vector3f z = o + T.block<3, 1>(0, 2).normalized() * len;
        glBegin(GL_LINES);
        glColor3f(1, 0, 0); glVertex3f(o.x(), o.y(), o.z()); glVertex3f(x.x(), x.y(), x.z());
        glColor3f(0, 1, 0); glVertex3f(o.x(), o.y(), o.z()); glVertex3f(y.x(), y.y(), y.z());
        glColor3f(0, 0, 1); glVertex3f(o.x(), o.y(), o.z()); glVertex3f(z.x(), z.y(), z.z());
        glEnd();
        };

    auto drawTrajectory = [&](void) {
        std::lock_guard<std::mutex> lock(dataMutex);
        if (poses.empty()) return;
        glColor3f(1.0, 0.0, 0.0);
        glBegin(GL_LINE_STRIP);
        for (auto& T : poses)
            glVertex3f(T(0, 3), T(1, 3), T(2, 3));
        glEnd();
        };

    // main render loop
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 直接从 odomQueue 取位姿
        OdomResult odom;
        int got = 0;
        while (odomQueue.try_pop(odom)) {
            std::lock_guard<std::mutex> lock(dataMutex);
            poses.push_back(odom.pose);
            ++got;
        }
        if (got > 0)
            std::cout << "[Viewer] pulled " << got << " OdomResult(s), total=" << poses.size() << std::endl;

        d_cam.Activate(s_cam);
        glDrawAxis(Eigen::Matrix4f::Identity(), 2.0f);
        drawTrajectory();

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    featureThread.stop();
    odomThread.stop();
    if (feeder.joinable()) feeder.join();

    std::string out_path = R"(F:\桌面\KITTI_00_est_cam.txt)";

    {
        std::lock_guard<std::mutex> lock(dataMutex);
        std::vector<Eigen::Matrix4f> camPoses;
        camPoses.reserve(poses.size());
        for (const auto& T_lidar : poses) {
            camPoses.push_back(Tr * T_lidar);
        }
        if (save_poses_kitti(out_path, camPoses))
            std::cout << "[Main] saved " << camPoses.size() << " camera poses to " << out_path << std::endl;
        else
            std::cerr << "[Main] failed to save poses!\n";
    }

    return 0;
}
