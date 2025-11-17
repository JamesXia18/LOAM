#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include <filesystem>
#include <queue>
#include <fstream>

static std::vector<std::filesystem::path> list_bin_files(const std::string& folder) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    for (auto& p : fs::directory_iterator(folder)) {
        if (!p.is_regular_file()) continue;
        if (p.path().extension() == ".bin") files.push_back(p.path());
    }

    std::sort(files.begin(), files.end());
    return files;
}


static bool save_poses_kitti(const std::string& out_file, const std::vector<Eigen::Matrix4f>& poses) {
    std::ofstream ofs(out_file);
    if (!ofs.is_open()) return false;
    ofs << std::fixed << std::setprecision(9);
    for (const auto& T : poses) {
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

template <typename T>
class SafeQueue {
public:
    void push(const T& value) {
        std::unique_lock<std::mutex> lock(mtx_);
        q_.push(value);
        lock.unlock();
        cond_.notify_one();
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mtx_);
        cond_.wait(lock, [&] { return !q_.empty() || stop_; });
        if (q_.empty()) return false;
        value = std::move(q_.front());
        q_.pop();
        return true;
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (q_.empty()) return false;
        value = std::move(q_.front());
        q_.pop();
        return true;
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mtx_);
        stop_ = true;
        lock.unlock();
        cond_.notify_all();
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mtx_);
        return q_.empty();
    }

private:
    std::queue<T> q_;
    std::mutex mtx_;
    std::condition_variable cond_;
    bool stop_ = false;
};


struct FeatureCloud {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerSharp;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerLessSharp;
    pcl::PointCloud<pcl::PointXYZI>::Ptr surfFlat;
    pcl::PointCloud<pcl::PointXYZI>::Ptr surfLessFlat;
};

std::vector<float> read_lidar_data(const std::string& lidar_data_path);