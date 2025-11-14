# ALOAM

参考[HKUST-Aerial-Robotics/A-LOAM: Advanced implementation of LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM)

### 依赖库

atomic [标准库] 原子操作库 

mutex [标准库] 互斥锁库

thread [标准库] 线程库

condition_variable [标准库] 条件变量库

Eigen3 [第三方库] 线性代数库 version 3.4.0

ceres-solver[第三方库] 非线性优化库 version 2.2.0

pangolin [第三方库] opengl 3维可视化库 version 0.9.4

### 备注

可以替换函数使用ICP标准算法

``` C++
    pcl::PointCloud<pcl::PointXYZI>::Ptr lastCloud_;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeLast_;

void LaserOdometryThread::matchAndOptimize(const FeatureCloud& currFrame)
{
    // 如果没有上一帧，直接初始化并返回（和你原逻辑保持一致）
    if (lastCorner_->empty()) {
        *lastCorner_ = *currFrame.cornerSharp;
        *lastSurf_ = *currFrame.surfFlat;
        // 合并为上一帧参考点云
        lastCloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        *lastCloud_ += *lastCorner_;
        *lastCloud_ += *lastSurf_;
        kdtreeLast_->setInputCloud(lastCloud_);
        std::cout << "[LaserOdometryThread][ICP] First frame initialized." << std::endl;
        return;
    }

    // --- 组装当前帧总点云（corner + surf） ---
    pcl::PointCloud<pcl::PointXYZI>::Ptr currCloud(new pcl::PointCloud<pcl::PointXYZI>());
    *currCloud += *currFrame.cornerSharp;
    *currCloud += *currFrame.surfFlat;

    // 如果上一帧的KD树没有数据，则构建
    if (!lastCloud_ || lastCloud_->empty()) {
        lastCloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        *lastCloud_ += *lastCorner_;
        *lastCloud_ += *lastSurf_;
        kdtreeLast_->setInputCloud(lastCloud_);
    }

    // ICP 参数（可调）
    const int    MAX_ICP_ITER = 30;
    const double CONVERGE_TRANSLATION = 1e-4; // m
    const double CONVERGE_ROTATION = 1e-6;    // rad (approx)
    const double DISTANCE_REJECT_THRESH = 1.0; // max correspondence distance (m)
    const int    MIN_CORRESPONDENCES = 20;

    // 初始化变换（4x4 单位）
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

    // 工作副本：把当前点云拷贝到 workingCloud，用于每次迭代变换
    pcl::PointCloud<pcl::PointXYZI>::Ptr workingCloud(new pcl::PointCloud<pcl::PointXYZI>());
    *workingCloud = *currCloud;

    // KD-tree for last cloud (target)
    kdtreeLast_->setInputCloud(lastCloud_);

    int total_correspondences = 0;

    for (int iter = 0; iter < MAX_ICP_ITER; ++iter) {
        std::vector<Eigen::Vector3f> src_pts; // points from workingCloud (source)
        std::vector<Eigen::Vector3f> tgt_pts; // corresponding points from lastCloud_ (target)

        src_pts.reserve(workingCloud->size());
        tgt_pts.reserve(workingCloud->size());

        // 对每个源点查找目标最近点
        std::vector<int> nn_indices(1);
        std::vector<float> nn_dists(1);

        for (size_t i = 0; i < workingCloud->size(); ++i) {
            const auto& p = workingCloud->points[i];
            pcl::PointXYZI query;
            query.x = p.x; query.y = p.y; query.z = p.z; query.intensity = p.intensity;

            int found = kdtreeLast_->nearestKSearch(query, 1, nn_indices, nn_dists);
            if (found <= 0) continue;
            if (nn_dists.empty()) continue;

            // 距离筛选（去除过远的对应）
            if (nn_dists[0] > DISTANCE_REJECT_THRESH * DISTANCE_REJECT_THRESH) continue;

            const auto& q = lastCloud_->points[nn_indices[0]];

            Eigen::Vector3f ps(p.x, p.y, p.z);
            Eigen::Vector3f pt(q.x, q.y, q.z);

            src_pts.push_back(ps);
            tgt_pts.push_back(pt);
        }

        total_correspondences = static_cast<int>(src_pts.size());

        // 若对应数太少，退出
        if (total_correspondences < MIN_CORRESPONDENCES) {
            std::cout << "[LaserOdometryThread][ICP] Too few correspondences: " << total_correspondences << std::endl;
            break;
        }

        // --- 计算质心 ---
        Eigen::Vector3f src_mean = Eigen::Vector3f::Zero();
        Eigen::Vector3f tgt_mean = Eigen::Vector3f::Zero();
        for (int i = 0; i < total_correspondences; ++i) {
            src_mean += src_pts[i];
            tgt_mean += tgt_pts[i];
        }
        src_mean /= float(total_correspondences);
        tgt_mean /= float(total_correspondences);

        // --- 计算协方差矩阵 H = sum (p_i - mean_p) * (q_i - mean_q)^T ---
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        for (int i = 0; i < total_correspondences; ++i) {
            Eigen::Vector3f psd = src_pts[i] - src_mean;
            Eigen::Vector3f qtd = tgt_pts[i] - tgt_mean;
            H += psd * qtd.transpose();
        }

        // SVD 分解
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f V = svd.matrixV();

        Eigen::Matrix3f R = V * U.transpose();

        // 处理奇异情况：保证 det(R)=1（刚体变换）
        if (R.determinant() < 0) {
            Eigen::Matrix3f V2 = V;
            V2.col(2) *= -1.0f;
            R = V2 * U.transpose();
        }

        Eigen::Vector3f t = tgt_mean - R * src_mean;

        // 构建本次迭代的 4x4 变换（作用于 workingCloud）
        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3, 3>(0, 0) = R;
        delta.block<3, 1>(0, 3) = t;

        // 应用变换到 workingCloud（更新源）
        for (size_t i = 0; i < workingCloud->size(); ++i) {
            Eigen::Vector4f p;
            p << workingCloud->points[i].x, workingCloud->points[i].y, workingCloud->points[i].z, 1.0f;
            Eigen::Vector4f p2 = delta * p;
            workingCloud->points[i].x = p2.x();
            workingCloud->points[i].y = p2.y();
            workingCloud->points[i].z = p2.z();
        }

        // 更新整体变换 T = delta * T (注意变换顺序，T 表示从当前原始帧到目标帧的变换)
        T = delta * T;

        // 收敛判断：平移与旋转是否很小
        double trans_norm = t.norm();
        // 计算角度： acos((trace(R)-1)/2)
        double traceR = R.trace();
        double angle = std::max(-1.0, std::min(1.0, (traceR - 1.0) / 2.0));
        double rot_ang = std::acos(angle); // rad

        if (trans_norm < CONVERGE_TRANSLATION && rot_ang < CONVERGE_ROTATION) {
            // 已收敛
            break;
        }
    } // end ICP iterations

    // 若找不到对应或没有足够的对应，仍以 T = Identity 不更新 poseWorld_
    if (total_correspondences < MIN_CORRESPONDENCES) {
        std::cout << "[LaserOdometryThread][ICP] Insufficient correspondences, skip pose update." << std::endl;

        // 输出估计（原始poseWorld_）
        OdomResult result;
        result.pose = poseWorld_;
        result.cornerCount = static_cast<int>(currFrame.cornerSharp->size());
        result.planeCount = static_cast<int>(currFrame.surfFlat->size());
        result.registeredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        *result.registeredCloud += *currFrame.cornerSharp;
        *result.registeredCloud += *currFrame.surfFlat;
        outputQueue_.push(result);

        // 更新 last frame
        *lastCorner_ = *currFrame.cornerSharp;
        *lastSurf_ = *currFrame.surfFlat;
        *lastCloud_ = *workingCloud; // 也可以更新为 currCloud
        kdtreeLast_->setInputCloud(lastCloud_);

        return;
    }

    // --- Step: 将 4x4 T 转换为 Eigen::Matrix4f（已经是 Matrix4f）并更新 poseWorld_ ---
    // 注意：这里我们的 delta 是把源(当前帧原始) -> 目标(上一帧)，
    // 若 poseWorld_ 表示 世界 <- 上一帧 的变换，则更新方式为 poseWorld_ = poseWorld_ * T_last_curr
    // 需要根据你的 poseWorld_ 定义来选择正确的乘法顺序。这里按你原来代码的顺序：
    poseWorld_ = poseWorld_ * T;

    // --- Step: 输出结果（和你原来保持字段一致） ---
    OdomResult result;
    result.pose = poseWorld_;
    result.cornerCount = total_correspondences; // 使用对应数作为一个粗略指标
    result.planeCount = 0; // ICP 不区分 plane / corner
    result.registeredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    *result.registeredCloud += *currFrame.cornerSharp;
    *result.registeredCloud += *currFrame.surfFlat;
    outputQueue_.push(result);

    // --- 更新上一帧点云为当前帧（为下一次匹配做参考） ---
    *lastCorner_ = *currFrame.cornerSharp;
    *lastSurf_ = *currFrame.surfFlat;
    lastCloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    *lastCloud_ += *lastCorner_;
    *lastCloud_ += *lastSurf_;
    kdtreeLast_->setInputCloud(lastCloud_);

    // Done
    std::cout << "[LaserOdometryThread][ICP] correspondences=" << total_correspondences
        << ", updated pose." << std::endl;
}

```

### 下阶段计划

将std::cout转换为日志输出

添加Qt依赖

添加单元测试

```C++
int main() {
    SafeQueue<pcl::PointCloud<pcl::PointXYZI>::Ptr> rawCloudQueue;
    SafeQueue<FeatureCloud> featureQueue;
    SafeQueue<OdomResult> odomQueue;
    //SafeQueue<MapResult> mapQueue;

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

            // 根据你的处理速度决定是否 sleep（可短暂等待以让处理线程追上）
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
```
