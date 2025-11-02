# ALOAM



### 依赖库

atomic [标准库] 原子操作库 

mutex [标准库] 互斥锁库

thread [标准库] 线程库

condition_variable [标准库] 条件变量库

Eigen3 [第三方库] 线性代数库 version 3.4.0

ceres-solver[第三方库] 非线性优化库 version 2.2.0

ICP算法

``` C++
void LaserOdometryThread::matchAndOptimize(const FeatureCloud& currFrame)
{
    // 初帧直接建立KD-Tree
    if (lastCorner_->empty()) {
        *lastCorner_ = *currFrame.cornerLessSharp;
        *lastSurf_ = *currFrame.surfLessFlat;
        kdtreeCorner_->setInputCloud(lastCorner_);
        kdtreeSurf_->setInputCloud(lastSurf_);
        return;
    }

    // 构建kd-tree以进行最近邻搜索
    kdtreeCorner_->setInputCloud(lastCorner_);
    kdtreeSurf_->setInputCloud(lastSurf_);

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

    // 简化版：使用最近邻ICP匹配
    int iter = 0;
    for (; iter < 5; ++iter) {
        std::vector<Eigen::Vector3f> srcPts;
        std::vector<Eigen::Vector3f> tgtPts;

        // 角点匹配
        for (auto& p : currFrame.cornerLessSharp->points) {
            std::vector<int> idx;
            std::vector<float> dist;
            if (kdtreeCorner_->nearestKSearch(p, 1, idx, dist) > 0 && dist[0] < 0.5) {
                Eigen::Vector3f ps(p.x, p.y, p.z);
                Eigen::Vector3f pt(lastCorner_->points[idx[0]].x, lastCorner_->points[idx[0]].y, lastCorner_->points[idx[0]].z);
                srcPts.push_back(ps);
                tgtPts.push_back(pt);
            }
        }

        // 面点匹配
        for (auto& p : currFrame.surfLessFlat->points) {
            std::vector<int> idx;
            std::vector<float> dist;
            if (kdtreeSurf_->nearestKSearch(p, 1, idx, dist) > 0 && dist[0] < 0.5) {
                Eigen::Vector3f ps(p.x, p.y, p.z);
                Eigen::Vector3f pt(lastSurf_->points[idx[0]].x, lastSurf_->points[idx[0]].y, lastSurf_->points[idx[0]].z);
                srcPts.push_back(ps);
                tgtPts.push_back(pt);
            }
        }

        if (srcPts.size() < 10) {
            std::cout << "[LaserOdometry] Not enough correspondences, skip frame." << std::endl;
            return;
        }

        // 计算质心
        Eigen::Vector3f srcMean = Eigen::Vector3f::Zero();
        Eigen::Vector3f tgtMean = Eigen::Vector3f::Zero();
        for (size_t i = 0; i < srcPts.size(); ++i) {
            srcMean += srcPts[i];
            tgtMean += tgtPts[i];
        }
        srcMean /= srcPts.size();
        tgtMean /= tgtPts.size();

        // 去均值
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        for (size_t i = 0; i < srcPts.size(); ++i) {
            H += (srcPts[i] - srcMean) * (tgtPts[i] - tgtMean).transpose();
        }

        // SVD 解旋转矩阵
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
        if (R.determinant() < 0) {
            Eigen::Matrix3f V = svd.matrixV();
            V.col(2) *= -1;
            R = V * svd.matrixU().transpose();
        }
        Eigen::Vector3f t = tgtMean - R * srcMean;

        // 更新增量
        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3, 3>(0, 0) = R;
        delta.block<3, 1>(0, 3) = t;

        T = delta * T;

        // 判断收敛
        if (t.norm() < 0.001 && (Eigen::AngleAxisf(R)).angle() < 0.001)
            break;
    }

    poseWorld_ = poseWorld_ * T;

    // 保存当前帧作为下一帧参考
    *lastCorner_ = *currFrame.cornerLessSharp;
    *lastSurf_ = *currFrame.surfLessFlat;

    // 输出
    OdomResult result;
    result.pose = poseWorld_;
    result.registeredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    *result.registeredCloud = *currFrame.surfLessFlat + *currFrame.cornerLessSharp;

    outputQueue_.push(result);

    std::cout << "[LaserOdometry] Iter=" << iter
        << ", pose: \n" << poseWorld_ << std::endl;
}
```

