# LOAM系列学习代码

### Reference
[HKUST-Aerial-Robotics/A-LOAM: Advanced implementation of LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM)

### Dependency

Dataset: Kitti odometry velodyne

Standard Library of C++ 17 or C++20

Eigen3: Linear Algebra library version 3.4.0

ceres-solver: Nonlinear optimization library version 2.2.0

pangolin: opengl 3D visualization library version 0.9.4

### Note

The mapping accuracy at the back end is still not very good

### Plan

1. Add more unit tests
2. Cross-platform support, such as docker
3. Command-line program, add json configuration