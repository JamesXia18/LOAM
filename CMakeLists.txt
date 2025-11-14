cmake_minimum_required(VERSION 3.10)
project(ALOAM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres CONFIG REQUIRED)
find_package(Pangolin CONFIG REQUIRED)
# 列出所有 .cpp 源文件
set(SRC_FILES
    src/main.cpp
    src/feature_extraction.cpp
    src/laser_odometry.cpp
    src/laser_mapping.cpp
)

add_executable(ALOAM ${SRC_FILES})

# 添加项目的 include 目录和 PCL 的 include
target_include_directories(ALOAM PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
)

# 如果 PCL 导出了编译定义，传递给目标
if (DEFINED PCL_DEFINITIONS)
    target_compile_definitions(ALOAM PRIVATE ${PCL_DEFINITIONS})
endif()

# 链接库：使用 Ceres::ceres 和 Eigen3::Eigen 目标（推荐）
target_link_libraries(ALOAM PRIVATE 
${PCL_LIBRARIES} 
Ceres::ceres 
Eigen3::Eigen
${Pangolin_LIBRARIES}
)

# 把头文件作为 target_sources 暴露给 IDE
target_sources(ALOAM PRIVATE
    ${CMAKE_SOURCE_DIR}/include/feature_extraction.h
    ${CMAKE_SOURCE_DIR}/include/laser_odometry.h
    ${CMAKE_SOURCE_DIR}/include/laser_mapping.h
    ${CMAKE_SOURCE_DIR}/include/lidarFactor.hpp
    ${CMAKE_SOURCE_DIR}/include/common.hpp
)
