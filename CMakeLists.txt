cmake_minimum_required(VERSION 3.10)
project(ALOAM)

set(CMAKE_CXX_STANDARD 20)

find_package(PCL CONFIG REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} include)

add_executable(ALOAM src/main.cpp src/feature_extraction.cpp)

target_link_libraries(ALOAM PRIVATE ${PCL_LIBRARIES})
