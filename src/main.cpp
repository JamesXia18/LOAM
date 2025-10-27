#include <iostream>
#include <fstream>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

std::string path = "F:\\桌面\\sequences\\00\\velodyne\\000000.bin";

std::vector<float> read_lidar_data(const std::string lidar_data_path) {
    // Velodyne的点云为二进制文件
    // 向前为x，向左为y，向上为z
    // 每行存储了x(m),y(m),z(m),intensity(0-255无量纲;强度值)
    // Ex: 7E 97 53 42 FA 54 BC 3C 49 BE FF 3F 0A D7 A3 3D
    // float是4个字节,bin文件是小端序:数据的低位字节存储在低地址处，而高位字节存储在高地址处
    // 7E 97 53 42 -> 0x 42 53 97 7E -> 
    // 0100 0010 0101 0011 1001 0111 0111 1110
    // 符号位0 指数：1000 0100 尾数:101 0011 1001 0111 0111 1110 = 52.8979416
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    // 先将流指针指向ios::end
    lidar_data_file.seekg(0, std::ios::end);
    // 计算文件总共包含多少个float值
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    // 将流指针重置到开始
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements * sizeof(float));
    return lidar_data_buffer;
}


int main(int argc,char* argv[]) {
    
    std::vector<float> lidar_data = read_lidar_data(path);
    std::cout << "total: " << lidar_data.size() / 4.0 << std::endl;

    pcl::PointCloud<pcl::PointXYZI> laser_cloud;
    for (size_t i = 0; i < lidar_data.size(); i += 4) {
        pcl::PointXYZI point;
        point.x = lidar_data[i];
        point.y = lidar_data[i+1];
        point.z = lidar_data[i+2];
        point.intensity = lidar_data[i + 3];
        laser_cloud.push_back(point);
    }
    laser_cloud.width = laser_cloud.points.size();
    laser_cloud.height = 1;
    pcl::io::savePLYFileASCII("output_ascii.ply", laser_cloud);

    std::cout << "New World!";

    return 0;
}
