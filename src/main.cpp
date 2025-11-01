#include "feature_extraction.h"
#include <pcl/io/ply_io.h>


int main(int argc, char* argv[]) {

	std::string path = "F:\\桌面\\sequences\\00\\velodyne\\000001.bin";
	std::vector<float> lidar_data = read_lidar_data(path);

	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
	for (size_t i = 0; i < lidar_data.size(); i += 4) {
		pcl::PointXYZI p;
		p.x = lidar_data[i];
		p.y = lidar_data[i + 1];
		p.z = lidar_data[i + 2];
		p.intensity = lidar_data[i + 3];
		cloud->push_back(p);
	}

	FeatureCloud features = extractFeatures(cloud);
	std::cout << "Feature extraction done!" << std::endl;


	return 0;
}