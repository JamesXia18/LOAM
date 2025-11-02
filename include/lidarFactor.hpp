#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>

// 表示点到直线的距离残差
struct LidarEdgeFactor {
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,Eigen::Vector3d last_point_b_) 
		: curr_point(curr_point_), 
		  last_point_a(last_point_a_), 
		  last_point_b(last_point_b_)
	{
		// PASS
	}

	/// <summary>
	/// costFunction
	/// </summary>
	/// <typeparam name="T">数据类型: 例如double</typeparam>
	/// <param name="q">待优化变量，表示旋转四元数</param>
	/// <param name="t">待优化变量，表示平移向量</param>
	/// <param name="residual">残差</param>
	/// <returns></returns>
	template<typename T>
	bool operator()(const T* q, const T* t, T* residual) const {
		// 将Vector3d转换成 Eigen::Matrix<T, 3, 1>
		Eigen::Matrix<T, 3, 1> cp{ T(curr_point.x()), T(curr_point.y()), T(curr_point.z()) };
		Eigen::Matrix<T, 3, 1> lpa{ T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z()) };
		Eigen::Matrix<T, 3, 1> lpb{ T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z()) };

		Eigen::Quaternion<T> q_last_curr( q[3], q[0], q[1], q[2] ); // 将待优化四元数double数组转换为Eigen的四元数类型
		Eigen::Matrix<T, 3, 1> t_last_curr{ t[0], t[1], t[2] }; // 平移转换

		// 通过四元数和平移向量变换到上一帧的LiDAR坐标系
		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		residual[0] = nu.x() / de.norm();
		residual[1] = nu.y() / de.norm();
		residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction* Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
		const Eigen::Vector3d last_point_b_)
	{
		return (new ceres::AutoDiffCostFunction<
			LidarEdgeFactor, 3, 4, 3>(
				new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_)));
	}

private:

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
};

// 表示点到平面的距离残差
struct LidarPlaneFactor {

	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
		Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		last_point_m(last_point_m_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	/// <summary>
	/// 
	/// </summary>
	/// <typeparam name="T">数据类型: 例如double</typeparam>
	/// <param name="q">待优化变量，表示旋转四元数</param>
	/// <param name="t">待优化变量，表示平移向量</param>
	/// <param name="residual">残差</param>
	/// <returns></returns>
	template <typename T>
	bool operator()(const T* q, const T* t, T* residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{ T(curr_point.x()), T(curr_point.y()), T(curr_point.z()) };
		Eigen::Matrix<T, 3, 1> lpj{ T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z()) };
		Eigen::Matrix<T, 3, 1> ljm{ T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z()) };

		Eigen::Quaternion<T> q_last_curr{ q[3], q[0], q[1], q[2] };
		Eigen::Matrix<T, 3, 1> t_last_curr{ t[0], t[1], t[2] };

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		residual[0] = (lp - lpj).dot(ljm);

		return true;
	}

	static ceres::CostFunction* Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
		const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_)
	{
		return (new ceres::AutoDiffCostFunction<
			LidarPlaneFactor, 1, 4, 3>(
				new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_)));
	}

private:
	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm; // l,j,m 平面法向量
};