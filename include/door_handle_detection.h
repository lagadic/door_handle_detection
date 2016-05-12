#ifndef __DOOR_HANDLE_DETECTION_H__
#define __DOOR_HANDLE_DETECTION_H__

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <std_msgs/String.h>
#include <visp_bridge/image.h>
#include <visp_bridge/3dpose.h>
#include <visp_bridge/camera.h>
#include <sensor_msgs/SetCameraInfo.h>
#include <visp/vpDisplayX.h>
#include <visp/vpImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <object_recognition_msgs/TableArray.h>
#include <visp3/core/vpTranslationVector.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpRotationMatrix.h>
#include <visp3/core/vpMeterPixelConversion.h>
#include <visp3/core/vpPixelMeterConversion.h>
#include <visp3/core/vpImageMorphology.h>
#include <visp3/core/vpColor.h>
#include <tf/transform_broadcaster.h>
#include <visp/vpRobust.h>

#if defined(Success)
#undef Success
#endif

#include <iostream>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/PointIndices.h>
#include <pcl_msgs/PointIndices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/console/parse.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>

class DoorHandleDetectionNode
{
public:
  DoorHandleDetectionNode(ros::NodeHandle n);
  virtual ~DoorHandleDetectionNode();

public:
  vpColVector getCoeffLineWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  pcl::ModelCoefficients::Ptr getPlaneCoefficients(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  pcl::PointIndices::Ptr getPlaneInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  vpColVector getCenterPCL(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  static double computeX(const vpColVector coeffs, const double y, const double z);
  static double computeY(const vpColVector coeffs, const double x, const double z);
  static double computeZ(const vpColVector coeffs, const double x, const double y);
  vpHomogeneousMatrix createTFPlane(const vpColVector coeffs, const double x, const double y, const double z);
  pcl::PointCloud<pcl::PointXYZ>::Ptr createPlaneFromInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, const pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr createPCLBetweenTwoPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, vpColVector coefficients, double h_min, double h_max);
  void displayImage(const sensor_msgs::Image::ConstPtr& image);
  void spin();
  void getCoeffPlaneWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const double centroidx, const double centroidy, const double centroidz, vpColVector normal);
  void mainComputation(const sensor_msgs::PointCloud2::ConstPtr &image);
  void segColor(const sensor_msgs::PointCloud2::ConstPtr &image);
  void createTFLine(const vpColVector coeffs, vpColVector normal, const double x, const double y, const double z, const vpRotationMatrix cRp, const vpHomogeneousMatrix cMp);

protected:
  ros::NodeHandle n;
  ros::Publisher pcl_plane_pub;
  ros::Publisher pcl_dh_pub;
  ros::Publisher door_handle_final_pub;
  ros::Subscriber pcl_frame_sub;
  std::string pclTopicName;
  bool m_is_previous_initialized;
  vpColVector m_direction_line_previous;
  vpColVector m_direction_line_pre_previous;
  vpColVector m_centroidDH_previous;
  vpColVector m_centroidDH_pre_previous;
};

#endif
