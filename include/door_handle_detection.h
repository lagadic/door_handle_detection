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
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpXmlParserCamera.h>
#include <tf/transform_broadcaster.h>
#include <visp/vpRobust.h>
#include <visp3/core/vpImagePoint.h>

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
  pcl::ModelCoefficients::Ptr getPlaneCoefficients(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  pcl::PointIndices::Ptr getPlaneInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  vpColVector getCoeffLineWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  vpColVector getCoeffPlaneWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const double centroidx, const double centroidy, const double centroidz, vpColVector normal);
  static double computeX(const vpColVector coeffs, const double y, const double z);
  static double computeY(const vpColVector coeffs, const double x, const double z);
  static double computeZ(const vpColVector coeffs, const double x, const double y);
  vpColVector getCenterPCL(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr getOnlyUsefulCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  vpHomogeneousMatrix createTFPlane(const vpColVector coeffs, const double x, const double y, const double z);
  vpHomogeneousMatrix createTFLine(const vpColVector coeffs, vpColVector normal, const double x, const double y, const double z, const vpRotationMatrix cRp, const vpHomogeneousMatrix cMp);
  pcl::PointCloud<pcl::PointXYZ>::Ptr createPlaneFromInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, const pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr createPCLBetweenTwoPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, vpColVector coefficients, double height_dh);
  void spin();
  void getImageVisp(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  void initDisplayVisp();
  void setupCameraParameters(const sensor_msgs::CameraInfoConstPtr &cam_rgb);
  void getExtrinsicParameters(const sensor_msgs::CameraInfoConstPtr &cam_depth);
  void mainComputation(const sensor_msgs::PointCloud2::ConstPtr &image);
  void displayImage(const sensor_msgs::Image::ConstPtr& image);

protected:
  ros::NodeHandle n;
  ros::Publisher homogeneous_matrix_pub;
  ros::Publisher pcl_plane_pub;
  ros::Publisher pcl_dh_pub;
  ros::Publisher door_handle_final_pub;
  ros::Subscriber pcl_frame_sub;
  ros::Subscriber image_frame_sub;
  ros::Subscriber cam_rgb_info_sub;
  ros::Subscriber cam_depth_info_sub;
  std::string m_imageTopicName;
  std::string m_pclTopicName;
  std::string m_cameraRGBTopicName;
  std::string m_cameraDepthTopicName;
  bool m_is_previous_initialized;
  bool m_dh_right;
  std::string m_parent_tf;
  vpColVector m_direction_line_previous;
  vpColVector m_direction_line_pre_previous;
  vpColVector m_centroidDH_previous;
  vpColVector m_centroidDH_pre_previous;
  double m_lenght_dh;
  double m_height_dh;
  double m_x_min;
  double m_x_max;
  double m_y_min;
  double m_y_max;
  double m_X_min;
  double m_X_center;
  double m_X_max;
  double m_Y_min;
  double m_Y_center;
  double m_Y_max;
  double m_Z_min;
  double m_Z_center;
  double m_Z_max;
  bool m_is_door_handle_present;
  bool m_cam_is_initialized;
  bool m_plane_is_initialized;
  bool m_useful_cloud_is_initialized;
  bool m_disp_is_initialized;
  bool m_extrinsic_param_are_initialized;
  bool m_bbox_is_fixed;
  vpImage<unsigned char> m_img_;
  vpImage<unsigned char> m_img_mono;
  vpCameraParameters m_cam_rgb;
  vpTranslationVector m_extrinsicParam;
  vpHomogeneousMatrix m_dMh;
  vpHomogeneousMatrix m_cMh;
  vpRect m_bboxhandle;
  vpRect m_bboxplane;
  vpDisplay* m_disp;
  vpDisplay* m_disp_mono;
  bool m_soft;
};

#endif
