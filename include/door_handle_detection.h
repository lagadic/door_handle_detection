#ifndef __DOOR_HANDLE_DETECTION_H__
#define __DOOR_HANDLE_DETECTION_H__

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/SetCameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>

//ViSP
#include <visp/vpDisplayX.h>
#include <visp/vpImage.h>
#include <visp_bridge/image.h>
#include <visp_bridge/3dpose.h>
#include <visp_bridge/camera.h>
#include <visp3/core/vpTranslationVector.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpRotationMatrix.h>
#include <visp3/core/vpMeterPixelConversion.h>
#include <visp3/core/vpPixelMeterConversion.h>
#include <visp3/core/vpImageMorphology.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpXmlParserCamera.h>
#include <visp3/core/vpImagePoint.h>
#include <visp3/blob/vpDot2.h>
#include <visp3/blob/vpDot.h>
#include <visp3/core/vpPolygon.h>
#include <tf/transform_broadcaster.h>

//PCL
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>

//Kalman Filter
#include <opencv2/video/tracking.hpp>


struct inliersAndCoefficients
{
  pcl::ModelCoefficients::Ptr coefficients;
  pcl::PointIndices::Ptr inliers;
};

class DoorHandleDetectionNode
{
public:
  DoorHandleDetectionNode(ros::NodeHandle n);
  virtual ~DoorHandleDetectionNode();

public:
  vpHomogeneousMatrix createTFPlane(const vpColVector coeffs, const double x, const double y, const double z);
  vpHomogeneousMatrix createTFLine(const vpColVector direction_axis, vpColVector normal, const double x, const double y, const double z);
  pcl::PointCloud<pcl::PointXYZ>::Ptr createPlanePC(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, const pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr createPCLSandwich(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, vpColVector coefficients);
  void displayImage(const sensor_msgs::Image::ConstPtr& image);
  vpColVector getCentroidPCL(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  vpColVector getCoeffLineWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  void getExtrinsicParameters(const sensor_msgs::CameraInfoConstPtr &cam_depth);
  void morphoSandwich(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr getOnlyUsefulHandle(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  inliersAndCoefficients getPlaneInliersAndCoefficients(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  void initDisplayVisp();
  void mainComputation(const sensor_msgs::PointCloud2::ConstPtr &image);
  void setupCameraParameters(const sensor_msgs::CameraInfoConstPtr &cam_rgb);
  void spin();

protected:
  //ROS
  ros::NodeHandle n;
  ros::Publisher pose_handle_pub;
  ros::Publisher point_handle_pub;
  ros::Publisher pcl_plane_pub;
  ros::Publisher pcl_dh_pub;
  ros::Publisher door_handle_status_pub;
  ros::Publisher debug_pcl_pub;
  ros::Subscriber pcl_frame_sub;
  ros::Subscriber image_frame_sub;
  ros::Subscriber cam_rgb_info_sub;
  ros::Subscriber cam_depth_info_sub;
  std::string m_imageTopicName;
  std::string m_pclTopicName;
  std::string m_cameraRGBTopicName;
  std::string m_cameraDepthTopicName;

  //Dimensions of the door handle
  double m_lenght_dh;
  double m_height_dh;

  //Initialisations
  bool m_cam_is_initialized;
  bool m_tracking_is_initialized;
  bool m_disp_is_initialized;
  bool m_extrinsic_param_are_initialized;

  //Status
  int m_is_door_handle_present;
  bool debug;
  bool m_dh_right;
  bool m_tracking_works;
  bool m_stop_detection;

  //Images and Diplays
  vpImage<unsigned char> m_img_mono;
  vpImage<vpRGBa> m_img_;
  vpDisplay* m_disp_mono;
  vpDisplay* m_disp;

  //Camera parameters
  vpCameraParameters m_cam_rgb;
  vpCameraParameters m_cam_depth;
  vpTranslationVector m_extrinsicParam;
  std::string m_parent_depth_tf;
  std::string m_parent_rgb_tf;

  //Poses and point of the door handle
  vpHomogeneousMatrix m_dMh;
  vpHomogeneousMatrix m_cMh;
  vpHomogeneousMatrix m_cMh_filtered_kalman;
  vpImagePoint m_pointPoseHandle;

  //Kalman Filter
  cv::KalmanFilter m_KF;

  //Limits to reduce the field of view for the detection of the handle
  double m_x_min;
  double m_x_max;
  double m_y_min;
  double m_y_max;
  double m_X_min;
  double m_X_max;
  double m_Y_min;
  double m_Y_max;
  double m_Z_topleft;
  double m_Z_topright;
  double m_Z_bottomleft;
  double m_Z_bottomright;

  //Blob used for tracking
  vpDot m_blob;

};

#endif
