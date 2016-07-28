#include "door_handle_detection.h"
#include <visp3/io/vpImageIo.h>

#include <std_msgs/Int8.h>

void initKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt)
{
  KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
  cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(8e-6));       // set process noise
  cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(6e-5));   // set measurement noise
  cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance
  /* DYNAMIC MODEL */
  //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
  //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
  // position
  KF.transitionMatrix.at<double>(0,3) = dt;
  KF.transitionMatrix.at<double>(1,4) = dt;
  KF.transitionMatrix.at<double>(2,5) = dt;
  KF.transitionMatrix.at<double>(3,6) = dt;
  KF.transitionMatrix.at<double>(4,7) = dt;
  KF.transitionMatrix.at<double>(5,8) = dt;
  KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);
  // orientation
  KF.transitionMatrix.at<double>(9,12) = dt;
  KF.transitionMatrix.at<double>(10,13) = dt;
  KF.transitionMatrix.at<double>(11,14) = dt;
  KF.transitionMatrix.at<double>(12,15) = dt;
  KF.transitionMatrix.at<double>(13,16) = dt;
  KF.transitionMatrix.at<double>(14,17) = dt;
  KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);
  /* MEASUREMENT MODEL */
  //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
  KF.measurementMatrix.at<double>(0,0) = 1;  // x
  KF.measurementMatrix.at<double>(1,1) = 1;  // y
  KF.measurementMatrix.at<double>(2,2) = 1;  // z
  KF.measurementMatrix.at<double>(3,9) = 1;  // roll
  KF.measurementMatrix.at<double>(4,10) = 1; // pitch
  KF.measurementMatrix.at<double>(5,11) = 1; // yaw
}

// Converts a given Rotation Matrix to Euler angles
cv::Mat rot2euler(const cv::Mat & rotationMatrix)
{
  cv::Mat euler(3,1,CV_64F);

  double m00 = rotationMatrix.at<double>(0,0);
  double m02 = rotationMatrix.at<double>(0,2);
  double m10 = rotationMatrix.at<double>(1,0);
  double m11 = rotationMatrix.at<double>(1,1);
  double m12 = rotationMatrix.at<double>(1,2);
  double m20 = rotationMatrix.at<double>(2,0);
  double m22 = rotationMatrix.at<double>(2,2);

  double x, y, z;

  // Assuming the angles are in radians.
  if (m10 > 0.998) { // singularity at north pole
    x = 0;
    y = CV_PI/2;
    z = atan2(m02,m22);
  }
  else if (m10 < -0.998) { // singularity at south pole
    x = 0;
    y = -CV_PI/2;
    z = atan2(m02,m22);
  }
  else
  {
    x = atan2(-m12,m11);
    y = asin(m10);
    z = atan2(-m20,m00);
  }

  euler.at<double>(0) = x;
  euler.at<double>(1) = y;
  euler.at<double>(2) = z;

  return euler;
}

// Converts a given Euler angles to Rotation Matrix
cv::Mat euler2rot(const cv::Mat & euler)
{
  cv::Mat rotationMatrix(3,3,CV_64F);

  double x = euler.at<double>(0);
  double y = euler.at<double>(1);
  double z = euler.at<double>(2);

  // Assuming the angles are in radians.
  double ch = cos(z);
  double sh = sin(z);
  double ca = cos(y);
  double sa = sin(y);
  double cb = cos(x);
  double sb = sin(x);

  double m00, m01, m02, m10, m11, m12, m20, m21, m22;

  m00 = ch * ca;
  m01 = sh*sb - ch*sa*cb;
  m02 = ch*sa*sb + sh*cb;
  m10 = sa;
  m11 = ca*cb;
  m12 = -ca*sb;
  m20 = -sh*ca;
  m21 = sh*sa*cb + ch*sb;
  m22 = -sh*sa*sb + ch*cb;

  rotationMatrix.at<double>(0,0) = m00;
  rotationMatrix.at<double>(0,1) = m01;
  rotationMatrix.at<double>(0,2) = m02;
  rotationMatrix.at<double>(1,0) = m10;
  rotationMatrix.at<double>(1,1) = m11;
  rotationMatrix.at<double>(1,2) = m12;
  rotationMatrix.at<double>(2,0) = m20;
  rotationMatrix.at<double>(2,1) = m21;
  rotationMatrix.at<double>(2,2) = m22;

  return rotationMatrix;
}

void fillMeasurements( cv::Mat &measurements, const cv::Mat &translation_measured, const cv::Mat &rotation_measured)
{
  // Convert rotation matrix to euler angles
  cv::Mat measured_eulers(3, 1, CV_64F);
  measured_eulers = rot2euler(rotation_measured);
  // Set measurement to predict
  measurements.at<double>(0) = translation_measured.at<double>(0); // x
  measurements.at<double>(1) = translation_measured.at<double>(1); // y
  measurements.at<double>(2) = translation_measured.at<double>(2); // z
  measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
  measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
  measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}

void updateKalmanFilter( cv::KalmanFilter &KF, cv::Mat &measurement, cv::Mat &translation_estimated, cv::Mat &rotation_estimated )
{
  // First predict, to update the internal statePre variable
  cv::Mat prediction = KF.predict();
  // The "correct" phase that is going to use the predicted value and our measurement
  cv::Mat estimated = KF.correct(measurement);

  // Estimated translation
  translation_estimated.at<double>(0) = estimated.at<double>(0);
  translation_estimated.at<double>(1) = estimated.at<double>(1);
  translation_estimated.at<double>(2) = estimated.at<double>(2);
  // Estimated euler angles
  cv::Mat eulers_estimated(3, 1, CV_64F);
  eulers_estimated.at<double>(0) = estimated.at<double>(9);
  eulers_estimated.at<double>(1) = estimated.at<double>(10);
  eulers_estimated.at<double>(2) = estimated.at<double>(11);
  // Convert estimated quaternion to rotation matrix
  rotation_estimated = euler2rot(eulers_estimated);
}

DoorHandleDetectionNode::DoorHandleDetectionNode(ros::NodeHandle nh)
{
  n = nh;

  m_img_.init(480,640);
  m_img_mono.init(480,640);

  m_extrinsic_param_are_initialized = false;
  m_tracking_is_initialized = false;
  m_disp_is_initialized = false;
  m_cam_is_initialized = false;

  m_is_door_handle_present = 0;
  m_tracking_works = false;
  m_stop_detection = false;


  n.param<std::string>("imageTopicName", m_imageTopicName, "/camera/rgb/image_raw");
  n.param<std::string>("pclTopicName", m_pclTopicName, "/softkinetic_camera/depth/points");
  n.param<std::string>("cameraRGBTopicName", m_cameraRGBTopicName, "/softkinetic_camera/rgb/camera_info");
  n.param<std::string>("cameraDepthTopicName", m_cameraDepthTopicName, "/softkinetic_camera/depth/camera_info");
  n.param("dh_right", m_dh_right, true);
  n.param("lenght_dh",m_lenght_dh, 0.1);
  n.param("height_dh", m_height_dh, 0.055);
  n.param("debug", debug, false);

  //Subscribe to services
  cam_rgb_info_sub = n.subscribe( m_cameraRGBTopicName, 1, (boost::function < void(const sensor_msgs::CameraInfo::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::setupCameraParameters, this, _1 ));
  cam_depth_info_sub = n.subscribe( m_cameraDepthTopicName, 1, (boost::function < void(const sensor_msgs::CameraInfo::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::getExtrinsicParameters, this, _1 ));
  pcl_frame_sub = n.subscribe( m_pclTopicName, 1, (boost::function < void(const sensor_msgs::PointCloud2::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::mainComputation, this, _1 ));
  image_frame_sub = n.subscribe( m_imageTopicName, 1, (boost::function < void(const sensor_msgs::Image::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::displayImage, this, _1 ));

  ROS_INFO("Beautiful weather, isn't it ?");
  //Advertise the publishers
  pose_handle_pub = n.advertise< geometry_msgs::PoseStamped >("pose_handle", 1);
  door_handle_status_pub = n.advertise< std_msgs::Int8 >("status", 1);
  point_handle_pub = n.advertise< geometry_msgs::PointStamped >("point_handle", 1);
  if (debug)
  {
    pcl_plane_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PC_Plane", 1);
    pcl_dh_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PC_Sandwich", 1);
    debug_pcl_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("pc_debug", 1);
  }

  m_blob.setGraphics(true);
  m_blob.setGraphicsThickness(1);
  //  m_blob.setEllipsoidShapePrecision(0.);
  m_blob.setGrayLevelMin(170);
  m_blob.setGrayLevelMax(255);

  //Kalman Filter
  int nStates = 18;            // the number of states
  int nMeasurements = 6;       // the number of measured states
  int nInputs = 0;             // the number of action control
  double dt = 1/12.5;           // time between measurements (1/FPS)
  initKalmanFilter(m_KF, nStates, nMeasurements, nInputs, dt);    // init function
}

DoorHandleDetectionNode::~DoorHandleDetectionNode()
{
  if (m_disp_is_initialized)
  {
    delete m_disp;
    delete m_disp_mono;
  }

}


void DoorHandleDetectionNode::spin()
{
  ros::Rate loop_rate(100);
  while(ros::ok()){
    ros::spinOnce();
    loop_rate.sleep();
  }
}

void DoorHandleDetectionNode::mainComputation(const sensor_msgs::PointCloud2::ConstPtr &image)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bbox(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sandwich(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_debug(new pcl::PointCloud<pcl::PointXYZ>);
  vpColVector direction_line(3);
  vpColVector normal(3);
  vpHomogeneousMatrix dMp;
  geometry_msgs::PoseStamped cMdh_msg;
  vpColVector coeffs;
  struct inliersAndCoefficients plane_variables;
  pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
  std_msgs::Int8 status_handle_msg;

  //Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
  pcl::fromROSMsg (*image, *cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

  //Downsample the point cloud given by the camera
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud( cloud );
  vg.setLeafSize( 0.005f, 0.005f, 0.005f );
  vg.filter(*cloud_filtered);


  if(m_cam_is_initialized)
  {

    //Find the coefficients of the biggest plane of the point cloud
    plane_variables = DoorHandleDetectionNode::getPlaneInliersAndCoefficients(cloud_filtered);
    coeffs.stack(plane_variables.coefficients->values[0]);
    coeffs.stack(plane_variables.coefficients->values[1]);
    coeffs.stack(plane_variables.coefficients->values[2]);
    coeffs.stack(plane_variables.coefficients->values[3]);

    //Invert the normal in order to have the z axis toward the camera
    normal[0] = -coeffs[0];
    normal[1] = -coeffs[1];
    normal[2] = -coeffs[2];

    //Check if the size of the plane if big enough
    if (plane_variables.inliers->indices.size() < 100)
    {
      std::cout << "Could not find a plane in the scene." << std::endl;
      vpDisplay::displayText(m_img_, 60, 5, "No plane detected", vpColor::red);
      vpDisplay::flush(m_img_);
    }
    else
    {
      if (debug)
      {
        //Copy the inliers of the plane to a new cloud.
        plane = DoorHandleDetectionNode::createPlanePC(cloud, plane_variables.inliers, plane_variables.coefficients);

        //Create the center of the plane
        vpColVector centroidPlane(3);
        centroidPlane = DoorHandleDetectionNode::getCentroidPCL(plane);

        //Create a tf for the plane
        dMp = DoorHandleDetectionNode::createTFPlane(coeffs, centroidPlane[0], centroidPlane[1], centroidPlane[2]);
      }

      //Compute the Z of the corners of the smaller detection
      m_Z_topleft = -(coeffs[3] + m_height_dh)/(coeffs[0]*m_x_min + coeffs[1]*m_y_min + coeffs[2]) ;
      m_Z_topright = -(coeffs[3] + m_height_dh)/(coeffs[0]*m_x_max + coeffs[1]*m_y_min + coeffs[2]) ;
      m_Z_bottomright = -(coeffs[3] + m_height_dh)/(coeffs[0]*m_x_max + coeffs[1]*m_y_max + coeffs[2]) ;
      m_Z_bottomleft = -(coeffs[3] + m_height_dh)/(coeffs[0]*m_x_min + coeffs[1]*m_y_max + coeffs[2]) ;

      //Creating a cloud with all the points of the door handle
      if (m_tracking_works)
      {
        cloud_bbox = DoorHandleDetectionNode::getOnlyUsefulHandle(cloud);
        cloud_sandwich = DoorHandleDetectionNode::createPCLSandwich(cloud_bbox, coeffs);
      }
      else
        cloud_sandwich = DoorHandleDetectionNode::createPCLSandwich(cloud, coeffs);

      if (debug)
      {
        //Create a point cloud of only the top left and bottom right points used for the narrow detection
        cloud_debug->width = 2;
        cloud_debug->height = 1;
        cloud_debug->points.resize (cloud_debug->width * cloud_debug->height);
        cloud_debug->points[0].x = m_X_min;
        cloud_debug->points[0].y = m_Y_min;
        cloud_debug->points[0].z = m_Z_topleft;
        cloud_debug->points[1].x = m_X_max;
        cloud_debug->points[1].y = m_Y_max;
        cloud_debug->points[1].z = m_Z_bottomright;
        cloud_debug->header.frame_id = m_parent_depth_tf;
        debug_pcl_pub.publish(cloud_debug);

        //Publish the point cloud of the door handle with noise
        cloud_sandwich->header.frame_id = m_parent_depth_tf;
        pcl_dh_pub.publish(*cloud_sandwich);

        //Publish the point clouds of the plane
        plane->header.frame_id = m_parent_depth_tf;
        pcl_plane_pub.publish(*plane);

      }

      //Check if the size of the point cloud of the door handle if big enough to localize it or not and check if the detection should be done or not
      if (cloud_sandwich->size()<50 || m_stop_detection){
        m_is_door_handle_present = 0;
        ROS_INFO_STREAM("No door handle detected : " << cloud_sandwich->size());
        m_tracking_works = false;
        vpPixelMeterConversion::convertPoint(m_cam_depth, 0, 0, m_x_min, m_y_min);
        vpPixelMeterConversion::convertPoint(m_cam_depth, m_img_mono.getWidth() - 1, m_img_mono.getHeight() - 1, m_x_max, m_y_max);
      }
      else
      {
        m_is_door_handle_present = 1;

        //Get the axis from the cloud sandwich with the odr method
        direction_line = getCoeffLineWithODR(cloud_sandwich);

        //Set the axis of the door handle towards the right way
        if (m_dh_right)
        {
          if (direction_line[0]<0)
          {
            direction_line[0] = -direction_line[0];
            direction_line[1] = -direction_line[1];
            direction_line[2] = -direction_line[2];
          }
        }
        else
        {
          if (direction_line[0]>0)
          {
            direction_line[0] = -direction_line[0];
            direction_line[1] = -direction_line[1];
            direction_line[2] = -direction_line[2];
          }
        }

        //Create the black&white image used to reduce the detection
        DoorHandleDetectionNode::morphoSandwich(cloud_sandwich );

        //Get the centroid of the door handle
        vpColVector centroidDH = DoorHandleDetectionNode::getCentroidPCL(cloud_sandwich);

        //Compute the pose of the door handle with respect to the Depth camera and publish a tf of it
        m_dMh = DoorHandleDetectionNode::createTFLine(direction_line, normal, centroidDH[0], centroidDH[1], centroidDH[2]);
        vpHomogeneousMatrix cMd;
        for(unsigned int i=0; i<3; i++)
          cMd[i][3] = m_extrinsicParam[i];

        //Compute the pose of the door handle with respect to the RGB camera
        m_cMh = cMd * m_dMh;

        //Kalman Filter
        vpTranslationVector T_cMh = m_cMh.getTranslationVector();
        vpRotationMatrix R_cMh = m_cMh.getRotationMatrix();
        cv::Mat translation_measured(3, 1, CV_64F);
        cv::Mat rotation_measured(3, 3, CV_64F);
        for(int i = 0; i < 3; i++){
          for(int j = 0; j < 3; j++){
            rotation_measured.at<double>(i,j) = R_cMh[i][j];
          }
          translation_measured.at<double>(i) = T_cMh[i];
        }
        cv::Mat measurements(6, 1, CV_64F);
        // fill the measurements vector
        fillMeasurements(measurements, translation_measured, rotation_measured);

        // Instantiate estimated translation and rotation
        cv::Mat translation_estimated(3, 1, CV_64F);
        cv::Mat rotation_estimated(3, 3, CV_64F);
        // update the Kalman filter with good measurements
        updateKalmanFilter( m_KF, measurements, translation_estimated, rotation_estimated);
        for(int i = 0; i < 3; i++){
          for(int j = 0; j < 3; j++){
            m_cMh_filtered_kalman[i][j] = rotation_estimated.at<double>(i,j) ;
          }
          m_cMh_filtered_kalman[i][3] = translation_estimated.at<double>(i);
        }

        //Publish the pose of the handle with respect to the camera
        cMdh_msg.header.stamp = ros::Time::now();
        cMdh_msg.pose = visp_bridge::toGeometryMsgsPose(m_cMh_filtered_kalman);
        cMdh_msg.header.frame_id = m_parent_rgb_tf;
        pose_handle_pub.publish(cMdh_msg);

      }

    }

  }

  //Publish the status of the door handle : 0 if no door handle, 1 if a door handle is found
  status_handle_msg.data = m_is_door_handle_present;
  door_handle_status_pub.publish( status_handle_msg );

}

vpHomogeneousMatrix DoorHandleDetectionNode::createTFLine(const vpColVector direction_axis, vpColVector normal, const double x, const double y, const double z)
{
  vpRotationMatrix dRdh;
  vpHomogeneousMatrix dMdh;
  vpHomogeneousMatrix dMdh_border;
  geometry_msgs::Pose dMdh_msg_border;
  tf::Transform transformdh;
  static tf::TransformBroadcaster br;
  vpColVector direction_line(3);
  vpColVector centroid;
  vpColVector centroidBorder;

  centroid.stack(x);
  centroid.stack(y);
  centroid.stack(z);
  centroid.stack(1);

  //Normalize the direction axis
  direction_line[0]=direction_axis[0];
  direction_line[1]=direction_axis[1];
  direction_line[2]=direction_axis[2];
  direction_line.normalize();

  vpColVector y_dh(3);
  y_dh = vpColVector::cross(normal,direction_line);

  //Create the Rotation Matrix
  dRdh[0][0] = direction_line[0];
  dRdh[1][0] = direction_line[1];
  dRdh[2][0] = direction_line[2];
  dRdh[0][1] = y_dh[0];
  dRdh[1][1] = y_dh[1];
  dRdh[2][1] = y_dh[2];
  dRdh[0][2] = normal[0];
  dRdh[1][2] = normal[1];
  dRdh[2][2] = normal[2];

  //Create the pose of the handle with respect to the depth camera in the cog of the handle
  dMdh = vpHomogeneousMatrix(vpTranslationVector(x, y, z), dRdh);

  //Put the pose of the handle in its rotation axis instead of its cog
  centroidBorder = dMdh.inverse() * centroid;
  centroidBorder[0] = centroidBorder[0] - (m_lenght_dh / 2) + 0.015;
  centroidBorder = dMdh * centroidBorder;

  //Create the pose of the handle with respect to the depth camera in the rotation axis of the handle
  dMdh_border = vpHomogeneousMatrix(vpTranslationVector(centroidBorder[0], centroidBorder[1], centroidBorder[2]), dRdh);
  dMdh_msg_border = visp_bridge::toGeometryMsgsPose(dMdh_border);

  //Publish the tf of the handle with respect to the depth camera
  transformdh.setOrigin( tf::Vector3(centroidBorder[0], centroidBorder[1], centroidBorder[2] ));

  tf::Quaternion qdh;
  qdh.setX(dMdh_msg_border.orientation.x);
  qdh.setY(dMdh_msg_border.orientation.y);
  qdh.setZ(dMdh_msg_border.orientation.z);
  qdh.setW(dMdh_msg_border.orientation.w);

  transformdh.setRotation(qdh);
  br.sendTransform(tf::StampedTransform(transformdh, ros::Time::now(), m_parent_depth_tf, "door_handle_tf"));

  return dMdh_border;
}

inliersAndCoefficients DoorHandleDetectionNode::getPlaneInliersAndCoefficients(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  struct inliersAndCoefficients plane;
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

  //Use a plane segmentation to find the inliers and coefficients for the biggest plan found
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);
  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);

  plane.coefficients = coefficients;
  plane.inliers = inliers;

  return plane;
}

vpHomogeneousMatrix DoorHandleDetectionNode::createTFPlane(const vpColVector coeffs, const double x, const double y, const double z)
{
  vpColVector xp(3);
  vpColVector yp(3);
  vpColVector normal(3);
  vpRotationMatrix dRp;
  vpTranslationVector P0;
  vpHomogeneousMatrix dMp;
  geometry_msgs::Pose dMp_msg;
  tf::Transform transform;
  static tf::TransformBroadcaster br;

  //Create a normal to the plan from the coefficients
  normal[0] = -coeffs[0];
  normal[1] = -coeffs[1];
  normal[2] = -coeffs[2];

  normal.normalize();

  //Create a xp vector that is following the equation of the plane
  xp[0] = - (coeffs[1]*y + coeffs[2]*(z+0.05) + coeffs[3]) / (coeffs[0]) - x;
  xp[1] = 0;
  xp[2] = 0.05;
  xp.normalize();
  //Create a yp vector with the normal and xp
  yp = vpColVector::cross(normal,xp);

  //Create the Rotation Matrix
  dRp[0][0] = xp[0];
  dRp[1][0] = xp[1];
  dRp[2][0] = xp[2];
  dRp[0][1] = yp[0];
  dRp[1][1] = yp[1];
  dRp[2][1] = yp[2];
  dRp[0][2] = normal[0];
  dRp[1][2] = normal[1];
  dRp[2][2] = normal[2];

  transform.setOrigin( tf::Vector3(x, y, z) );

  //Calculate the z0 for the translation vector
  double z0 = -(coeffs[0]*x + coeffs[1]*y + coeffs[3])/(coeffs[2]);

  //Create the translation Vector
  P0 = vpTranslationVector(x, y, z0);

  //Create the homogeneous Matrix
  dMp = vpHomogeneousMatrix(P0, dRp);

  //Publish the tf of the plane
  dMp_msg = visp_bridge::toGeometryMsgsPose(dMp);
  tf::Quaternion q;
  q.setX(dMp_msg.orientation.x);
  q.setY(dMp_msg.orientation.y);
  q.setZ(dMp_msg.orientation.z);
  q.setW(dMp_msg.orientation.w);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), m_parent_depth_tf, "tf_plane"));

  return dMp;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr DoorHandleDetectionNode::createPlanePC(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, const pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr convexHull(new pcl::PointCloud<pcl::PointXYZ>);

  //Object for retrieving the convex hull.
  pcl::ConvexHull<pcl::PointXYZ> hull;
  hull.setInputCloud(plane);
  hull.reconstruct(*convexHull);

  //Create the point cloud to display the plane segmented
  pcl::ProjectInliers<pcl::PointXYZ> proj;
  proj.setModelType (pcl::SACMODEL_PLANE);
  proj.setInputCloud (cloud);
  proj.setModelCoefficients (coefficients);
  proj.filter (*plane);

  return plane;
}


vpColVector DoorHandleDetectionNode::getCentroidPCL(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  vpColVector centroid(3);
  double sumx = 0, sumy = 0, sumz = 0;
  for (int i = 0; i < cloud->size(); i++){
    sumx += cloud->points[i].x;
    sumy += cloud->points[i].y;
    sumz += cloud->points[i].z;
  }
  centroid[0] = sumx/cloud->size();
  centroid[1] = sumy/cloud->size();
  centroid[2] = sumz/cloud->size();

  return centroid;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr DoorHandleDetectionNode::createPCLSandwich(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, vpColVector coefficients)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outside(new pcl::PointCloud<pcl::PointXYZ>);
  cloud_outside->width = 300000;
  cloud_outside->height = 1;
  cloud_outside->points.resize (cloud_outside->width * cloud_outside->height);

  double xc, yc, zc, z_min, z_max;
  size_t width_outside = 0;

  for(int i = 0; i < cloud->size(); i++)
  {
    if (cloud->points[i].z !=0)
    {
      xc = cloud->points[i].x;
      yc = cloud->points[i].y;
      zc = cloud->points[i].z;

      //Create a zmin and zmax for every point to check if the point is outside or inside the detection
      z_min = -(coefficients[0]*xc + coefficients[1]*yc + (coefficients[3] + m_height_dh + 0.02) )/(coefficients[2]);
      z_max = -(coefficients[0]*xc + coefficients[1]*yc + (coefficients[3] + m_height_dh - 0.02) )/(coefficients[2]);

      //If the point is inside, we add it to the new point cloud
      if (cloud->points[i].z > z_min && cloud->points[i].z < z_max )
      {
        width_outside++;
        cloud_outside->points[width_outside-1].x = xc;
        cloud_outside->points[width_outside-1].y = yc;
        cloud_outside->points[width_outside-1].z = zc;
      }
    }
  }
  cloud_outside->width = width_outside;
  cloud_outside->points.resize (cloud_outside->width * cloud_outside->height);

  return cloud_outside;
}

void DoorHandleDetectionNode::setupCameraParameters(const sensor_msgs::CameraInfoConstPtr &cam_rgb)
{
  if (! m_cam_is_initialized && m_extrinsic_param_are_initialized) {
    //Init the RGB camera parameters
    m_cam_rgb = visp_bridge::toVispCameraParameters(*cam_rgb);
    ROS_INFO_STREAM("RGB camera param  = \n" << m_cam_rgb);
    m_parent_rgb_tf = cam_rgb->header.frame_id;

    m_cam_is_initialized = true;
    cam_rgb_info_sub.shutdown();
  }
}

void DoorHandleDetectionNode::getExtrinsicParameters(const sensor_msgs::CameraInfoConstPtr &cam_depth)
{
  if (! m_extrinsic_param_are_initialized) {
    //Init extrinsic parameters between Depth and RGB
    m_extrinsicParam[0] = cam_depth->P[3];
    m_extrinsicParam[1] = cam_depth->P[7];
    m_extrinsicParam[2] = cam_depth->P[11];
    ROS_INFO_STREAM("Extrinsic param  = \n" << m_extrinsicParam);

    //Init the depth camera parameters
    m_cam_depth = visp_bridge::toVispCameraParameters(*cam_depth);
    vpPixelMeterConversion::convertPoint(m_cam_depth, 0, 0, m_x_min, m_y_min);
    vpPixelMeterConversion::convertPoint(m_cam_depth, m_img_mono.getWidth() - 1, m_img_mono.getHeight() - 1, m_x_max, m_y_max);

    m_parent_depth_tf = cam_depth->header.frame_id;

    cam_depth_info_sub.shutdown();
    m_extrinsic_param_are_initialized = true;
  }
}

void DoorHandleDetectionNode::initDisplayVisp()
{
  if (! m_disp_is_initialized) {
    //Init graphical interface
    m_disp = new vpDisplayX(m_img_mono, 750, 0, "Image black&white viewer");
    vpDisplay::flush(m_img_mono);
    vpDisplay::display(m_img_mono);
    vpDisplay::flush(m_img_mono);

    m_disp_mono = new vpDisplayX();
    m_disp_mono->init(m_img_);
    m_disp_mono->setTitle("Image RGB viewer");
    vpDisplay::flush(m_img_);
    vpDisplay::display(m_img_);
    vpDisplay::flush(m_img_);

    m_disp_is_initialized = true;
  }

}

void DoorHandleDetectionNode::displayImage(const sensor_msgs::Image::ConstPtr& image)
{
  initDisplayVisp();
  vpMouseButton::vpMouseButtonType button;
  vpImagePoint pointClicked;
  geometry_msgs::PointStamped point_handle;
  double X, Y, Z, x, y, u, v;

  m_img_ = visp_bridge::toVispImageRGBa(*image);

  vpDisplay::display(m_img_);

  //Right click to select a point to initialize the tracking of a blob, middle click to stop the detection, left click to shutdown the node
  if ( vpDisplay::getClick( m_img_, pointClicked, button, false) )
  {
    if (button == vpMouseButton::button1)
    {
      m_pointPoseHandle = pointClicked;

      m_stop_detection = false;
      m_tracking_is_initialized = false;
      m_tracking_works = false;

    }
    else if(button == vpMouseButton::button2)
    {
      m_tracking_is_initialized = false;
      m_tracking_works = false;
      m_stop_detection = true;

      vpPixelMeterConversion::convertPoint(m_cam_depth, 0, 0, m_x_min, m_y_min);
      vpPixelMeterConversion::convertPoint(m_cam_depth, m_img_mono.getWidth() - 1, m_img_mono.getHeight() - 1, m_x_max, m_y_max);
    }
    else if(button == vpMouseButton::button3)
      ros::shutdown();

  }
  else if ( m_is_door_handle_present )
  {
    //cMh is on the handle rotation axis, we put it on the cog
    vpHomogeneousMatrix hMhcog(m_lenght_dh / 2,0,0,0,0,0);
    vpHomogeneousMatrix cMhcog = m_cMh* hMhcog;
    vpTranslationVector c_t_hcog = cMhcog.getTranslationVector();
    double x_hcog = c_t_hcog[0]/c_t_hcog[2];
    double y_hcog = c_t_hcog[1]/c_t_hcog[2];

    vpMeterPixelConversion::convertPoint(m_cam_rgb, x_hcog, y_hcog, m_pointPoseHandle);
  }
  if ( m_is_door_handle_present && !m_stop_detection)
  {
    //Convert the 3D point of the handle in pixels
    X = m_cMh[0][3] ;
    Y = m_cMh[1][3] ;
    Z = m_cMh[2][3] ;
    x = X/Z;
    y = Y/Z;
    vpMeterPixelConversion::convertPoint(m_cam_rgb, x, y, u, v);
    //Publish the point of the handle
    point_handle.header.stamp = ros::Time::now();
    point_handle.point.x = u;
    point_handle.point.y = v;
    point_handle.point.z = 0.0;
    point_handle_pub.publish(point_handle);

    //Display the pose of the handle
    vpDisplay::displayFrame(m_img_, m_cMh_filtered_kalman, m_cam_rgb, 0.05, vpColor::none, 2);
  }
  vpDisplay::flush(m_img_);

}

void DoorHandleDetectionNode::morphoSandwich(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  m_img_mono = 0;
  double X,Y,Z,x,y,u,v;
  vpImagePoint bottomRightBBoxHandle;
  vpImagePoint topLeftBBoxHandle;
  vpMouseButton::vpMouseButtonType button;
  vpRect bboxhandle, searchingField;

  //Convert the 3D points in 2D
  for(int i = 0; i < cloud->size(); i++ ){
    X = cloud->points[i].x ;
    Y = cloud->points[i].y ;
    Z = cloud->points[i].z ;
    x = X/Z;
    y = Y/Z;

    //Convert the points in meters to points in pixels
    vpMeterPixelConversion::convertPoint(m_cam_depth, x, y, u, v);

    //Color theses points in white
    if(u < m_img_mono.getWidth()-1 && v < m_img_mono.getHeight()-1 && u > 0 && v > 0 )
      m_img_mono.bitmap[ (int)v * m_img_mono.getWidth() + (int)u ] = 250;
  }

  //Use some dilatation/erosion to have a blob that is easy to track
  vpImageMorphology::dilatation(m_img_mono, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
  vpImageMorphology::dilatation(m_img_mono, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
  vpImageMorphology::erosion(m_img_mono, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
  vpImageMorphology::erosion(m_img_mono, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
  vpImageMorphology::erosion(m_img_mono, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
  vpImageMorphology::dilatation(m_img_mono, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);

  //In case we need to save the images
  if (0){
    static int iter = 0;
    char filename[255];
    sprintf(filename, "/tmp/bheintz/I%04d.png", iter ++);
    vpImageIo::write(m_img_mono, filename);
  }

  //Display and track the white blob inside the image
  vpDisplay::display(m_img_mono);
  if ( !m_stop_detection )
  {
    try {
      if ( m_is_door_handle_present )
      {
        if ( !m_tracking_is_initialized )
        {
          if ( m_pointPoseHandle.get_u() > 0 && m_pointPoseHandle.get_v() > 0 && m_pointPoseHandle.get_u() < m_img_mono.getWidth()-1 && m_pointPoseHandle.get_v() < m_img_mono.getHeight()-1 )
          {
            m_blob.initTracking(m_img_mono, m_pointPoseHandle, 150, 255);
            m_tracking_is_initialized = true;
          }
        }
        else
        {
          m_blob.track(m_img_mono);
          bboxhandle = m_blob.getBBox();
          m_tracking_works = true;
        }
      }
    }
    catch(...) {
      m_tracking_is_initialized = false;
      m_tracking_works = false;
      ROS_INFO_STREAM("Tracking failed");

    }
  }

  //Use the bounding box of the blob tracker to reduce the field of view of the detection
  if ( m_tracking_works )
  {
    topLeftBBoxHandle.set_uv( bboxhandle.getLeft() - bboxhandle.getWidth()/32, bboxhandle.getTop() - bboxhandle.getHeight()/32);
    bottomRightBBoxHandle.set_uv( bboxhandle.getRight() + bboxhandle.getWidth()/32, bboxhandle.getBottom() + bboxhandle.getHeight()/32 );

    vpPixelMeterConversion::convertPoint(m_cam_depth, topLeftBBoxHandle, m_x_min, m_y_min);
    vpPixelMeterConversion::convertPoint(m_cam_depth, bottomRightBBoxHandle, m_x_max, m_y_max);

    searchingField.setTopLeft(topLeftBBoxHandle);
    searchingField.setBottomRight(bottomRightBBoxHandle);
    vpDisplay::displayRectangle(m_img_mono, bboxhandle, vpColor::yellow, 0, 2);
  }
  //If the tracking doesn't work, the field of view is set back to all the image
  else
  {
    vpPixelMeterConversion::convertPoint(m_cam_depth, 0, 0, m_x_min, m_y_min);
    vpPixelMeterConversion::convertPoint(m_cam_depth, m_img_mono.getWidth() - 1, m_img_mono.getHeight() - 1, m_x_max, m_y_max);
  }

  vpDisplay::displayRectangle(m_img_mono, searchingField, vpColor::purple);
  vpDisplay::displayText(m_img_mono, 15, 5, "Left click to quit.", vpColor::red);
  vpDisplay::flush(m_img_mono);
  if(vpDisplay::getClick(m_img_mono, button, false)) {
    if(button == vpMouseButton::button3)
      ros::shutdown();
  }

}

pcl::PointCloud<pcl::PointXYZ>::Ptr DoorHandleDetectionNode::getOnlyUsefulHandle(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_useful(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

  //Select the maximum Top Left and Bottom Right corner for the reduced detection
  if (m_x_min * m_Z_topleft > m_x_min * m_Z_bottomleft)
    m_X_min = m_x_min * m_Z_bottomleft;
  else
    m_X_min = m_x_min * m_Z_topleft;

  if (m_x_max * m_Z_topright < m_x_max * m_Z_bottomright)
    m_X_max = m_x_max * m_Z_bottomright;
  else
    m_X_max = m_x_max * m_Z_topright;

  if (m_y_min * m_Z_topright > m_y_min * m_Z_topleft)
    m_Y_min = m_y_min * m_Z_topleft;
  else
    m_Y_min = m_y_min * m_Z_topright;

  if (m_y_max * m_Z_bottomright < m_y_max * m_Z_bottomleft)
    m_Y_max = m_y_max * m_Z_bottomleft;
  else
    m_Y_max = m_y_max * m_Z_bottomright;

  //Use the passthrough filter to elimate all the point that are not inside the limits
  pcl::PassThrough<pcl::PointXYZ> pass1;
  pass1.setInputCloud (cloud);
  pass1.setFilterFieldName ("x");
  pass1.setFilterLimits (m_X_min, m_X_max);
  pass1.filter (*cloud_filtered);

  pcl::PassThrough<pcl::PointXYZ> pass2;
  pass2.setInputCloud (cloud_filtered);
  pass2.setFilterFieldName ("y");
  pass2.setFilterLimits (m_Y_min, m_Y_max);
  pass2.filter (*cloud_useful);

  return cloud_useful;
}

vpColVector DoorHandleDetectionNode::getCoeffLineWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  vpMatrix M(cloud->size(),3);
  vpRowVector m(3);
  vpColVector centroid = DoorHandleDetectionNode::getCentroidPCL(cloud);

  //Create a Matrix(n,3) with the coordinates of all the points
  for(unsigned int i=0; i<cloud->size(); i++) {
    m[0] = cloud->points[i].x - centroid[0];
    m[1] = cloud->points[i].y - centroid[1];
    m[2] = cloud->points[i].z - centroid[2];
    for(unsigned int j=0;j<3;j++)
      M[i][j] = m[j];
  }

  vpMatrix A = M.t() * M;

  vpColVector D;
  vpMatrix V;
  A.svd(D, V);

  double largestSv = D[0];
  unsigned int indexLargestSv = 0 ;
  for (unsigned int i = 1; i < D.size(); i++) {
    if ((D[i] > largestSv) ) {
      largestSv = D[i];
      indexLargestSv = i;
    }
  }

  vpColVector h = V.getCol(indexLargestSv);

  return h;
}

int main( int argc, char** argv )
{
  ros::init(argc, argv, "door_handle_detection");
  ros::NodeHandle n(std::string("~"));

  DoorHandleDetectionNode *node = new DoorHandleDetectionNode(n);

  node->spin();

  delete node;

  printf( "\nQuitting... \n" );
  return 0;
}
