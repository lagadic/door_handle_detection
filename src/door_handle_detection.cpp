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
#if 1
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
#else
    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);

    // Estimated euler angles
    cv::Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = prediction.at<double>(9);
    eulers_estimated.at<double>(1) = prediction.at<double>(10);
    eulers_estimated.at<double>(2) = prediction.at<double>(11);
    // Convert estimated quaternion to rotation matrix
    rotation_estimated = euler2rot(eulers_estimated);
#endif
}

DoorHandleDetectionNode::DoorHandleDetectionNode(ros::NodeHandle nh)
{
    n = nh;

    m_img_.init(480,640);
    m_img_2.init(480,640);
    m_img_mono.init(480,640);
    ROS_INFO("Launch Test ros node");

    m_Z = 0.5;
    m_extrinsic_param_are_initialized = false;
    m_useful_cloud_is_initialized = false;
    m_is_previous_initialized = false;
    m_tracking_is_initialized = false;
    m_is_door_handle_present = 0;
    m_plane_is_initialized =false;
    m_disp_is_initialized = false;
    m_cam_is_initialized = false;
    m_tracking_works = false;
    m_bbox_is_fixed = false;
    m_stop_detection = false;


    n.param<std::string>("imageTopicName", m_imageTopicName, "/camera/rgb/image_raw");
    n.param<std::string>("pclTopicName", m_pclTopicName, "/softkinetic_camera/depth/points");
    n.param<std::string>("cameraRGBTopicName", m_cameraRGBTopicName, "/softkinetic_camera/rgb/camera_info");
    n.param<std::string>("cameraDepthTopicName", m_cameraDepthTopicName, "/softkinetic_camera/depth/camera_info");
    n.param<std::string>("parent_tf", m_parent_tf, "softkinetic_camera_link");
    n.param("dh_right", m_dh_right, true);
    n.param("debug", debug, false);
    n.param("lenght_dh",m_lenght_dh, 0.1);
    n.param("height_dh", m_height_dh, 0.055);
    //  n.param("m_plane_is_initialized", m_plane_is_initialized, false);
    //  n.param("m_Z_max", m_Z_center, 0.8);
    // subscribe to services
    ROS_INFO("Beautiful weather, isn't it ?");
    pose_handle_pub = n.advertise< geometry_msgs::PoseStamped >("pose_handle", 1);
    point_handle_pub = n.advertise< geometry_msgs::PointStamped >("point_handle", 1);
    if (debug)
    {
        pcl_plane_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PCL_Plane", 1);
        pcl_dh_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PCL_Sandwich", 1);
    }
    door_handle_status_pub = n.advertise< std_msgs::Int8 >("status", 1);
    door_handle_final_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PCL_DH", 1);
    cam_rgb_info_sub = n.subscribe( m_cameraRGBTopicName, 1, (boost::function < void(const sensor_msgs::CameraInfo::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::setupCameraParameters, this, _1 ));
    cam_depth_info_sub = n.subscribe( m_cameraDepthTopicName, 1, (boost::function < void(const sensor_msgs::CameraInfo::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::getExtrinsicParameters, this, _1 ));
    pcl_frame_sub = n.subscribe( m_pclTopicName, 1, (boost::function < void(const sensor_msgs::PointCloud2::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::mainComputation, this, _1 ));
    image_frame_sub = n.subscribe( m_imageTopicName, 1, (boost::function < void(const sensor_msgs::Image::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::displayImage, this, _1 ));

    debug_pcl_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("pcl_debug", 1);

    m_blob.setGraphics(true);
    m_blob.setGraphicsThickness(1);
//    m_blob.setEllipsoidShapePrecision(0.);
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
        delete m_disp2;
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_handle (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_debug(new pcl::PointCloud<pcl::PointXYZ>);
    vpColVector direction_line(3);
    vpColVector normal(3);
    vpRotationMatrix cRp;
    vpHomogeneousMatrix cMp;
    vpHomogeneousMatrix cMdh;
    geometry_msgs::PoseStamped cMdh_msg;
    vpColVector coeffs;
    struct inliersAndCoefficients plane_variables;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
    std_msgs::Int8 status_handle_msg;
//    ros::Time time_init;
//    ros::Time begin = ros::Time::now();
    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    pcl::fromROSMsg (*image, *cloud);
//    ROS_INFO_STREAM(" Time taken until now :" << ros::Time::now() - begin);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud( cloud );
    vg.setLeafSize( 0.005f, 0.005f, 0.005f );
    vg.filter(*cloud_filtered);

//    ROS_INFO_STREAM("After voxel : Time taken until now :" << ros::Time::now() - begin);

    if(m_cam_is_initialized)
    {

//        time_init = ros::Time::now();
        plane_variables = DoorHandleDetectionNode::getPlaneInliersAndCoefficients(cloud_filtered);
//        time_taken = ros::Time::now() - time_init;
//        ROS_INFO_STREAM("Time taken for the segmentation of the plane :" << ros::Time::now() - time_init<< " Time taken until now :" << ros::Time::now() - begin);
        coeffs.stack(plane_variables.coefficients->values[0]);
        coeffs.stack(plane_variables.coefficients->values[1]);
        coeffs.stack(plane_variables.coefficients->values[2]);
        coeffs.stack(plane_variables.coefficients->values[3]);

        normal[0] = -coeffs[0];
        normal[1] = -coeffs[1];
        normal[2] = -coeffs[2];

        if (plane_variables.inliers->indices.size() < 100)
        {
            std::cout << "Could not find a plane in the scene." << std::endl;
            vpDisplay::displayText(m_img_mono, 60, 5, "No plane detected", vpColor::red);
            vpDisplay::flush(m_img_mono);
        }
        else
        {
            if (debug)
            {
                // Copy the inliers of the plane to a new cloud.
                plane = DoorHandleDetectionNode::createPlaneFromInliers(cloud, plane_variables.inliers, plane_variables.coefficients);

                //Create the center of the plane
                vpColVector centroidPlane(3);
                centroidPlane = DoorHandleDetectionNode::getCenterPCL(plane);

                //Create a tf for the plane
                cMp = DoorHandleDetectionNode::createTFPlane(coeffs, centroidPlane[0], centroidPlane[1], centroidPlane[2]);
            }

            m_Z = -coeffs[3]/(coeffs[0]*m_x_min + coeffs[1]*m_y_min + coeffs[2]) - m_height_dh;
            m_Z -= m_extrinsicParam[2];
//            ROS_INFO("z = %lf", m_Z);

//            time_init = ros::Time::now();
            cloud_bbox = DoorHandleDetectionNode::getOnlyUsefulHandle(cloud);
//            ROS_INFO_STREAM("Time taken for the cloud useful :" << ros::Time::now() - time_init<< " Time taken until now :" << ros::Time::now() - begin);
            if (debug)
            {
                cloud_debug->width = 2;
                cloud_debug->height = 1;
                cloud_debug->points.resize (cloud_debug->width * cloud_debug->height);
                cloud_debug->points[0].x = m_X_min;
                cloud_debug->points[0].y = m_Y_min;
                cloud_debug->points[0].z = m_Z;
                cloud_debug->points[1].x = m_X_max;
                cloud_debug->points[1].y = m_Y_max;
                cloud_debug->points[1].z = m_Z;
                cloud_debug->header.frame_id = m_parent_tf;
                debug_pcl_pub.publish(cloud_debug);
            }
            //Creating a cloud with all the points of the door handle
//            time_init = ros::Time::now();
            cloud_sandwich = DoorHandleDetectionNode::createPCLBetweenTwoPlanes(cloud_bbox, coeffs);
//            ROS_INFO_STREAM("Time taken for the cloud sandwich :" << ros::Time::now() - time_init<< " Time taken until now :" << ros::Time::now() - begin);

            if (debug)
            {
                //Publish the pcl of the door handle with noise
                cloud_sandwich->header.frame_id = m_parent_tf;
                pcl_dh_pub.publish(*cloud_sandwich);
            }

            if (cloud_sandwich->size()<30){
                m_is_door_handle_present = 0;
                vpDisplay::displayText(m_img_mono, 60, 5, "No door Handle detected", vpColor::red);
                vpDisplay::flush(m_img_mono);
                ROS_INFO("No door handle detected : %d", m_is_door_handle_present);
                vpPixelMeterConversion::convertPoint(m_cam_rgb, 0, 0, m_x_min, m_y_min);
                vpPixelMeterConversion::convertPoint(m_cam_rgb, 640, 480, m_x_max, m_y_max);
                m_x_min -= (m_extrinsicParam[0] );
                m_x_max -= (m_extrinsicParam[0] );
                m_y_min -= (m_extrinsicParam[1] );
                m_y_max -= (m_extrinsicParam[1] );
            }
            else
            {
                m_is_door_handle_present = 1;
                std::vector<int> inliers2;
                Eigen::VectorXf Coeff_line;

//                time_init = ros::Time::now();

//                pcl::applyMorphologicalOperator(cloud_sandwich, 0.01, 3, cloud_handle );


                //Create a RandomSampleConsensus object and compute the model of a line
                pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr  model_l (new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cloud_sandwich));
                pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_l);
                ransac.setDistanceThreshold (.026);
                ransac.computeModel();
                ransac.getInliers(inliers2);
                ransac.getModelCoefficients(Coeff_line);
                // copies all inliers of the model computed to another PointCloud
                pcl::copyPointCloud<pcl::PointXYZ>(*cloud_sandwich, inliers2, *cloud_handle);
//                ROS_INFO_STREAM("Time taken for the line :" << ros::Time::now() - time_init << " Time taken until now :" << ros::Time::now() - begin);

                //Get the direction line from the cloud sandwich with the odr method
                direction_line = getCoeffLineWithODR(cloud_handle);
//                if (m_dh_right)
//                {
//                    if (Coeff_line[3]>0)
//                    {
//                        direction_line[0] = Coeff_line[3];
//                        direction_line[1] = Coeff_line[4];
//                        direction_line[2] = Coeff_line[5];
//                    }
//                    else
//                    {
//                        direction_line[0] = -Coeff_line[3];
//                        direction_line[1] = -Coeff_line[4];
//                        direction_line[2] = -Coeff_line[5];
//                    }
//                }
//                else
//                {
//                    if (Coeff_line[3]>0)
//                    {
//                        direction_line[0] = -Coeff_line[3];
//                        direction_line[1] = -Coeff_line[4];
//                        direction_line[2] = -Coeff_line[5];
//                    }
//                    else
//                    {
//                        direction_line[0] = Coeff_line[3];
//                        direction_line[1] = Coeff_line[4];
//                        direction_line[2] = Coeff_line[5];
//                    }

//                }

                if (m_dh_right)
                {
                    if (direction_line[0]>0)
                    {
                        direction_line[0] = direction_line[0];
                        direction_line[1] = direction_line[1];
                        direction_line[2] = direction_line[2];
                    }
                    else
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
                    else
                    {
                        direction_line[0] = direction_line[0];
                        direction_line[1] = direction_line[1];
                        direction_line[2] = direction_line[2];
                    }

                }



                //Publish the pcl of the door handle
                cloud_handle->header.frame_id = m_parent_tf;
                door_handle_final_pub.publish(*cloud_handle);
//                ROS_INFO("DEBUG 3");

//                time_init = ros::Time::now();
//                DoorHandleDetectionNode::morphoSandwich(cloud_handle );
                DoorHandleDetectionNode::morphoSandwich(cloud_sandwich );
//                ROS_INFO_STREAM("Time taken for the display and morphomat :" << ros::Time::now() - time_init << " Time taken until now :" << ros::Time::now() - begin);
//                ROS_INFO("DEBUG 4");

                vpColVector centroidDH = DoorHandleDetectionNode::getCenterPCL(cloud_handle);
//                vpColVector centroidDH = DoorHandleDetectionNode::getCenterPCL(cloud_sandwich);

                //Create the door handle tf
                m_dMh = DoorHandleDetectionNode::createTFLine(direction_line, normal, centroidDH[0], centroidDH[1], centroidDH[2], cRp, cMp);
                vpHomogeneousMatrix cMd;
                for(unsigned int i=0; i<3; i++)
                  cMd[i][3] = m_extrinsicParam[i];

                m_cMh = cMd * m_dMh;

//                m_cMh = m_dMh;
//                m_cMh[0][3] += m_extrinsicParam[0];
//                m_cMh[1][3] += m_extrinsicParam[1];
//                m_cMh[2][3] += m_extrinsicParam[2];
//                std::cout << "Pose :" << m_cMh << std::endl;

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
//                std::cout << "Pose Filtered :" << m_cMh_filtered_kalman << std::endl;

                //Mean Filtering with the t-1 and t-2
                if (!m_is_previous_initialized)
                {
                    m_direction_line_previous = direction_line;
                    m_direction_line_pre_previous = direction_line;
                    m_centroidDH_previous = centroidDH;
                    m_centroidDH_pre_previous = centroidDH;
                    m_is_previous_initialized = true;
                }
                else
                {
                    if (m_direction_line_previous[0] != direction_line[0])
                    {
                        direction_line = (m_direction_line_pre_previous + m_direction_line_previous + direction_line) / 3;
                        //          ROS_INFO("vec[t-2]: %lf %lf %lf", m_direction_line_pre_previous[0], m_direction_line_pre_previous[1], m_direction_line_pre_previous[2]);
                        //          ROS_INFO("vec[t-1]: %lf %lf %lf", m_direction_line_previous[0], m_direction_line_previous[1], m_direction_line_previous[2]);
                        //          ROS_INFO("vec[t]  : %lf %lf %lf", direction_line[0], direction_line[1], direction_line[2]);
                        m_direction_line_pre_previous = m_direction_line_previous;
                        m_direction_line_previous = direction_line;
                    }
                    if (m_centroidDH_previous[0] != centroidDH[0])
                    {
                        centroidDH = (m_centroidDH_pre_previous + m_centroidDH_previous + centroidDH) / 3;
                        m_centroidDH_pre_previous = m_centroidDH_previous;
                        m_centroidDH_previous = centroidDH;
                    }
                }
                m_cMh_filtered_mean = DoorHandleDetectionNode::createTFLine(direction_line, normal, centroidDH[0], centroidDH[1], centroidDH[2], cRp, cMp);
                m_cMh_filtered_mean[0][3] += m_extrinsicParam[0];
                m_cMh_filtered_mean[1][3] += m_extrinsicParam[1];
                m_cMh_filtered_mean[2][3] += m_extrinsicParam[2];

                cMdh_msg.header.stamp = ros::Time::now();
                cMdh_msg.pose = visp_bridge::toGeometryMsgsPose(m_cMh);
                pose_handle_pub.publish(cMdh_msg);

            }


        }

        if (debug)
        {
            //Publish the point clouds of the plan
            plane->header.frame_id = m_parent_tf;
            pcl_plane_pub.publish(*plane);
        }
    }

    status_handle_msg.data = m_is_door_handle_present;
    door_handle_status_pub.publish( status_handle_msg );
//    ROS_INFO_STREAM("Time taken for the total :" << ros::Time::now() - begin);

}

vpHomogeneousMatrix DoorHandleDetectionNode::createTFLine(const vpColVector coeffs, vpColVector normal, const double x, const double y, const double z, const vpRotationMatrix cRp, const vpHomogeneousMatrix cMp)
{
    vpRotationMatrix cRdh;
    vpTranslationVector Pdh;
    vpTranslationVector Pdh_border;
    vpHomogeneousMatrix cMdh;
    vpHomogeneousMatrix cMdh_border;
    vpRotationMatrix pRdh;
    vpHomogeneousMatrix pMdh;
    geometry_msgs::Pose cMdh_msg;
    geometry_msgs::Pose cMdh_msg_border;
    geometry_msgs::Pose pMdh_msg;
    tf::Transform transformdh;
    static tf::TransformBroadcaster br;
    vpColVector direction_line(3);
    vpColVector centroid;
    vpColVector centroidBorder;

    centroid.stack(x);
    centroid.stack(y);
    centroid.stack(z);
    centroid.stack(1);

    direction_line[0]=coeffs[0];
    direction_line[1]=coeffs[1];
    direction_line[2]=coeffs[2];
    direction_line.normalize();

    vpColVector y_dh(3);
    y_dh = vpColVector::cross(normal,direction_line);

    ////Create the Rotation Matrix
    cRdh[0][0] = direction_line[0];
    cRdh[1][0] = direction_line[1];
    cRdh[2][0] = direction_line[2];
    cRdh[0][1] = y_dh[0];
    cRdh[1][1] = y_dh[1];
    cRdh[2][1] = y_dh[2];
    cRdh[0][2] = normal[0];
    cRdh[1][2] = normal[1];
    cRdh[2][2] = normal[2];

    //Create the translation Vector
    Pdh = vpTranslationVector(x, y, z);

    //Create the homogeneous Matrix
    cMdh = vpHomogeneousMatrix(Pdh, cRdh);
    cMdh_msg = visp_bridge::toGeometryMsgsPose(cMdh);

    centroidBorder = cMdh.inverse() * centroid;
    centroidBorder[0] = centroidBorder[0] - (m_lenght_dh / 2) + 0.015;
    centroidBorder = cMdh * centroidBorder;

    transformdh.setOrigin( tf::Vector3(centroidBorder[0], centroidBorder[1], centroidBorder[2] ));
    Pdh_border = vpTranslationVector(centroidBorder[0], centroidBorder[1], centroidBorder[2]);
    cMdh_border = vpHomogeneousMatrix(Pdh_border, cRdh);
    cMdh_msg_border = visp_bridge::toGeometryMsgsPose(cMdh_border);

    pRdh = cRp.inverse() * cRdh;
    pMdh = vpHomogeneousMatrix(Pdh, pRdh);

    pMdh_msg = visp_bridge::toGeometryMsgsPose(pMdh);

    tf::Quaternion qdh;
    qdh.setX(cMdh_msg_border.orientation.x);
    qdh.setY(cMdh_msg_border.orientation.y);
    qdh.setZ(cMdh_msg_border.orientation.z);
    qdh.setW(cMdh_msg_border.orientation.w);

    transformdh.setRotation(qdh);
    br.sendTransform(tf::StampedTransform(transformdh, ros::Time::now(), m_parent_tf, "door_handle_tf"));
    //  ROS_INFO_STREAM("cMdh dans le tfLine = \n" << cMdh_border);
    return cMdh_border;
}

vpColVector DoorHandleDetectionNode::getCoeffPlaneWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const double centroidx, const double centroidy, const double centroidz, vpColVector normal)
{
    // Minimization
    vpMatrix M;
    vpRowVector m(3);

    for(unsigned int i=0; i<cloud->size(); i++) {
        m[0] = cloud->points[i].x - centroidx;
        m[1] = cloud->points[i].y - centroidy;
        m[2] = cloud->points[i].z - centroidz;
        M.stack(m);
    }

    vpMatrix A = M.t() * M;

    vpColVector D;
    vpMatrix V;
    A.svd(D, V);

    ROS_INFO_STREAM("A:\n" << A << "\n");
    ROS_INFO_STREAM("D:" << D.t() << "\n");
    ROS_INFO_STREAM("V:\n" << V << "\n");

    double smallestSv = D[0];
    unsigned int indexSmallestSv = 0 ;
    for (unsigned int i = 1; i < D.size(); i++) {
        if ((D[i] < smallestSv) ) {
            smallestSv = D[i];
            indexSmallestSv = i;
        }
    }

    vpColVector h = V.getCol(indexSmallestSv);

    ROS_INFO_STREAM("Ground normal vector with the pcl method: " << normal.t() << "\n");

    ROS_INFO_STREAM("Estimated normal Vector: " << h.t() << "\n");

    return h;
}

inliersAndCoefficients DoorHandleDetectionNode::getPlaneInliersAndCoefficients(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

//    pcl::VoxelGrid<pcl::PointXYZ> vg;
//    vg.setInputCloud( cloud );
//    vg.setLeafSize( 0.01f, 0.01f, 0.01f );
//    vg.filter(*cloud_filtered);

    struct inliersAndCoefficients plane;
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);
    //Create the inliers and coefficients for the biggest plan found
    seg.setInputCloud (cloud/*_filtered*/);
    seg.segment (*inliers, *coefficients);

    //  double Z_min, Z_max;
    //  if(!m_plane_is_initialized)
    //  {
    m_plane_is_initialized = true;
    //  }
    //  ROS_INFO("z_min = %lf , z_max = %lf", m_Z, m_Z);

    plane.coefficients = coefficients;
    plane.inliers = inliers;

    return plane;
}


double DoorHandleDetectionNode::computeX(const vpColVector coeffs, const double y, const double z)
{
    return -(coeffs[1]*y + coeffs[2]*z + coeffs[3])/(coeffs[0]);
}

double DoorHandleDetectionNode::computeY(const vpColVector coeffs, const double x, const double z)
{
    return -(coeffs[0]*x + coeffs[2]*z + coeffs[3])/(coeffs[1]);
}

double DoorHandleDetectionNode::computeZ(const vpColVector coeffs, const double x, const double y)
{
    return -(coeffs[0]*x + coeffs[1]*y + coeffs[3])/(coeffs[2]);
}

vpHomogeneousMatrix DoorHandleDetectionNode::createTFPlane(const vpColVector coeffs, const double x, const double y, const double z)
{
    vpColVector xp(3);
    vpColVector yp(3);
    vpColVector normal(3);
    vpRotationMatrix cRp;
    vpTranslationVector P0;
    vpHomogeneousMatrix cMp;
    geometry_msgs::Pose cMp_msg;
    tf::Transform transform;
    static tf::TransformBroadcaster br;

    //Create a normal to the plan
    normal[0] = -coeffs[0];
    normal[1] = -coeffs[1];
    normal[2] = -coeffs[2];

    normal.normalize();

    //Create yp and xp
    xp[0] = DoorHandleDetectionNode::computeX(coeffs, y, z+0.05) - x;
    xp[1] = 0;
    xp[2] = 0.05;
    xp.normalize();
    yp = vpColVector::cross(normal,xp);

    //Create the Rotation Matrix
    cRp[0][0] = xp[0];
    cRp[1][0] = xp[1];
    cRp[2][0] = xp[2];
    cRp[0][1] = yp[0];
    cRp[1][1] = yp[1];
    cRp[2][1] = yp[2];
    cRp[0][2] = normal[0];
    cRp[1][2] = normal[1];
    cRp[2][2] = normal[2];

    transform.setOrigin( tf::Vector3(x, y, z) );

    //Calculate the z0
    double z0 = DoorHandleDetectionNode::computeZ(coeffs, x, y);

    //Create the translation Vector
    P0 = vpTranslationVector(x, y, z0);

    //Create the homogeneous Matrix
    cMp = vpHomogeneousMatrix(P0, cRp);
    cMp_msg = visp_bridge::toGeometryMsgsPose(cMp);
    tf::Quaternion q;
    q.setX(cMp_msg.orientation.x);
    q.setY(cMp_msg.orientation.y);
    q.setZ(cMp_msg.orientation.z);
    q.setW(cMp_msg.orientation.w);
    transform.setRotation(q);

    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), m_parent_tf, "tf_plane"));
    return cMp;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr DoorHandleDetectionNode::createPlaneFromInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, const pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr convexHull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.filter(*cloud_filtered);

    // Object for retrieving the convex hull.
    pcl::ConvexHull<pcl::PointXYZ> hull;
    hull.setInputCloud(plane);
    hull.reconstruct(*convexHull);

    // Create the point cloud to display all the plane in white
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType (pcl::SACMODEL_PLANE);
    proj.setInputCloud (cloud);
    proj.setModelCoefficients (coefficients);
    proj.filter (*plane);

    return plane;
}


vpColVector DoorHandleDetectionNode::getCenterPCL(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
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

pcl::PointCloud<pcl::PointXYZ>::Ptr DoorHandleDetectionNode::createPCLBetweenTwoPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, vpColVector coefficients)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outside(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_outside->width = 300000;
    cloud_outside->height = 1;
    cloud_outside->points.resize (cloud_outside->width * cloud_outside->height);

    double xc,yc,zc,z_min, z_max;
    size_t width_outside = 0;

//    pcl::PassThrough<pcl::PointXYZ> pass;
//    pass.setInputCloud (cloud);
//    pass.setFilterFieldName ("z");
//    pass.setFilterLimits (m_X_min, m_X_max);
//    //pass.setFilterLimitsNegative (true);
//    pass.filter (*cloud_outside);


    for(int i = 0; i < cloud->size(); i++)
    {
        xc = cloud->points[i].x;
        yc = cloud->points[i].y;
        zc = cloud->points[i].z;
        z_min = -(coefficients[0]*xc + coefficients[1]*yc + coefficients[3])/(coefficients[2]) - (m_height_dh + 0.03);
        z_max = z_min - 0.06;
        if (zc < z_min && zc > z_max )
        {
            width_outside++;
            cloud_outside->points[width_outside-1].x = xc;
            cloud_outside->points[width_outside-1].y = yc;
            cloud_outside->points[width_outside-1].z = z_min - 0.03;
        }
    }
    cloud_outside->width = width_outside;
    cloud_outside->points.resize (cloud_outside->width * cloud_outside->height);

    return cloud_outside;
}

void DoorHandleDetectionNode::setupCameraParameters(const sensor_msgs::CameraInfoConstPtr &cam_rgb)
{
    if (! m_cam_is_initialized && m_extrinsic_param_are_initialized) {
        //init m_camera parameters
        m_cam_rgb = visp_bridge::toVispCameraParameters(*cam_rgb);
        vpPixelMeterConversion::convertPoint(m_cam_rgb, 0, 0, m_x_min, m_y_min);
        vpPixelMeterConversion::convertPoint(m_cam_rgb, 640, 480, m_x_max, m_y_max);
        m_x_min -= (m_extrinsicParam[0] );
        m_x_max -= (m_extrinsicParam[0] );
        m_y_min -= (m_extrinsicParam[1] );
        m_y_max -= (m_extrinsicParam[1] );
        ROS_INFO_STREAM("Camera param  = \n" << m_cam_rgb);

        m_cam_is_initialized = true;
        cam_rgb_info_sub.shutdown();
    }
}

void DoorHandleDetectionNode::getExtrinsicParameters(const sensor_msgs::CameraInfoConstPtr &cam_depth)
{
    if (! m_extrinsic_param_are_initialized) {
        //init extrinsic parameters between Depth and RGB
        m_extrinsicParam[0] = cam_depth->P[3];
        m_extrinsicParam[1] = cam_depth->P[7];
        m_extrinsicParam[2] = cam_depth->P[11];
        ROS_INFO_STREAM("Extrinsic param  = \n" << m_extrinsicParam);

        cam_depth_info_sub.shutdown();
        m_extrinsic_param_are_initialized = true;
    }
}

void DoorHandleDetectionNode::initDisplayVisp()
{
    if (! m_disp_is_initialized) {
        //init graphical interface
        m_disp = new vpDisplayX(m_img_, 750, 0, "Image viewer");
        vpDisplay::flush(m_img_);
        vpDisplay::display(m_img_);
        ROS_INFO("Initialisation done");
        vpDisplay::flush(m_img_);

        m_disp_mono = new vpDisplayX();
        m_disp_mono->init(m_img_mono);
        m_disp_mono->setTitle("Image Mono viewer");
        vpDisplay::flush(m_img_mono);
        vpDisplay::display(m_img_mono);
        ROS_INFO("Initialisation done");
        vpDisplay::flush(m_img_mono);

        m_disp_is_initialized = true;
    }

}

void DoorHandleDetectionNode::displayImage(const sensor_msgs::Image::ConstPtr& image)
{
    initDisplayVisp();
    vpMouseButton::vpMouseButtonType button;
    vpImagePoint pointClicked;
    vpImagePoint bottomRightBBoxHandle;
    vpImagePoint topLeftBBoxHandle;
    geometry_msgs::PointStamped point_handle;
    double X, Y, Z, x, y, u, v;

    m_img_mono = visp_bridge::toVispImageRGBa(*image);

    vpDisplay::display(m_img_mono);
    //vpDisplay::displayText(m_img_mono, 15, 5, "Right click to select a region for the detection of the door handle,", vpColor::red);
    //vpDisplay::displayText(m_img_mono, 30, 5, "Middle click to delete the bounding box,", vpColor::red);
    //vpDisplay::displayText(m_img_mono, 45, 5, "Left click to quit.", vpColor::red);
    //  vpDisplay::flush(m_img_mono);
    if ( vpDisplay::getClick( m_img_mono, pointClicked, button, false) )
    {
        if (button == vpMouseButton::button1)
        {
            m_pointPoseHandle = pointClicked;
            m_bbox_is_fixed = true;
            if (m_dh_right)
            {
                bottomRightBBoxHandle.set_uv( pointClicked.get_u() + 50, pointClicked.get_v() + 50 );
                topLeftBBoxHandle.set_uv( pointClicked.get_u() - 50, pointClicked.get_v() - 50);
            }
            else
            {
                bottomRightBBoxHandle.set_uv( pointClicked.get_u() + 50, pointClicked.get_v() + 50 );
                topLeftBBoxHandle.set_uv( pointClicked.get_u() - 50, pointClicked.get_v() - 50);
            }
            m_bboxdetectionhandle.setTopLeft(topLeftBBoxHandle);
            m_bboxdetectionhandle.setBottomRight(bottomRightBBoxHandle);
            if ( !m_tracking_works)
            {
                vpPixelMeterConversion::convertPoint(m_cam_rgb, bottomRightBBoxHandle, m_x_max, m_y_max);
                vpPixelMeterConversion::convertPoint(m_cam_rgb, topLeftBBoxHandle, m_x_min, m_y_min);
            }

            m_stop_detection = false;
            vpDisplay::displayRectangle(m_img_mono, m_bboxdetectionhandle, vpColor::green);

            //      ROS_INFO_STREAM("Coordinates Top Left  : " << m_x_min << ", " << m_y_min << "\nCoordinates Bottom Right : " << m_x_max << ", " << m_y_max << "\n");

        }
        else if(button == vpMouseButton::button2)
        {
            m_bbox_is_fixed = false;
            m_plane_is_initialized = false;
            m_tracking_is_initialized = false;
            m_tracking_works = false;
            m_stop_detection = true;
        }
        else if(button == vpMouseButton::button3)
            ros::shutdown();

    }
    if ( m_is_door_handle_present )
    {
        //cMh is on the handle rotation axis, we put it on the cog
        vpHomogeneousMatrix hMhcog(m_lenght_dh / 2,0,0,0,0,0);
        vpHomogeneousMatrix cMhcog = m_cMh* hMhcog;
        vpTranslationVector c_t_hcog = cMhcog.getTranslationVector();
        double x_hcog = c_t_hcog[0]/c_t_hcog[2];
        double y_hcog = c_t_hcog[1]/c_t_hcog[2];

        vpMeterPixelConversion::convertPoint(m_cam_rgb, x_hcog, y_hcog, m_pointPoseHandle);
        if (m_dh_right)
        {
            bottomRightBBoxHandle.set_uv( m_pointPoseHandle.get_u() + 150, m_pointPoseHandle.get_v() + 80 );
            topLeftBBoxHandle.set_uv( m_pointPoseHandle.get_u() - 50, m_pointPoseHandle.get_v() - 80);
        }
        else
        {
            bottomRightBBoxHandle.set_uv( m_pointPoseHandle.get_u() + 50, m_pointPoseHandle.get_v() + 80 );
            topLeftBBoxHandle.set_uv( m_pointPoseHandle.get_u() - 150, m_pointPoseHandle.get_v() - 80);
        }
        m_bboxdetectionhandle.setBottomRight(bottomRightBBoxHandle);
        m_bboxdetectionhandle.setTopLeft(topLeftBBoxHandle);
    }
    if ( m_bbox_is_fixed && m_tracking_works)
    {
//        vpDisplay::displayRectangle(m_img_mono, m_bboxdetectionhandle, vpColor::green);
//        vpDisplay::displayRectangle(m_img_mono, m_bboxhandle, vpColor::yellow, 0, 2);
    }
    if ( m_bbox_is_fixed && m_is_door_handle_present)
    {
        X = m_cMh[0][3] ;
        Y = m_cMh[1][3] ;
        Z = m_cMh[2][3] ;
        x = X/Z;
        y = Y/Z;
        vpMeterPixelConversion::convertPoint(m_cam_rgb, x, y, u, v);
        point_handle.header.stamp = ros::Time::now();
        point_handle.point.x = u;
        point_handle.point.y = v;
        point_handle.point.z = 0.0;
        point_handle_pub.publish(point_handle);
//        vpDisplay::displayFrame(m_img_mono, m_cMh, m_cam_rgb, 0.1, vpColor::red, 1);
        vpDisplay::displayFrame(m_img_mono, m_cMh_filtered_kalman, m_cam_rgb, 0.1, vpColor::green, 1);
//        vpDisplay::displayFrame(m_img_mono, m_cMh_test, m_cam_rgb, 0.1, vpColor::red, 1);
//        vpDisplay::displayFrame(m_img_mono, m_cMh_filtered_mean, m_cam_rgb, 0.1, vpColor::yellow, 1);
    }
    vpDisplay::flush(m_img_mono);

}

void DoorHandleDetectionNode::morphoSandwich(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    m_img_ = 0;
    double X,Y,Z,x,y,u,v;
    vpImagePoint bottomRightBBoxHandle;
    vpImagePoint topLeftBBoxHandle;
    vpMouseButton::vpMouseButtonType button;
    vpRect searchingField;


    for(int i = 0; i < cloud->size(); i++ ){
        X = cloud->points[i].x + m_extrinsicParam[0] ;
        Y = cloud->points[i].y + m_extrinsicParam[1] ;
        Z = cloud->points[i].z + m_extrinsicParam[2] ;
        x = X/Z;
        y = Y/Z;

        vpMeterPixelConversion::convertPoint(m_cam_rgb, x, y, u, v);

        if(u < m_img_.getWidth()-1 && v < m_img_.getHeight()-1 && u > 0 && v > 0 )
        {
//            ROS_INFO("DEBUG la : u: %lf et v: %lf", u, v );
            m_img_.bitmap[ (int)v * m_img_.getWidth() + (int)u ] = 250;
        }
//        ROS_INFO("DEBUFG");
    }
//    ROS_INFO("DEBUG Morpho");
    vpImageMorphology::dilatation(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
    vpImageMorphology::dilatation(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
    vpImageMorphology::erosion(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
    vpImageMorphology::erosion(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
    vpImageMorphology::erosion(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
    vpImageMorphology::dilatation(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
    //  vpImageMorphology::dilatation(m_img_, (unsigned char) 255, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);


    if (0){
      static int iter = 0;
      char filename[255];
      sprintf(filename, "/tmp/bheintz/I%04d.png", iter ++);
      vpImageIo::write(m_img_, filename);
    }
    vpDisplay::display(m_img_);
    if ( !m_stop_detection )
    {
        try {
            if ( m_is_door_handle_present )
            {
                if ( !m_tracking_is_initialized )
                {
                    if ( m_pointPoseHandle.get_u() > 0 && m_pointPoseHandle.get_v() > 0 && m_pointPoseHandle.get_u() < m_img_.getWidth()-1 && m_pointPoseHandle.get_v() < m_img_.getHeight()-1 )
                    {
                        m_blob.initTracking(m_img_, m_pointPoseHandle, 150, 255);
                        ROS_INFO_STREAM( "Blob from init :" << m_blob.getCog() << " Gray min : " << m_blob.getGrayLevelMin() << " Gray max : " << m_blob.getGrayLevelMax() << " Precision : " << m_blob.getGrayLevelPrecision());
                        m_tracking_is_initialized = true;
                    }
                }
                else
                {
                    m_blob.track(m_img_);
                    m_bboxhandle = m_blob.getBBox();
                    m_tracking_works = true;
                    m_bbox_is_fixed = true;

                    //        ROS_INFO_STREAM( "Blob tracking " << m_blob.getCog() << " Gray min : " << m_blob.getGrayLevelMin() << " Gray max : " << m_blob.getGrayLevelMax() << " Precision : " << m_blob.getGrayLevelPrecision());
                }
            }
        }
        catch(...) {
            m_tracking_is_initialized = false;
            m_tracking_works = false;
            m_bbox_is_fixed = false;
            ROS_INFO_STREAM("Tracking failed");

        }
    }

    if ( m_tracking_works )
    {
        if (m_bboxhandle.getSize() < 2000)
        {
            topLeftBBoxHandle.set_uv( m_bboxdetectionhandle.getLeft(), m_bboxdetectionhandle.getTop() );
            bottomRightBBoxHandle.set_uv( m_bboxdetectionhandle.getRight(), m_bboxdetectionhandle.getBottom() );

        }
        else
        {
            topLeftBBoxHandle.set_uv( m_bboxhandle.getLeft() - m_bboxhandle.getWidth()/4, m_bboxhandle.getTop() - m_bboxhandle.getHeight()/4);
            bottomRightBBoxHandle.set_uv( m_bboxhandle.getRight() + m_bboxhandle.getWidth()/4, m_bboxhandle.getBottom() + m_bboxhandle.getHeight()/4 );
        }

        vpPixelMeterConversion::convertPoint(m_cam_rgb, topLeftBBoxHandle, m_x_min, m_y_min);
        vpPixelMeterConversion::convertPoint(m_cam_rgb, bottomRightBBoxHandle, m_x_max, m_y_max);

        searchingField.setTopLeft(topLeftBBoxHandle);
        searchingField.setBottomRight(bottomRightBBoxHandle);
        vpDisplay::displayRectangle(m_img_, m_bboxhandle, vpColor::yellow, 0, 2);
    }
    else
    {
        vpPixelMeterConversion::convertPoint(m_cam_rgb, 0, 0, m_x_min, m_y_min);
        vpPixelMeterConversion::convertPoint(m_cam_rgb, m_img_.getWidth() - 1, m_img_.getHeight() - 1, m_x_max, m_y_max);

    }


    //  vpDisplay::displayRectangle(m_img_, searchingField, vpColor::purple);
//    vpDisplay::displayRectangle(m_img_, m_bboxdetectionhandle, vpColor::green);
    vpDisplay::displayText(m_img_, 15, 5, "Left click to quit.", vpColor::red);
//    vpDisplay::displayFrame(m_img_, m_cMh, m_cam_rgb, 0.1);
    vpDisplay::flush(m_img_);
    if(vpDisplay::getClick(m_img_, button, false)) {
        //    if (button == vpMouseButton::button1)
        //      m_bbox_is_fixed = true;
        //    else if(button == vpMouseButton::button2)
        //      m_bbox_is_fixed = false;
        if(button == vpMouseButton::button3)
            ros::shutdown();
    }

}

pcl::PointCloud<pcl::PointXYZ>::Ptr DoorHandleDetectionNode::getOnlyUsefulHandle(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_useful(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    m_X_min = m_x_min * m_Z;
    m_X_max = m_x_max * m_Z;
    m_Y_min = m_y_min * m_Z;
    m_Y_max = m_y_max * m_Z;
//    ROS_INFO("XMIN = %lf, XMAX = %lf,    YMIN = %lf, YMAX = %lf,    Z = %lf",m_X_min, m_X_max, m_Y_min, m_Y_max, m_Z);
    m_X_min -= m_extrinsicParam[0];
    m_X_max -= m_extrinsicParam[0];
    m_Y_min -= m_extrinsicParam[1];
    m_Y_max -= m_extrinsicParam[1];

    if (m_X_min > m_X_max)
    {
        double tmp = m_X_min;
        m_X_min = m_X_max;
        m_X_max = tmp;
    }
    if (m_Y_min > m_Y_max)
    {
        double tmp = m_Y_min;
        m_Y_min = m_Y_max;
        m_Y_max = tmp;
    }

//    ROS_INFO("XMIN = %lf, XMAX = %lf,    YMIN = %lf, YMAX = %lf,    Z = %lf",m_X_min, m_X_max, m_Y_min, m_Y_max, m_Z);
    pcl::PassThrough<pcl::PointXYZ> pass1;
    pass1.setInputCloud (cloud);
    pass1.setFilterFieldName ("x");
    pass1.setFilterLimits (m_X_min, m_X_max);
    //pass.setFilterLimitsNegative (true);
    pass1.filter (*cloud_filtered);

    pcl::PassThrough<pcl::PointXYZ> pass2;
    pass2.setInputCloud (cloud_filtered);
    pass2.setFilterFieldName ("y");
    pass2.setFilterLimits (m_Y_min, m_Y_max);
    //pass.setFilterLimitsNegative (true);
    pass2.filter (*cloud_useful);
    //  ROS_INFO("size of cloud useful = %ld", width_useful);
    return cloud_useful;
}

vpColVector DoorHandleDetectionNode::getCoeffLineWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    vpMatrix M(cloud->size(),3);
    vpRowVector m(3);
    vpColVector centroid = DoorHandleDetectionNode::getCenterPCL(cloud);

    for(unsigned int i=0; i<cloud->size(); i++) {
        m[0] = cloud->points[i].x - centroid[0];
        m[1] = cloud->points[i].y - centroid[1];
        m[2] = cloud->points[i].z - centroid[2];
        //M.stack(m);
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

