#include "door_handle_detection.h"


DoorHandleDetectionNode::DoorHandleDetectionNode(ros::NodeHandle nh)
{
  n = nh;

  m_img_.init(480,640);
  m_img_2.init(480,640);
  m_img_mono.init(480,640);
  ROS_INFO("Launch Test ros node");

  m_Z_min = 0.5;
  m_Z_max = 0.5;
  m_extrinsic_param_are_initialized = false;
  m_useful_cloud_is_initialized = false;
  m_is_previous_initialized = false;
  m_tracking_is_initialized = false;
  m_is_door_handle_present = false;
  m_plane_is_initialized =false;
  m_disp_is_initialized = false;
  m_cam_is_initialized = false;
  m_tracking_works = false;
  m_bbox_is_fixed = false;


  n.param<std::string>("m_image_topic_name", m_imageTopicName, "/camera/rgb/image_raw");
  n.param<std::string>("m_pclTopicName", m_pclTopicName, "/softkinetic_camera/depth/points");
  n.param<std::string>("m_cameraRGBTopicName", m_cameraRGBTopicName, "/softkinetic_camera/rgb/camera_info");
  n.param<std::string>("m_cameraDepthTopicName", m_cameraDepthTopicName, "/softkinetic_camera/depth/camera_info");
//  n.param<std::string>("m_parent_tf", m_parent_tf, "softkinetic_camera_link");
  if (n.param("m_soft", m_soft))
    m_parent_tf = "softkinetic_camera_link";
  else
    m_parent_tf = "camera_depth_optical_frame";
//  n.getParam("m_parent_tf", m_parent_tf);
  n.param("m_dh_right", m_dh_right, true);
  n.param("debug", debug, false);
  n.param("m_lenght_dh",m_lenght_dh, 0.1);
  n.param("m_height_dh", m_height_dh, 0.055);
//  n.param("m_plane_is_initialized", m_plane_is_initialized, false);
//  n.param("m_Z_max", m_Z_center, 0.8);
  // subscribe to services
  ROS_INFO("Beautiful weather, isn't it ?");
  homogeneous_matrix_pub = n.advertise< geometry_msgs::Pose >("dh_tf", 1);
  if (debug)
  {
    pcl_plane_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PCL_Plane", 1);
    pcl_dh_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PCL_Outside", 1);
  }
  door_handle_final_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PCL_DH", 1);
  cam_rgb_info_sub = n.subscribe( m_cameraRGBTopicName, 1, (boost::function < void(const sensor_msgs::CameraInfo::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::setupCameraParameters, this, _1 ));
  cam_depth_info_sub = n.subscribe( m_cameraDepthTopicName, 1, (boost::function < void(const sensor_msgs::CameraInfo::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::getExtrinsicParameters, this, _1 ));
  pcl_frame_sub = n.subscribe( m_pclTopicName, 1, (boost::function < void(const sensor_msgs::PointCloud2::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::mainComputation, this, _1 ));
  image_frame_sub = n.subscribe( m_imageTopicName, 1, (boost::function < void(const sensor_msgs::Image::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::displayImage, this, _1 ));
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
  vpColVector direction_line(3);
  vpColVector normal(3);
  vpRotationMatrix cRp;
  vpHomogeneousMatrix cMp;
  vpHomogeneousMatrix cMdh;
  geometry_msgs::Pose cMdh_msg;
  vpColVector coeffs;
  struct inliersAndCoefficients plane_variables;
  pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
//  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);

  // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
  pcl::fromROSMsg (*image, *cloud);

  if(m_cam_is_initialized)
  {
    cloud_bbox = DoorHandleDetectionNode::getOnlyUsefulHandle(cloud);

    plane_variables = DoorHandleDetectionNode::getPlaneInliersAndCoefficients(cloud);
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

      //Creating a cloud with all the points of the door handle
      cloud_sandwich = DoorHandleDetectionNode::createPCLBetweenTwoPlanes(cloud_bbox, coeffs, m_height_dh);

      if (debug)
      {
      //Publish the pcl of the door handle with noise
        cloud_sandwich->header.frame_id = m_parent_tf;
        pcl_dh_pub.publish(*cloud_sandwich);
      }

      if (cloud_sandwich->size()<30){
        m_is_door_handle_present = false;
        vpDisplay::displayText(m_img_mono, 60, 5, "No door Handle detected", vpColor::red);
        vpDisplay::flush(m_img_mono);
        ROS_INFO("No door handle detected : %d", m_is_door_handle_present);
      }
      else
      {
        m_is_door_handle_present = true;
        std::vector<int> inliers2;
        Eigen::VectorXf Coeff_line;

        //Create a RandomSampleConsensus object and compute the model of a line
        pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr  model_l (new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cloud_sandwich));
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_l);
        ransac.setDistanceThreshold (.001);
        ransac.computeModel();
        ransac.getInliers(inliers2);
        ransac.getModelCoefficients(Coeff_line);
        // copies all inliers of the model computed to another PointCloud
        pcl::copyPointCloud<pcl::PointXYZ>(*cloud_sandwich, inliers2, *cloud_handle);
        if (m_dh_right)
        {
          if (Coeff_line[3]>0)
          {
            direction_line[0] = Coeff_line[3];
            direction_line[1] = Coeff_line[4];
            direction_line[2] = Coeff_line[5];
          }
          else
          {
            direction_line[0] = -Coeff_line[3];
            direction_line[1] = -Coeff_line[4];
            direction_line[2] = -Coeff_line[5];
          }
        }
        else
        {
          if (Coeff_line[3]>0)
          {
            direction_line[0] = -Coeff_line[3];
            direction_line[1] = -Coeff_line[4];
            direction_line[2] = -Coeff_line[5];
          }
          else
          {
            direction_line[0] = Coeff_line[3];
            direction_line[1] = Coeff_line[4];
            direction_line[2] = Coeff_line[5];
          }

        }
        //Publish the pcl of the door handle
        cloud_handle->header.frame_id = m_parent_tf;
        door_handle_final_pub.publish(*cloud_handle);

        DoorHandleDetectionNode::getImageVisp(cloud_sandwich);

        vpColVector centroidDH = DoorHandleDetectionNode::getCenterPCL(cloud_handle);

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

        //      vpColVector directionLineCoeff(3);
        //      directionLineCoeff = DoorHandleDetectionNode::getCoeffLineWithODR(cloud_handle);

        //Create the door handle tf
        m_dMh = DoorHandleDetectionNode::createTFLine(direction_line, normal, centroidDH[0], centroidDH[1], centroidDH[2], cRp, cMp);
        m_cMh = m_dMh;
        m_cMh[0][3] += m_extrinsicParam[0];
        m_cMh[1][3] += m_extrinsicParam[1];
        m_cMh[2][3] += m_extrinsicParam[2];
        //      ROS_INFO_STREAM("dMh = \n" << cMdh);
        cMdh_msg = visp_bridge::toGeometryMsgsPose(cMdh);
        homogeneous_matrix_pub.publish(cMdh_msg);
        //      vpHomogeneousMatrix test = visp_bridge::toVispHomogeneousMatrix(cMdh_msg);
        //      ROS_INFO_STREAM("dMh_2nd = \n" << test);
      }

    }

    if (debug)
    {
      //Publish the point clouds of the plan
      plane->header.frame_id = m_parent_tf;
      pcl_plane_pub.publish(*plane);
    }
  }

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
  struct inliersAndCoefficients plane;
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.05);
  //Create the inliers and coefficients for the biggest plan found
  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);

//  double Z_min, Z_max;
//  if(!m_plane_is_initialized)
//  {
    m_Z_min = - (coefficients->values[3]) / (coefficients->values[2] + coefficients->values[0] * m_x_min + coefficients->values[1] * m_y_min) + m_height_dh;
    m_Z_max = - (coefficients->values[3]) / (coefficients->values[2] + coefficients->values[0] * m_x_max + coefficients->values[1] * m_y_max) + m_height_dh;
    m_Z_min -= m_extrinsicParam[2];
    m_Z_max -= m_extrinsicParam[2];
    m_plane_is_initialized = true;
//  }
//  ROS_INFO("z_min = %lf , z_max = %lf", m_Z_min, m_Z_max);

  plane.coefficients = coefficients;
  plane.inliers = inliers;

  return plane;
}

pcl::PointIndices::Ptr DoorHandleDetectionNode::getPlaneInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.05);
  //Create the inliers and coefficients for the biggest plan found
  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);

  return inliers;
}

double DoorHandleDetectionNode::computeX(const vpColVector coeffs, const double y, const double z)
{
  double x = -(coeffs[1]*y + coeffs[2]*z + coeffs[3])/(coeffs[0]);
  return x;
}

double DoorHandleDetectionNode::computeY(const vpColVector coeffs, const double x, const double z)
{
  double y = -(coeffs[0]*x + coeffs[2]*z + coeffs[3])/(coeffs[1]);
  return y;
}

double DoorHandleDetectionNode::computeZ(const vpColVector coeffs, const double x, const double y)
{
  double z = -(coeffs[0]*x + coeffs[1]*y + coeffs[3])/(coeffs[2]);
  return z;
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

pcl::PointCloud<pcl::PointXYZ>::Ptr DoorHandleDetectionNode::createPCLBetweenTwoPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, vpColVector coefficients, double height_dh)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outside(new pcl::PointCloud<pcl::PointXYZ>);
  cloud_outside->width = 1;
  cloud_outside->height = 1;
  cloud_outside->points.resize (cloud_outside->width * cloud_outside->height);

  double xc,yc,zc,z_min, z_max;
  size_t width_outside = 0;

  for(int i = 0; i < cloud->size(); i++)
  {
    xc = cloud->points[i].x;
    yc = cloud->points[i].y;
    zc = cloud->points[i].z;
    z_min = DoorHandleDetectionNode::computeZ(coefficients, xc, yc) - (height_dh - 0.025);
    z_max = DoorHandleDetectionNode::computeZ(coefficients, xc, yc) - (height_dh + 0.025);
    if (zc < z_min && zc > z_max )
    {
      width_outside++;
      cloud_outside->width = width_outside;
      cloud_outside->points.resize (cloud_outside->width * cloud_outside->height);
      cloud_outside->points[width_outside-1].x = xc;
      cloud_outside->points[width_outside-1].y = yc;
      cloud_outside->points[width_outside-1].z = DoorHandleDetectionNode::computeZ(coefficients, xc, yc) - height_dh;
    }
  }

  return cloud_outside;
}

void DoorHandleDetectionNode::setupCameraParameters(const sensor_msgs::CameraInfoConstPtr &cam_rgb)
{
  if (! m_cam_is_initialized) {
    //init m_camera parameters
    m_cam_rgb = visp_bridge::toVispCameraParameters(*cam_rgb);
//    vpXmlParserCamera p; // Create a XML parser
//    vpCameraParameters::vpCameraParametersProjType projModel; // Projection model
//    projModel = vpCameraParameters::perspectiveProjWithDistortion;
//    if (p.parse(m_cam, "camera.xml", "Camera", projModel,640,480) != vpXmlParserCamera::SEQUENCE_OK) {
//      std::cout << "Cannot found Camera" << std::endl;
//    }
    vpPixelMeterConversion::convertPoint(m_cam_rgb, 0, 0, m_x_min, m_y_min);
    vpPixelMeterConversion::convertPoint(m_cam_rgb, 640, 480, m_x_max, m_y_max);
    m_x_min -= (m_extrinsicParam[0] );
    m_x_max -= (m_extrinsicParam[0] );
    m_y_min -= (m_extrinsicParam[1] );
    m_y_max -= (m_extrinsicParam[1] );
    ROS_INFO_STREAM("Camera param  = \n" << m_cam_rgb);

    m_cam_is_initialized = true;
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

    m_extrinsic_param_are_initialized = true;
  }
}

void DoorHandleDetectionNode::initDisplayVisp()
{
  if (! m_disp_is_initialized) {
    //init graphical interface
    m_disp = new vpDisplayX(m_img_, 750, 0, "Image viewer");
//    m_disp->init(m_img_);
//    m_disp->setTitle("Image viewer");
    vpDisplay::flush(m_img_);
    vpDisplay::display(m_img_);
    ROS_INFO("Initialisation done");
    vpDisplay::flush(m_img_);

//    m_disp2 = new vpDisplayX(m_img_2, 1500, 0, "Image viewer final");
//    vpDisplay::flush(m_img_2);
//    vpDisplay::display(m_img_2);
//    ROS_INFO("Initialisation done");
//    vpDisplay::flush(m_img_2);

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
  double x_center, y_center, X_center, Y_center, Z_center;
  m_img_mono = visp_bridge::toVispImage(*image);

  vpDisplay::display(m_img_mono);
  vpDisplay::displayText(m_img_mono, 15, 5, "Right click to select a region for the detection of the door handle,", vpColor::red);
  vpDisplay::displayText(m_img_mono, 30, 5, "Middle click to delete the bounding box,", vpColor::red);
  vpDisplay::displayText(m_img_mono, 45, 5, "Left click to quit.", vpColor::red);
//  vpDisplay::flush(m_img_mono);
  if ( vpDisplay::getClick( m_img_mono, pointClicked, button, false) )
  {
    if (button == vpMouseButton::button1)
    {
      m_pointPoseHandle = pointClicked;
      m_bbox_is_fixed = true;
      if (m_dh_right)
      {
        bottomRightBBoxHandle.set_uv( pointClicked.get_u() + 150, pointClicked.get_v() + 100 );
        topLeftBBoxHandle.set_uv( pointClicked.get_u() - 50, pointClicked.get_v() - 100);
      }
      else
      {
        bottomRightBBoxHandle.set_uv( pointClicked.get_u() + 50, pointClicked.get_v() + 100 );
        topLeftBBoxHandle.set_uv( pointClicked.get_u() - 150, pointClicked.get_v() - 100);
      }
//      vpDisplay::displayPoint(m_img_mono, bottomRight, vpColor::red);
//      vpDisplay::displayPoint(m_img_mono, topLeft, vpColor::red);
      m_bboxdetectionhandle.setTopLeft(topLeftBBoxHandle);
      m_bboxdetectionhandle.setBottomRight(bottomRightBBoxHandle);
      if ( !m_tracking_works)
      {
        vpPixelMeterConversion::convertPoint(m_cam_rgb, bottomRightBBoxHandle, m_x_max, m_y_max);
        vpPixelMeterConversion::convertPoint(m_cam_rgb, topLeftBBoxHandle, m_x_min, m_y_min);
        //      vpPixelMeterConversion::convertPoint(m_cam_rgb, pointClicked, x_center, y_center);
        m_x_min -= (m_extrinsicParam[0] );
        m_x_max -= (m_extrinsicParam[0] );
        m_y_min -= (m_extrinsicParam[1] );
        m_y_max -= (m_extrinsicParam[1] );
      }
//      m_X_center = (x_center - m_extrinsicParam[0]) * m_Z_center;
//      m_Y_center = (y_center - m_extrinsicParam[1]) * m_Z_center;
      ROS_INFO_STREAM("Coordinates Top Left  : " << m_x_min << ", " << m_y_min << "\nCoordinates Bottom Right : " << m_x_max << ", " << m_y_max << "\n");

    }
    else if(button == vpMouseButton::button2)
    {
      m_bbox_is_fixed = false;
      m_plane_is_initialized = false;
      m_tracking_is_initialized = false;
      m_tracking_works = false;
    }
    else if(button == vpMouseButton::button3)
      ros::shutdown();

  }
  if ( m_is_door_handle_present )
  {
    X_center = m_cMh[0][3];
    Y_center = m_cMh[1][3];
    Z_center = m_cMh[2][3];
    x_center = X_center / Z_center;
    y_center = Y_center / Z_center;
    vpMeterPixelConversion::convertPoint(m_cam_rgb, x_center, y_center, m_pointPoseHandle);
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
  if ( m_bbox_is_fixed )
  {
    vpDisplay::displayRectangle(m_img_mono, m_bboxdetectionhandle, vpColor::green);
    vpDisplay::displayRectangle(m_img_mono, m_bboxhandle, vpColor::yellow);
    vpDisplay::displayFrame(m_img_mono, m_cMh, m_cam_rgb, 0.1);
  }
  vpDisplay::flush(m_img_mono);

}

void DoorHandleDetectionNode::getImageVisp(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  m_img_ = 0;
//  m_img_2 = 0;
  double X,Y,Z,x,y,u,v;
  int taille_avant, taille_apres = 0;
  vpImagePoint bottomRightBBoxHandle;
  vpImagePoint topLeftBBoxHandle;
  vpMouseButton::vpMouseButtonType button;
  vpImagePoint pointTest;
  vpPolygon polygonTest;
//  std::vector<vpImagePoint> allPointsHandle;
//  m_blob.setGraphics(true);
//  m_blob.setGraphicsThickness(2);
  m_blob_2.setGraphics(true);
  m_blob_2.setGraphicsThickness(1);
  m_blob_2.setEllipsoidShapePrecision(0.);
  m_blob_2.setGrayLevelMin(170);
  m_blob_2.setGrayLevelMax(255);

  taille_avant = cloud->size();
  for(int i = 0; i < cloud->size(); i++ ){
    X = cloud->points[i].x + m_extrinsicParam[0] ;
    Y = cloud->points[i].y + m_extrinsicParam[1] ;
    Z = cloud->points[i].z + m_extrinsicParam[2] ;
    x = X/Z;
    y = Y/Z;

    vpMeterPixelConversion::convertPoint(m_cam_rgb, x, y, u, v);

    if(u < m_img_.getWidth() && v < m_img_.getHeight())
    {
      m_img_.bitmap[ int(v) * m_img_.getWidth() + int(u) ] = 250;
//      m_img_2.bitmap[ int(v) * m_img_.getWidth() + int(u) ] = 250;
    }
  }
  vpImageMorphology::dilatation(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
  vpImageMorphology::dilatation(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
  vpImageMorphology::erosion(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
  vpImageMorphology::erosion(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
  vpImageMorphology::erosion(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);
  vpImageMorphology::dilatation(m_img_, (unsigned char) 250, (unsigned char) 0, vpImageMorphology::CONNEXITY_8);
//  vpImageMorphology::dilatation(m_img_, (unsigned char) 255, (unsigned char) 0, vpImageMorphology::CONNEXITY_4);

  vpDisplay::display(m_img_);
  try {
    if ( m_is_door_handle_present )
    {
      if ( !m_tracking_is_initialized )
      {
//        m_blob.initTracking(m_img_, m_pointPoseHandle);
        m_blob_2.initTracking(m_img_, m_pointPoseHandle);
        ROS_INFO_STREAM( "Blob from init :" << m_blob_2.getCog() << " Gray min : " << m_blob_2.getGrayLevelMin() << " Gray max : " << m_blob_2.getGrayLevelMax() << " Precision : " << m_blob_2.getGrayLevelPrecision());
        m_tracking_is_initialized = true;
      }
      else
      {
//        m_blob.track(m_img_);
        m_blob_2.track(m_img_);
        m_bboxhandle = m_blob_2.getBBox();
        m_tracking_works = true;
        m_bbox_is_fixed = true;
        ROS_INFO_STREAM( "Blob tracking " << m_blob_2.getCog() << " Gray min : " << m_blob_2.getGrayLevelMin() << " Gray max : " << m_blob_2.getGrayLevelMax() << " Precision : " << m_blob_2.getGrayLevelPrecision());
      }
    }
  }
  catch(...) {
    m_tracking_is_initialized = false;
    m_tracking_works = false;
  }


  if ( m_tracking_works )
  {
    bottomRightBBoxHandle.set_uv( m_bboxhandle.getBottomRight().get_u() + 10, m_bboxhandle.getBottomRight().get_v() + 10 );
    topLeftBBoxHandle.set_uv( m_bboxhandle.getTopLeft().get_u() - 20, m_bboxhandle.getTopLeft().get_v() - 20);

//    polygonTest = m_blob.getPolygon();
    vpPixelMeterConversion::convertPoint(m_cam_rgb, bottomRightBBoxHandle, m_x_max, m_y_max);
    vpPixelMeterConversion::convertPoint(m_cam_rgb, topLeftBBoxHandle, m_x_min, m_y_min);
//      vpPixelMeterConversion::convertPoint(m_cam_rgb, pointClicked, x_center, y_center);
    m_x_min -= (m_extrinsicParam[0] );
    m_x_max -= (m_extrinsicParam[0] );
    m_y_min -= (m_extrinsicParam[1] );
    m_y_max -= (m_extrinsicParam[1] );

//    for(int v = 0; v < m_img_.getHeight() ; v++)
//    {
//      for(int u = 0; u < m_img_.getWidth(); u++ )
//      {
//        pointTest.set_uv(u,v);
////        if (polygonTest.isInside(pointTest))
////          m_img_2.bitmap[ v * m_img_.getWidth() + u ] = 128;
//        if ( m_img_.bitmap[ v * m_img_.getWidth() + u ] > 170 && !pointTest.inRectangle(m_bboxhandle) )
//        {
////          ROS_INFO("A point to delete %lf, %lf !", pointTest.get_u(), pointTest.get_v());
//          m_img_2.bitmap[ v * m_img_.getWidth() + u ] = 0;

//          //        vpPixelMeterConversion::convertPoint(m_cam, v, u, x, y);
//        }
//      }
//    }
  }
//  vpDisplay::display(m_img_2);
//  vpDisplay::displayRectangle(m_img_2, m_bboxhandle, vpColor::yellow);
//  vpDisplay::displayText(m_img_2, 15, 5, "Left click to quit.", vpColor::red);
//  vpDisplay::flush(m_img_2);
  if(vpDisplay::getClick(m_img_2, button, false)) {
    if(button == vpMouseButton::button3)
      ros::shutdown();
  }

//  if (!m_bbox_is_fixed && allPointsHandle.size() > 10){
//    m_bboxhandle = vpImagePoint::getBBox(allPointsHandle);
//  }
  vpDisplay::displayRectangle(m_img_, m_bboxdetectionhandle, vpColor::green);
  vpDisplay::displayRectangle(m_img_, m_bboxhandle, vpColor::yellow);
  vpDisplay::displayText(m_img_, 15, 5, "Left click to quit.", vpColor::red);
  vpDisplay::displayFrame(m_img_, m_cMh, m_cam_rgb, 0.1);
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

pcl::PointCloud<pcl::PointXYZ>::Ptr DoorHandleDetectionNode::getOnlyUsefulPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_useful(new pcl::PointCloud<pcl::PointXYZ>);
  cloud_useful->width = 1;
  cloud_useful->height = 1;
  cloud_useful->points.resize (cloud_useful->width * cloud_useful->height);
  static double x_min, x_max, y_min, y_max;
  double xc,yc,zc;

  vpPixelMeterConversion::convertPoint(m_cam_rgb, 480, 640, x_max, y_max);
  vpPixelMeterConversion::convertPoint(m_cam_rgb, 0, 0, x_min, y_min);

  size_t width_useful = 0;

  m_X_min = x_min * m_Z_min;
  m_X_max = x_max * m_Z_max;
  m_Y_min = y_min * m_Z_min;
  m_Y_max = y_max * m_Z_max;
//  m_X_min -= m_extrinsicParam[0];
//  m_X_max -= m_extrinsicParam[0];
//  m_Y_min -= m_extrinsicParam[1];
//  m_Y_max -= m_extrinsicParam[1];

  for(int i = 0; i < cloud->size(); i++)
  {
    xc = cloud->points[i].x;
    yc = cloud->points[i].y;
    zc = cloud->points[i].z;
    if (xc > m_X_min && xc < m_X_max && yc > m_Y_min && yc < m_Y_max )
    {
      width_useful++;
      cloud_useful->width = width_useful;
      cloud_useful->points.resize (cloud_useful->width * cloud_useful->height);
      cloud_useful->points[width_useful-1].x = xc;
      cloud_useful->points[width_useful-1].y = yc;
      cloud_useful->points[width_useful-1].z = zc;
    }
  }

  return cloud_useful;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr DoorHandleDetectionNode::getOnlyUsefulHandle(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_useful(new pcl::PointCloud<pcl::PointXYZ>);
  cloud_useful->width = 1;
  cloud_useful->height = 1;
  cloud_useful->points.resize (cloud_useful->width * cloud_useful->height);

  double xc,yc,zc;

  size_t width_useful = 0;

  m_X_min = m_x_min * m_Z_min;
  m_X_max = m_x_max * m_Z_max;
  m_Y_min = m_y_min * m_Z_min;
  m_Y_max = m_y_max * m_Z_max;
//  m_X_min -= m_extrinsicParam[0];
//  m_X_max -= m_extrinsicParam[0];
//  m_Y_min -= m_extrinsicParam[1];
//  m_Y_max -= m_extrinsicParam[1];

  for(int i = 0; i < cloud->size(); i++)
  {
    xc = cloud->points[i].x;
    yc = cloud->points[i].y;
    zc = cloud->points[i].z;
    if (xc > m_X_min && xc < m_X_max && yc > m_Y_min && yc < m_Y_max )
    {
      width_useful++;
      cloud_useful->width = width_useful;
      cloud_useful->points.resize (cloud_useful->width * cloud_useful->height);
      cloud_useful->points[width_useful-1].x = xc;
      cloud_useful->points[width_useful-1].y = yc;
      cloud_useful->points[width_useful-1].z = zc;
    }
  }
//  ROS_INFO("zmin = %lf,    zmax = %lf", m_Z_min, m_Z_max);
//  ROS_INFO("size of cloud useful = %ld", width_useful);
  return cloud_useful;
}

vpColVector DoorHandleDetectionNode::getCoeffLineWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  vpMatrix M;
  vpRowVector m(3);
  vpColVector centroid = DoorHandleDetectionNode::getCenterPCL(cloud);

  for(unsigned int i=0; i<cloud->size(); i++) {
    m[0] = cloud->points[i].x - centroid[0];
    m[1] = cloud->points[i].y - centroid[1];
    m[2] = cloud->points[i].z - centroid[2];
    M.stack(m);
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

