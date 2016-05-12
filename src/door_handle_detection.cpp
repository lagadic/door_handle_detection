#include "door_handle_detection.h"

bool m_is_previous_initialized = false;

DoorHandleDetectionNode::DoorHandleDetectionNode(ros::NodeHandle nh)
{
  n = nh;

  ROS_INFO("Launch Test ros node");

  n.param<std::string>("PCL_topic_name", pclTopicName, "/softkinetic_camera/depth/points");

  // subscribe to services
  ROS_INFO("Beautiful weather, isn't it ?");
  pcl_plane_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PCL_Plane", 1);
  pcl_dh_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PCL_Outside", 1);
  door_handle_final_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("Final", 1);
  pcl_frame_sub = n.subscribe( pclTopicName, 1000, (boost::function < void(const sensor_msgs::PointCloud2::ConstPtr&)>) boost::bind( &DoorHandleDetectionNode::mainComputation, this, _1 ));
}

DoorHandleDetectionNode::~DoorHandleDetectionNode()
{

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
  pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_dh(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);
  vpColVector direction_line(3);
  vpColVector normal(3);
  vpRotationMatrix cRp;
  vpHomogeneousMatrix cMp;

  // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
  pcl::fromROSMsg (*image, *cloud);

  pcl::ModelCoefficients::Ptr coefficients = DoorHandleDetectionNode::getPlaneCoefficients(cloud);
  vpColVector coeffs;
  coeffs.stack(coefficients->values[0]);
  coeffs.stack(coefficients->values[1]);
  coeffs.stack(coefficients->values[2]);
  coeffs.stack(coefficients->values[3]);

  normal[0] = -coeffs[0];
  normal[1] = -coeffs[1];
  normal[2] = -coeffs[2];

  pcl::PointIndices::Ptr inliers = DoorHandleDetectionNode::getPlaneInliers(cloud);
  if (inliers->indices.size() == 0)
    std::cout << "Could not find a plane in the scene." << std::endl;
  else
  {
    // Copy the inliers of the plane to a new cloud.
    plane = DoorHandleDetectionNode::createPlaneFromInliers(cloud, inliers, coefficients);

    //Create the center of the plane
    vpColVector centroidPlane(3);
    centroidPlane = DoorHandleDetectionNode::getCenterPCL(plane);

    //Create a tf for the plane
    cMp = DoorHandleDetectionNode::createTFPlane(coeffs, centroidPlane[0], centroidPlane[1], centroidPlane[2]);

    //getCoeffPlaneWithODR(cloud, xg, yg, zg, normal);

    //Creating a cloud with all the points of the door handle
    cloud_dh = DoorHandleDetectionNode::createPCLBetweenTwoPlanes(cloud, coeffs, 0.04, 0.07);

    //Publish the pcl of the door handle with noise
    cloud_dh->header.frame_id = "softkinetic_camera_rgb_optical_frame";
    pcl_dh_pub.publish(*cloud_dh);

    if (cloud_dh->size()<50){
      ROS_INFO("No door handle detected");
    }
    else
    {
      std::vector<int> inliers2;
      Eigen::VectorXf Coeff_line;

      //Create a RandomSampleConsensus object and compute the model of a line
      pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr  model_l (new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cloud_dh));
      pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_l);
      ransac.setDistanceThreshold (.002);
      ransac.computeModel();
      ransac.getInliers(inliers2);
      ransac.getModelCoefficients(Coeff_line);
      // copies all inliers of the model computed to another PointCloud
      pcl::copyPointCloud<pcl::PointXYZ>(*cloud_dh, inliers2, *final);
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
      door_handle_final_pub.publish(*final);

      vpColVector centroidDH = DoorHandleDetectionNode::getCenterPCL(final);

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
          ROS_INFO("vec[t-2]: %lf %lf %lf", m_direction_line_pre_previous[0], m_direction_line_pre_previous[1], m_direction_line_pre_previous[2]);
          ROS_INFO("vec[t-1]: %lf %lf %lf", m_direction_line_previous[0], m_direction_line_previous[1], m_direction_line_previous[2]);
          ROS_INFO("vec[t]  : %lf %lf %lf", direction_line[0], direction_line[1], direction_line[2]);
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
//      directionLineCoeff = DoorHandleDetectionNode::getCoeffLineWithODR(final);

      //Create the door handle tf with respect to the plane tf
      DoorHandleDetectionNode::createTFLine(direction_line, normal, centroidDH[0], centroidDH[1], centroidDH[2], cRp, cMp);
    }

  }

  //Publish the point clouds of the plan
  pcl_plane_pub.publish(*plane);

}

void DoorHandleDetectionNode::createTFLine(const vpColVector coeffs, vpColVector normal, const double x, const double y, const double z, const vpRotationMatrix cRp, const vpHomogeneousMatrix cMp)
{
  vpRotationMatrix cRdh;
  vpTranslationVector Pdh;
  vpHomogeneousMatrix cMdh;
  vpRotationMatrix pRdh;
  vpHomogeneousMatrix pMdh;
  geometry_msgs::Pose cMdh_msg;
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
  centroidBorder[0] = centroidBorder[0] - 0.05;
  centroidBorder = cMdh * centroidBorder;

  transformdh.setOrigin( tf::Vector3(centroidBorder[0], centroidBorder[1], centroidBorder[2] ));

  pRdh = cRp.inverse() * cRdh;
  pMdh = vpHomogeneousMatrix(Pdh, pRdh);

  pMdh_msg = visp_bridge::toGeometryMsgsPose(pMdh);

  tf::Quaternion qdh;
  qdh.setX(cMdh_msg.orientation.x);
  qdh.setY(cMdh_msg.orientation.y);
  qdh.setZ(cMdh_msg.orientation.z);
  qdh.setW(cMdh_msg.orientation.w);

  transformdh.setRotation(qdh);
  br.sendTransform(tf::StampedTransform(transformdh, ros::Time::now(), "softkinetic_camera_rgb_optical_frame", "door_handle_tf"));
}

void DoorHandleDetectionNode::getCoeffPlaneWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const double centroidx, const double centroidy, const double centroidz, vpColVector normal)
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

}

pcl::ModelCoefficients::Ptr DoorHandleDetectionNode::getPlaneCoefficients(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.1);
  //Create the inliers and coefficients for the biggest plan found
  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);

  return coefficients;
}

pcl::PointIndices::Ptr DoorHandleDetectionNode::getPlaneInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.1);
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

  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "softkinetic_camera_rgb_optical_frame", "tf_plane"));
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

pcl::PointCloud<pcl::PointXYZ>::Ptr DoorHandleDetectionNode::createPCLBetweenTwoPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, vpColVector coefficients, double h_min, double h_max)
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
    z_min = DoorHandleDetectionNode::computeZ(coefficients, xc, yc) - h_min;
    z_max = DoorHandleDetectionNode::computeZ(coefficients, xc, yc) - h_max;
    if (zc < z_min && zc > z_max )
    {
      width_outside++;
      cloud_outside->width = width_outside;
      cloud_outside->points.resize (cloud_outside->width * cloud_outside->height);
      cloud_outside->points[width_outside-1].x = xc;
      cloud_outside->points[width_outside-1].y = yc;
      cloud_outside->points[width_outside-1].z = DoorHandleDetectionNode::computeZ(coefficients, xc, yc) - (h_max+h_min)/2;
    }
  }

  return cloud_outside;
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
  ros::init(argc,argv, "door_handle_detection");
  ros::NodeHandle n(std::string("~"));

  DoorHandleDetectionNode *node = new DoorHandleDetectionNode(n);

  node->spin();

  delete node;

  printf( "\nQuitting... \n" );
  return 0;
}

