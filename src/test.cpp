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

class RosTestNode
{
public:
  RosTestNode(ros::NodeHandle n);
  virtual ~RosTestNode();

public:
  void setup(const sensor_msgs::CameraInfo::ConstPtr& cam_ros);
  vpColVector getDirectionLineODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  vpColVector getPlaneCoefficients(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  vpColVector getCenterPCL(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  static double computeX(const vpColVector coeffs, const double y, const double z);
  static double computeY(const vpColVector coeffs, const double x, const double z);
  static double computeZ(const vpColVector coeffs, const double x, const double y);
  vpHomogeneousMatrix createTFPlane(const vpColVector coeffs, const double x, const double y, const double z);
  pcl::PointCloud<pcl::PointXYZ> createPlaneFromInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, const pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients);
  void displayImage(const sensor_msgs::Image::ConstPtr& image);
  void spin();
  void getCoeffPlaneWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const double centroidx, const double centroidy, const double centroidz, vpColVector normal);
  void getImageAndPublish(const sensor_msgs::PointCloud2::ConstPtr &image);
  void segColor(const sensor_msgs::PointCloud2::ConstPtr &image);
  void init();
  void createTFLine(const vpColVector coeffs, vpColVector normal, const double x, const double y, const double z, const vpColVector xp, const vpRotationMatrix cRp, const vpHomogeneousMatrix cMp);

protected:
  ros::NodeHandle n;
  ros::Publisher cluster_pub;
  ros::Publisher pcl_image_pub;
  ros::Publisher pcl_inside_pub;
  ros::Publisher pcl_outside_pub;
  ros::Publisher homogeneous_pub;
  ros::Publisher homogeneous2_pub;
  ros::Publisher convex_hull_pub;
  ros::Publisher coeff_pub;
  ros::Publisher center_pub;
  ros::Publisher inliers_pub;
  ros::Subscriber pcl_frame_sub;
  ros::Subscriber seg_color_sub;
  ros::Subscriber cam_sub;
  ros::Subscriber image_frame_sub;
  std::string tableTopTopicName;
  std::string imageTopicName;
  std::string cameraInfo;

  vpImage<unsigned char> img_;
  vpCameraParameters cam;
  bool cam_is_initialized;
  bool is_initialized;
  vpDisplay* m_disp;
};

bool cam_is_initialized = false;
bool is_initialized = false;

RosTestNode::RosTestNode(ros::NodeHandle nh)
{
  // read in config options
  n = nh;

  ROS_INFO("Launch Test ros node");

  //  n.param<std::string>("Image_topic_name", imageTopicName, "Image");

  n.param<std::string>("PCL_topic_name", tableTopTopicName, "/softkinetic_camera/depth/points");
  n.param<std::string>("camera_topic_name", cameraInfo, "/camera/depth_registered/camera_info");
  img_.init(240,320);

  // subscribe to services
  ROS_INFO("Beautiful weather, isn't it ?");
  cluster_pub = n.advertise<pcl::PointCloud<pcl::PointXYZ> >( "Seg_color", 1 );
  inliers_pub = n.advertise< pcl_msgs::PointIndices >("Inliers", 1);
  center_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("Center", 1);
  coeff_pub = n.advertise< pcl_msgs::ModelCoefficients >("Coeff", 1);
  homogeneous_pub = n.advertise< geometry_msgs::Pose >("Homogeneous", 1);
  homogeneous2_pub = n.advertise< geometry_msgs::Pose >("Homogeneous_dh", 1);
  pcl_image_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PCL_Plane", 1);
  pcl_inside_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PCL_Inside", 1);
  pcl_outside_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("PCL_Outside", 1);
  convex_hull_pub = n.advertise< pcl::PointCloud<pcl::PointXYZ> >("Final", 1);
  cam_sub = n.subscribe( cameraInfo, 10, (boost::function < void(const sensor_msgs::CameraInfo::ConstPtr&)>) boost::bind( &RosTestNode::setup, this, _1 ));
  pcl_frame_sub = n.subscribe( tableTopTopicName, 1000, (boost::function < void(const sensor_msgs::PointCloud2::ConstPtr&)>) boost::bind( &RosTestNode::getImageAndPublish, this, _1 ));
  //    seg_color_sub = n.subscribe( tableTopTopicName, 1, (boost::function < void(const sensor_msgs::PointCloud2::ConstPtr&)>) boost::bind( &RosTestNode::segColor, this, _1 ));

  //    image_frame_sub = n.subscribe( tableTopTopicName, 1000, (boost::function < void(const sensor_msgs::PointCloud2::ConstPtr&)>) boost::bind( &RosTestNode::displayImage, this, _1 ));
}

RosTestNode::~RosTestNode()
{
  if (is_initialized)
    delete m_disp;
}


void RosTestNode::spin()
{
  ros::Rate loop_rate(100);
  while(ros::ok()){
    //this->publish();
    ros::spinOnce();
    loop_rate.sleep();
  }
}

void RosTestNode::getImageAndPublish(const sensor_msgs::PointCloud2::ConstPtr &image)
{
  // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr convexHull(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inside(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outside(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ> center;
  pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr final2 (new pcl::PointCloud<pcl::PointXYZ>);
  //  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  //  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  vpColVector xc(3);
  vpColVector normal(3);
  vpColVector yp(3);
  vpColVector xp(3);
  vpRotationMatrix cRp;
  vpTranslationVector P0;
  vpHomogeneousMatrix cMp;
  geometry_msgs::Pose cMp_msg;
  tf::Transform transform;
  static tf::TransformBroadcaster br;
  //  init();

  img_ = 0;

  pcl::fromROSMsg (*image, *cloud);

  xc[0]=1;

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  //  pcl::ModelCoefficients coefficients;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.1);
  //Create the inliers and coefficients for the biggest plan found
  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size() == 0)
    std::cout << "Could not find a plane in the scene." << std::endl;
  else
  {
    // Copy the inliers of the plane to a new cloud.
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.filter(*plane);

    // Object for retrieving the convex hull.
    pcl::ConvexHull<pcl::PointXYZ> hull;
    hull.setInputCloud(plane);
    hull.reconstruct(*convexHull);

    // Create the point cloud to display all the plane in white
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType (pcl::SACMODEL_PLANE);
    proj.setInputCloud (cloud);
    proj.setModelCoefficients (coefficients);
    proj.filter (*cloud_projected);

    //Create the center of the plan -> pas assez de points avec le convexHull
    size_t size = cloud_projected->width;
    double sumx = 0,sumy = 0,sumz = 0;
    double xg,yg,zg,z0;
    int i;
    for (i=0;i<size;i++){
      sumx+=cloud_projected->points[i].x;
      sumy+=cloud_projected->points[i].y;
      sumz+=cloud_projected->points[i].z;
    }
    xg=sumx/size;
    yg=sumy/size;
    zg=sumz/size;

    //Create a normal to the plan
    normal[0] = -coefficients->values[0];
    normal[1] = -coefficients->values[1];
    normal[2] = -coefficients->values[2];

    normal.normalize();

    //Create yp and xp
    xp[0] = (-(coefficients->values[1]*yg + coefficients->values[2]*(zg+0.05) + coefficients->values[3])/(coefficients->values[0]))-xg;
    xp[1] = 0;
    xp[2] = 0.05;
    xp.normalize();
    yp = vpColVector::cross(normal,xp);

    //xp = vpColVector::cross(yp,normal);

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

    ROS_INFO("Is the rotation matrix of the plane really a rotation matrix : %d", cRp.isARotationMatrix());

    transform.setOrigin( tf::Vector3(xg, yg, zg) );

    //Calculate the z0
    z0=-(coefficients->values[0]*xg + coefficients->values[1]*yg + coefficients->values[3])/(coefficients->values[2]);
    //    ROS_INFO("z0 = [%lf], zg = [%lf], percentage d'erreur = [%lf %%]",z0, zg, 100*(zg-z0)/z0);

    //Create the translation Vector
    P0 = vpTranslationVector(xg, yg, z0);

    //Create the homogeneous Matrix
    cMp = vpHomogeneousMatrix(P0, cRp);
    ROS_INFO("cMp = [%lf,  %lf,  %lf,  %lf]", cMp[0][0], cMp[0][1], cMp[0][2], cMp[0][3]);
    ROS_INFO("      [%lf,  %lf,  %lf,  %lf]", cMp[1][0], cMp[1][1], cMp[1][2], cMp[1][3]);
    ROS_INFO("      [%lf,  %lf,  %lf,  %lf]", cMp[2][0], cMp[2][1], cMp[2][2], cMp[2][3]);
    ROS_INFO("      [%lf,  %lf,  %lf,  %lf]", cMp[3][0], cMp[3][1], cMp[3][2], cMp[3][3]);
    cMp_msg = visp_bridge::toGeometryMsgsPose(cMp);
    tf::Quaternion q;
    q.setValue(cMp_msg.orientation.x,cMp_msg.orientation.y,cMp_msg.orientation.z,cMp_msg.orientation.w);
    transform.setRotation(q);

//    vpQuaternionVector qp = vpQuaternionVector( cMp.getRotationMatrix() );

//    ROS_INFO_STREAM("Plan rotation matrix visp: \n" << vpRotationMatrix(qp));
    //getCoeffPlaneWithODR(cloud, xg, yg, zg, normal);

    //Testing the points if they are inside or outside the plan
    size_t taille = cloud->width;
    cloud_outside->width = 1;
    cloud_outside->height = 1;
    cloud_inside->width = 1;
    cloud_inside->height = 1;
    cloud_outside->points.resize (cloud_outside->width * cloud_outside->height);
    cloud_inside->points.resize (cloud_inside->width * cloud_inside->height);

    cloud_inside->header.frame_id = "softkinetic_camera_rgb_optical_frame";
    cloud_outside->header.frame_id = "softkinetic_camera_rgb_optical_frame";
    double xc,yc,zc,z_calc_pd,z_calc_p, z_calc_ppd;
    size_t width_inside = 0, width_outside = 0;

    for(i=0;i<taille;i++){
      xc = cloud->points[i].x;
      yc = cloud->points[i].y;
      zc = cloud->points[i].z;
      z_calc_pd = -(coefficients->values[0]*xc + coefficients->values[1]*yc + coefficients->values[3])/(coefficients->values[2]) - 0.04;
      z_calc_ppd = -(coefficients->values[0]*xc + coefficients->values[1]*yc + coefficients->values[3])/(coefficients->values[2]) - 0.08;
      //      ROS_INFO("z_calc = [%lf], z_mes = [%lf], percentage d'erreur = [%lf %%]",z_calc_pd, zc, 100*(zc-z_calc_pd)/z_calc_pd);
      if (zc < z_calc_pd && zc > z_calc_ppd )
      {
        width_outside++;
        cloud_outside->width = width_outside;
        cloud_outside->points.resize (cloud_outside->width * cloud_outside->height);
        cloud_outside->points[width_outside-1].x = xc;
        cloud_outside->points[width_outside-1].y = yc;
        cloud_outside->points[width_outside-1].z = -(coefficients->values[0]*xc + coefficients->values[1]*yc + (coefficients->values[3] + 0.06))/(coefficients->values[2]);
      }
      else if (zc < z_calc_p )
      {
        width_inside++;
        cloud_inside->width = width_inside;
        cloud_inside->points.resize (cloud_inside->width * cloud_inside->height);
        cloud_inside->points[width_inside-1].x = xc;
        cloud_inside->points[width_inside-1].y = yc;
        cloud_inside->points[width_inside-1].z = zc;
      }

    }
    //Publish points inside the plan
    pcl_inside_pub.publish(*cloud_inside);

    //Publish points outside the plan
    pcl_outside_pub.publish(*cloud_outside);

    if (cloud_outside->size()<50){
      ROS_INFO("No door handle detected");
    }
    else
    {
      std::vector<int> inliers2;
      Eigen::VectorXf Coeff_line;
      vpColVector direction_line(3);

      // created RandomSampleConsensus object and compute the appropriated model
      pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr  model_l (new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cloud_outside));
      pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_l);
      ransac.setDistanceThreshold (.002);
      ransac.computeModel();
      ransac.getInliers(inliers2);
      ransac.getModelCoefficients(Coeff_line);
      // copies all inliers of the model computed to another PointCloud
      pcl::copyPointCloud<pcl::PointXYZ>(*cloud_outside, inliers2, *final);
      ROS_INFO("vec1: %lf %lf %lf", Coeff_line[3], Coeff_line[4], Coeff_line[5]);
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

      sumx=0;
      sumy=0;
      sumz=0;
      double xg_dh, yg_dh, zg_dh;
      for (int i=0; i<final->size(); i++){
        sumx+=final->points[i].x;
        sumy+=final->points[i].y;
        sumz+=final->points[i].z;
      }
      xg_dh=sumx/final->size();
      yg_dh=sumy/final->size();
      zg_dh=sumz/final->size();
////////////INITIALISATION///////////
//      int nbcoef_model = 2;
//      vpMatrix A;
//      vpRowVector coords(2);
//      vpColVector B(final->size(),1);
//      vpColVector Theta(nbcoef_model);
//      vpMatrix W(final->size(), final->size()) ; // Weight matrix
//      vpColVector w(final->size()) ;

//      // All the weights are set to 1 at the beginning to use a classical least
//      // square scheme
//      w = 1;
//      // Update the square matrix associated to the weights
//      for (int i=0; i < final->size(); i ++) {
//        W[i][i] = w[i];
//      }


//      for (int i = 0; i < final->size(); i++)
//      {
//        coords[0] = final->points[i].x;
//        coords[1] = final->points[i].y;
//        A.stack(coords);
//      }

//      ///////////////////////////////////////////////////////////////////
//      // Debut de l'estimation

//      // Initialisation du model a estimer
//      Theta = 0;


////      std::cout << "True ground motion model: " << std::endl
////          << model << std::endl;

//      // Theta_tmp = [WX]^+ WY
//      vpColVector Theta_tmp(nbcoef_model); // resultat du moindre carre pondere
//      int iter = 0;
//      int niter_max = 5;
//      double distance = 1;
//      vpRobust r(final->size()) ; // M-Estimator

//      vpMatrix WA;
//      // Weighted iterated least square estimation
//      while (iter < niter_max && distance > 0.1) {
//        // Compute the motion model parameters by weighted least square estimation
//        WA = W*A;
//        Theta_tmp = WA.pseudoInverse(1e-26) *W*B;

////        std::cout << "Model estime iter " << iter << ": " << std::endl
////            << Theta_tmp << std::endl;

//        vpColVector residu;

//        residu = W*B - WA * Theta_tmp;
//        //    std::cout << "residu: " << std::endl << residu << std::endl;

//        // Compute the weights using the Tubey biweight M-Estimator
//        r.setIteration(iter) ;
//        r.MEstimator(vpRobust::TUKEY, residu, w) ;

////        std::cout << "Weights: " << std::endl << w << std::endl;

//        // Update the weights matrix
//        for (int i=0; i < final->size(); i ++) {
//          W[i][i] = w[i];
//        }

//        // Compute the distance between the last estimated model and the previous
//        // one
//        distance = 0;
//        for (int i=0; i < nbcoef_model; i ++) {
//          distance += fabs(Theta[i] - Theta_tmp[i]);
//        }
////        std::cout << "Distance between estimated motion models: "
////            << distance << std::endl;

//        // Update the new motion model
//        Theta = Theta_tmp;

//        iter ++;
//      }


//      std::cout << "True ground motion model: " << std::endl
//          << model << std::endl;
//      std::cout << "Estimated motion model:  " << std::endl
//          << Theta << std::endl;


      ////For a 3d system
//      vpMatrix A;
//      vpRowVector coords(4);
//      vpColVector B;

//      for (int i = 0; i < final->size(); i++)
//      {
//        coords[0] = 1;
//        coords[1] = 0;
//        coords[2] = final->points[i].x;
//        coords[3] = 0;
//        A.stack(coords);
//        coords[0] = 0;
//        coords[1] = 1;
//        coords[2] = 0;
//        coords[3] = final->points[i].x;
//        A.stack(coords);
//        B.stack( final->points[i].y );
//        B.stack( final->points[i].z );
//      }
//      ROS_INFO(" de A = %d", A.getRows());
//      vpColVector Coeffs(4);
//      Coeffs = A.pseudoInverse()*B;
//      ROS_INFO_STREAM("Coeff : " << Coeffs.t());
      ////For a 2d system
//      vpMatrix A;
//      vpRowVector coords(2);
//      vpColVector B(final->size(),1);

//      for (int i = 0; i < final->size(); i++)
//      {
//        coords[0] = final->points[i].x;
//        coords[1] = final->points[i].y;
//        A.stack(coords);
//      }
//      ROS_INFO("Size de A = %d", A.getRows());

//      vpColVector first_coeff(2);

//      first_coeff = A.pseudoInverse()*B;

      vpColVector center_dh(4);
      center_dh[0] = xg_dh;
      center_dh[1] = yg_dh;
      center_dh[2] = zg_dh;
      center_dh[3] = 1;
      center_dh = cMp.inverse() * center_dh;

      ////Method of the ODR Line
//      // Minimization
//      vpColVector v(3);
//      vpMatrix M;
//      vpRowVector m(3);

//      for(unsigned int i=0; i<final->size(); i++) {
//        m[0] = final->points[i].x - xg_dh;
//        m[1] = final->points[i].y - yg_dh;
//        m[2] = final->points[i].z - zg_dh;
//        M.stack(m);
//      }
////      std::cout << "M:\n" << M << std::endl;

//      vpMatrix A = M.t() * M;

//      vpColVector D;
//      vpMatrix V;
//      A.svd(D, V);

//      std::cout << "A:\n" << A << std::endl;
//      std::cout << "D:" << D.t() << std::endl;
//      std::cout << "V:\n" << V << std::endl;

//      double largestSv = D[0];
//      unsigned int indexLargestSv = 0 ;
//      for (unsigned int i = 1; i < D.size(); i++) {
//        if ((D[i] > largestSv) ) {
//          largestSv = D[i];
//          indexLargestSv = i;
//        }
//      }

//      vpColVector h = V.getCol(indexLargestSv);
//      std::cout << "Estimated line vector: " << h.t() << std::endl;

       //Create the door handle tf with respect to the plane tf
      RosTestNode::createTFLine(direction_line, normal, center_dh[0], center_dh[1], center_dh[2], xp, cRp, cMp);

      //Create the door handle tf with respect to the camera tf
//      RosTestNode::createTFLine(h, normal, xg_dh, yg_dh, zg_dh, xp, cRp, cMp);

      //    double X,Y,Z,x,y,u,v;
      //    taille = width_outside;

      //    for(i=0;i<taille;i++){
      //      X = cloud_outside->points[i].x;
      //      Y = cloud_outside->points[i].y;
      //      Z = cloud_outside->points[i].z;
      //      x = X/Z;
      //      y = Y/Z;

      //      vpMeterPixelConversion::convertPoint(cam, x, y, u, v);

      //      if(u < img_.getWidth() && v < img_.getHeight())
      //        img_.bitmap[int(v)*img_.getWidth()+int(u)] = 255;
      //    }
      //    vpImageMorphology::dilatation(img_,(unsigned char) 255,(unsigned char) 0, vpImageMorphology::CONNEXITY_4);
      //    vpImageMorphology::dilatation(img_,(unsigned char) 255,(unsigned char) 0, vpImageMorphology::CONNEXITY_4);
      //    vpImageMorphology::erosion(img_,(unsigned char) 255,(unsigned char) 0, vpImageMorphology::CONNEXITY_4);
      //    vpImageMorphology::erosion(img_,(unsigned char) 255,(unsigned char) 0, vpImageMorphology::CONNEXITY_4);
      //    vpImageMorphology::erosion(img_,(unsigned char) 255,(unsigned char) 0, vpImageMorphology::CONNEXITY_4);
      //    vpImageMorphology::dilatation(img_,(unsigned char) 255,(unsigned char) 0, vpImageMorphology::CONNEXITY_4);

      //    //int w,h;
      //    double maxx=-1000, minx=1000;
      //    vpMatrix temp(102,2);
      //    i=0;
      //    for(int h=0; h<img_.getHeight() ; h++)
      //    {
      //      for(int w=0; w<img_.getWidth(); w++ )
      //      {
      //        if (img_.bitmap[h*img_.getWidth()+w]>250){
      //          vpPixelMeterConversion::convertPoint(cam, w, h, x, y);
      //          //          ROS_INFO("x = %lf, y = %lf.", x, y);
      //          if (i>100)
      //            break;
      //          if(maxx<x)
      //          {
      //            maxx=x;
      //          }
      //          if(minx>x)
      //          {
      //            minx=x;
      //          }
      //          temp[i][0] = x;
      //          temp[i][1] = y;
      //          i++;
      //        }
      //      }
      //      if (i>100)
      //        break;

      //    }

      //    ROS_INFO("i = %d",i);
      //    vpMatrix A(i,2);
      //    vpColVector B(i);

      //    for(int h=0; h<i; h++)
      //    {
      //      A[h][0]=temp[h][0];
      //      A[h][1]=temp[h][1];
      //      B[h]=1;
      //    }

      //    vpColVector Line(2);

      //    Line = A.pseudoInverse()*B;
      //    ROS_INFO("a= %lf, b = %lf", Line[0], Line[1]);

      //    double x1, y1, x2, y2, u1, v1, u2, v2;
      //    x1 = 0;
      //    y1 = (1-Line[0]*x1) / Line[1];

      //    x2 = maxx;
      //    y2 = (1-Line[0]*x2) / Line[1];
      //    ROS_INFO("x1= %lf, y1 = %lf, x2 = %lf, y2= %lf", x1, y1, x2, y2);

      //    vpMeterPixelConversion::convertPoint(cam, x1, y1, u1, v1);
      //    vpMeterPixelConversion::convertPoint(cam, x2, y2, u2, v2);

      //    ROS_INFO("u1= %lf, v1 = %lf, u2= %lf, v2= %lf   cam.py = %lf", u1, v1, u2, v2, cam.get_py());
      //    vpColor color(255,0,0);

      //    vpDisplay::display(img_);
      //    vpDisplay::displayLine(img_, int(v1), int(u1), int(v2), int(u2), color, 2 );
      //    vpDisplay::flush(img_);
      //    if (vpDisplay::getClick(img_, false))
      //      ros::shutdown();
    }

  }

  // Publish the model coefficients
  pcl_msgs::ModelCoefficients ros_coefficients;
  pcl_conversions::fromPCL(*coefficients, ros_coefficients);
  coeff_pub.publish (ros_coefficients);

  //Publish the convex hull
  convex_hull_pub.publish(*final);

  //Publish a tf for the plane
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "softkinetic_camera_rgb_optical_frame", "plane_tf"));

  //Publish the Homogeneous Matrix
  homogeneous_pub.publish(cMp_msg);

  //Publish the point clouds of the plan
  pcl_image_pub.publish(*cloud_projected);

  //  //Publish the inliers of the plan
  //  pcl_msgs::PointIndices ros_inliers;
  //  pcl_conversions::fromPCL(*inliers, ros_inliers);
  //  inliers_pub.publish(ros_inliers);

  //Publishing the center of the plan
  center.header.frame_id = "softkinetic_camera_rgb_optical_frame";
  center_pub.publish(center);


  //pcl_image_pub.publish(cloud);
  //  ROS_INFO("Publishing something");
}

void RosTestNode::createTFLine(const vpColVector coeffs, vpColVector normal, const double x, const double y, const double z, const vpColVector xp, const vpRotationMatrix cRp, const vpHomogeneousMatrix cMp)
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

  direction_line[0]=coeffs[0];
  direction_line[1]=coeffs[1];
  direction_line[2]=coeffs[2];

  direction_line.normalize();
  ROS_INFO("a= %lf, b = %lf, c = %lf", direction_line[0], direction_line[1], direction_line[2]);
  ROS_INFO("xg= %lf, yg = %lf, zg = %lf", x, y, z);

  vpColVector y_dh(3);
  y_dh = vpColVector::cross(normal,direction_line);
  double angle = acos(direction_line*xp);

  ROS_INFO("angle entre plan et poignee : %lf", angle);

  ////Create the Rotation Matrix
  //with the vector
  cRdh[0][0] = direction_line[0];
  cRdh[1][0] = direction_line[1];
  cRdh[2][0] = direction_line[2];
  cRdh[0][1] = y_dh[0];
  cRdh[1][1] = y_dh[1];
  cRdh[2][1] = y_dh[2];
  cRdh[0][2] = normal[0];
  cRdh[1][2] = normal[1];
  cRdh[2][2] = normal[2];

  transformdh.setOrigin( tf::Vector3(x, y, z ));

  //Create the translation Vector
  Pdh = vpTranslationVector(x, y, z);

  //Create the homogeneous Matrix
  cMdh = vpHomogeneousMatrix(Pdh, cRdh);
  ROS_INFO("cMdh= [%lf,  %lf,  %lf,  %lf]", cMdh[0][0], cMdh[0][1], cMdh[0][2], cMdh[0][3]);
  ROS_INFO("      [%lf,  %lf,  %lf,  %lf]", cMdh[1][0], cMdh[1][1], cMdh[1][2], cMdh[1][3]);
  ROS_INFO("      [%lf,  %lf,  %lf,  %lf]", cMdh[2][0], cMdh[2][1], cMdh[2][2], cMdh[2][3]);
  ROS_INFO("      [%lf,  %lf,  %lf,  %lf]", cMdh[3][0], cMdh[3][1], cMdh[3][2], cMdh[3][3]);
  cMdh_msg = visp_bridge::toGeometryMsgsPose(cMdh);

  bool isrot = cRdh.isARotationMatrix();
  ROS_INFO("cRdh is a rotation matrix : %d", isrot);
  isrot = cMdh.isAnHomogeneousMatrix();
  ROS_INFO("cMdh is a homogeneous matrix : %d", isrot);

  pRdh = cRp.inverse() * cRdh;
  pMdh = vpHomogeneousMatrix(Pdh, pRdh);
  isrot = pMdh.isAnHomogeneousMatrix();
  ROS_INFO("pMdh is a homogeneous matrix : %d", isrot);
//  pMdh = cMp.inverse() * cMdh ;
//  pMdh[2][0] = 0;
  ROS_INFO("AApMdh= [%lf,  %lf,  %lf,  %lf]", pMdh[0][0], pMdh[0][1], pMdh[0][2], pMdh[0][3]);
  ROS_INFO("AA      [%lf,  %lf,  %lf,  %lf]", pMdh[1][0], pMdh[1][1], pMdh[1][2], pMdh[1][3]);
  ROS_INFO("AA      [%lf,  %lf,  %lf,  %lf]", pMdh[2][0], pMdh[2][1], pMdh[2][2], pMdh[2][3]);
  ROS_INFO("AA      [%lf,  %lf,  %lf,  %lf]", pMdh[3][0], pMdh[3][1], pMdh[3][2], pMdh[3][3]);

  pMdh_msg = visp_bridge::toGeometryMsgsPose(pMdh);

  vpQuaternionVector q = vpQuaternionVector( pMdh.getRotationMatrix() );

  ROS_INFO_STREAM(" rotation matrix visp: \n" << vpRotationMatrix(q));

  tf::Quaternion qdh;
  qdh.setX(pMdh_msg.orientation.x);
  qdh.setY(pMdh_msg.orientation.y);
  qdh.setZ(pMdh_msg.orientation.z);
  qdh.setW(pMdh_msg.orientation.w);
  ROS_INFO(" quarternion ros: %lf %lf %lf %lf", qdh.getX(), qdh.getY(), qdh.getZ(), qdh.getW());

//  qdh.setValue(cMdh_msg.orientation.x,cMdh_msg.orientation.y,cMdh_msg.orientation.z,cMdh_msg.orientation.w);
  transformdh.setRotation(qdh);
  //respect to the plane tf
  br.sendTransform(tf::StampedTransform(transformdh, ros::Time::now(), "plane_tf", "door_handle_tf"));
  //respect to the camera tf
//  br.sendTransform(tf::StampedTransform(transformdh, ros::Time::now(), "softkinetic_camera_rgb_optical_frame", "door_handle_tf"));
  homogeneous2_pub.publish(pMdh_msg);
//  test = visp_bridge::toVispHomogeneousMatrix(cMdh_msg);
//  ROS_INFO_STREAM(" Test cMdh\n" << test);
}

void RosTestNode::getCoeffPlaneWithODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const double centroidx, const double centroidy, const double centroidz, vpColVector normal)
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
//  std::cout << "M:\n" << M << std::endl;

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

  ROS_INFO_STREAM("Ground DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDtrue normal vector: " << normal.t() << "\n");

  ROS_INFO_STREAM("Estimated SSSSSSSSSSSSSSnormal Vector: " << h.t() << "\n");

}

vpColVector RosTestNode::getPlaneCoefficients(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
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

  vpColVector coeffs;
  coeffs.stack(coefficients->values[0]);
  coeffs.stack(coefficients->values[1]);
  coeffs.stack(coefficients->values[2]);
  coeffs.stack(coefficients->values[3]);

  return coeffs;
}

double RosTestNode::computeX(const vpColVector coeffs, const double y, const double z)
{
  double x = -(coeffs[1]*y + coeffs[2]*z + coeffs[3])/(coeffs[0]);
  return x;
}

double RosTestNode::computeY(const vpColVector coeffs, const double x, const double z)
{
  double y = -(coeffs[0]*x + coeffs[2]*z + coeffs[3])/(coeffs[1]);
  return y;
}

double RosTestNode::computeZ(const vpColVector coeffs, const double x, const double y)
{
  double z = -(coeffs[0]*x + coeffs[1]*y + coeffs[3])/(coeffs[2]);
  return z;
}

vpHomogeneousMatrix RosTestNode::createTFPlane(const vpColVector coeffs, const double x, const double y, const double z)
{
  vpColVector xp;
  vpColVector yp;
  vpColVector normal;
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
  xp[1] = 0;
  xp[2] = 0.05;
  xp[0] = RosTestNode::computeX(coeffs, y, z+0.05) - x;
  xp.normalize();
  yp = vpColVector::cross(normal,xp);

  //xp = vpColVector::cross(yp,normal);

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

  ROS_INFO("Is the rotation matrix of the plane really a rotation matrix : %d", cRp.isARotationMatrix());

  transform.setOrigin( tf::Vector3(x, y, z) );

  //Calculate the z0
  double z0 = RosTestNode::computeZ(coeffs, x, y);

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

vpColVector RosTestNode::getCenterPCL(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  vpColVector centroid;
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

vpColVector RosTestNode::getDirectionLineODR(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  vpMatrix M;
  vpRowVector m(3);
  vpColVector centroid = RosTestNode::getCenterPCL(cloud);

  for(unsigned int i=0; i<cloud->size(); i++) {
    m[0] = cloud->points[i].x - centroid[0];
    m[1] = cloud->points[i].y - centroid[1];
    m[2] = cloud->points[i].z - centroid[2];
    M.stack(m);
  }
//      std::cout << "M:\n" << M << std::endl;

  vpMatrix A = M.t() * M;

  vpColVector D;
  vpMatrix V;
  A.svd(D, V);

//      std::cout << "A:\n" << A << std::endl;
//      std::cout << "D:" << D.t() << std::endl;
//      std::cout << "V:\n" << V << std::endl;

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
void RosTestNode::segColor(const sensor_msgs::PointCloud2::ConstPtr &image)
{
  // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_colored(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::fromROSMsg (*image, *cloud_colored);

  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  kdtree->setInputCloud(cloud_colored);

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(kdtree);
  normal_estimator.setInputCloud(cloud_colored);
  normal_estimator.setKSearch(50);
  normal_estimator.compute(*normals);

  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(cloud_colored);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0.0, 1.0);
  pass.filter(*indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize(50);
  reg.setMaxClusterSize(1000000);
  reg.setSearchMethod(kdtree);
  reg.setNumberOfNeighbours(30);
  reg.setInputCloud(cloud_colored);
  //reg.setIndices (indices);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold(1.0);

  std::vector <pcl::PointIndices> clusters;
  reg.extract(clusters);

  // kd-tree object for searches.
  //  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  //  kdtree->setInputCloud(cloud_colored);

  //  // Color-based region growing clustering object.
  //  pcl::RegionGrowingRGB<pcl::PointXYZRGB> clustering;
  //  clustering.setInputCloud(cloud_colored);
  //  clustering.setSearchMethod(kdtree);
  //  // Here, the minimum cluster size affects also the postprocessing step:
  //  // clusters smaller than this will be merged with their neighbors.
  //  clustering.setMinClusterSize(70);
  //  // Set the distance threshold, to know which points will be considered neighbors.
  //  clustering.setDistanceThreshold(5);
  //  // Color threshold for comparing the RGB color of two points.
  //  clustering.setPointColorThreshold(6);
  //  // Region color threshold for the postprocessing step: clusters with colors
  //  // within the threshold will be merged in one.
  //  clustering.setRegionColorThreshold(2);

  //  std::vector <pcl::PointIndices> clusters;
  //  clustering.extract(clusters);

  pcl::PointCloud <pcl::PointXYZRGB>::Ptr cluster_colored = reg.getColoredCloud ();
  cluster_colored->header.frame_id = "softkinetic_camera_rgb_optical_frame";

  //Publish the Segmentation colored cloud
  cluster_pub.publish(*cluster_colored);

}
void RosTestNode::setup(const sensor_msgs::CameraInfo::ConstPtr& cam_ros)
{
  if (! cam_is_initialized) {
    //init camera parameters
    cam = visp_bridge::toVispCameraParameters(*cam_ros);
    cam_is_initialized = true;
  }
}
void RosTestNode::init()
{
  if (! is_initialized) {
    //init graphical interface
    m_disp = new vpDisplayX();
    m_disp->init(img_);
    m_disp->setTitle("Image viewer");
    vpDisplay::flush(img_);
    vpDisplay::display(img_);
    ROS_INFO("Initialisation done");
    vpDisplay::flush(img_);

    is_initialized = true;
  }
}

void RosTestNode::displayImage(const sensor_msgs::Image::ConstPtr& image)
{
  img_ = visp_bridge::toVispImage(*image);

  init();

  vpDisplay::display(img_);
  vpDisplay::flush(img_);
  if (vpDisplay::getClick(img_, false))
    ros::shutdown();

}


int main( int argc, char** argv )
{
  ros::init(argc,argv, "test");
  ros::NodeHandle n(std::string("~"));

  RosTestNode *node = new RosTestNode(n);

  //  if( node->setup() != 0 )
  //  {
  //    printf( "Test setup failed... \n" );
  //    return -1;
  //  }

  node->spin();

  delete node;

  printf( "\nQuitting... \n" );
  return 0;
}

