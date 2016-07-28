# Door handle detection

## Description:
A node used to detect a door handle.

Node tested on the camera Intel Realsense SR300

## Instructions:
* Launch ROS node of the camera:
`$ roslaunch realsense_camera realsense_sr300.launch`
* Launch detection node:
`$ roslaunch door_handle_detection door_handle_detection_realsense.launch`

## Description of the algorithm step by step:

* Detection of the biggest plane of the point cloud which should be the door
* Creation of a point cloud which is only composed of points that between 0.05m and 0.09m from the plane detected
* Computation of the axis of the door handle which is inside the new point cloud created
* Localisation of the pose of the door handle
* Usage of a kalman filter to reduce the noise
* Creation of a bounding box to reduce the detection
* Go back to step 2 with points that are only inside the bounding box of the previous door handle