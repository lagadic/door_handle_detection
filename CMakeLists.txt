cmake_minimum_required(VERSION 2.8.3)
project(test_3d)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  roscpp
  std_msgs
  visp_bridge
  image_transport
#  visp_ros
  pcl_msgs
  pcl_ros
  pcl_conversions
)

find_package(VISP REQUIRED)
find_package(PCL 1.2 REQUIRED)

add_definitions(
${VISP_DEFINITIONS}
${PCL_DEFINITIONS}
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    include

  LIBRARIES
    ${PROJECT_NAME}

  CATKIN_DEPENDS
    roscpp
    std_msgs
    visp_bridge
    sensor_msgs 

  DEPENDS
    VISP
)

###########
## Build ##
###########
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${VISP_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

include_directories(include)

link_directories(${PCL_LIBRARY_DIRS})
## Declare a cpp library
add_library(test_3d
  include/door_handle_detection.h
  src/door_handle_detection.cpp
)

## Declare a cpp executable
add_executable(test_3d_node src/test.cpp)
add_executable(door_handle_detection_node src/door_handle_detection.cpp)


## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(dots_tracker_node dots_tracker_generate_messages_cpp)

## Specify libraries to link a library or executable target against
 target_link_libraries(test_3d_node
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
 )
 target_link_libraries(door_handle_detection_node
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
 )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS dots_tracker dots_tracker_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_dots_tracker.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)



#add_executable(dots_tracker 
#		src/dots_tracker.cpp
#		src/VScam.h
#		src/VScam.cpp
#		)


