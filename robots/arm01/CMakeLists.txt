# arm01_description: installs the arm's description so ROS tools resolve package://arm01_description/...
cmake_minimum_required(VERSION 3.8)
project(arm01_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes config calibration cad DESTINATION share/${PROJECT_NAME})

ament_package()
