cmake_minimum_required(VERSION 3.8)
project(hsim_mujoco_plugins)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(mujoco_vendor REQUIRED)
find_package(mujoco_ros2_control_msgs REQUIRED)
find_package(mujoco_ros2_control_plugins REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(wisco_ethercat_ros2_hw REQUIRED)
find_package(yaml-cpp REQUIRED)

# GroundTruthPlugin: the simulated world's state, from inside mujoco_ros2_control (sim stack, backend:=ros2_control)
add_library(hsim_mujoco_plugins SHARED src/ground_truth_plugin.cpp)
target_compile_features(hsim_mujoco_plugins PUBLIC cxx_std_17)
target_include_directories(hsim_mujoco_plugins PRIVATE include)
target_link_libraries(hsim_mujoco_plugins mujoco_ros2_control_plugins::mujoco_ros2_control_plugins pluginlib::pluginlib)
ament_target_dependencies(hsim_mujoco_plugins rclcpp std_msgs)
pluginlib_export_plugin_description_file(mujoco_ros2_control_plugins hsim_mujoco_plugins.xml)

# VirtualRobotSystem: the robot's EtherCAT hardware simulated, drives on MuJoCo (sim stack, backend:=ethercat_sim)
add_library(hsim_virtual_robot SHARED src/virtual_robot_system.cpp)
target_compile_features(hsim_virtual_robot PUBLIC cxx_std_17)
target_include_directories(hsim_virtual_robot PRIVATE include)
target_link_libraries(hsim_virtual_robot mujoco::mujoco pluginlib::pluginlib yaml-cpp
  wisco_ethercat_ros2_hw::wisco_ethercat_ros2_hw_mock_drives)
ament_target_dependencies(hsim_virtual_robot hardware_interface rclcpp rclcpp_lifecycle mujoco_ros2_control_msgs
  nav_msgs rosgraph_msgs std_msgs wisco_ethercat_ros2_hw)
pluginlib_export_plugin_description_file(hardware_interface virtual_robot_plugin.xml)

install(TARGETS hsim_mujoco_plugins hsim_virtual_robot ARCHIVE DESTINATION lib LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # the plugin on a real world: drives enable, the robot stands under mode-5 PD, a two-motor joint maps both ways
  ament_add_gtest(test_virtual_robot test/test_virtual_robot.cpp TIMEOUT 120)
  target_include_directories(test_virtual_robot PRIVATE include)
  target_link_libraries(test_virtual_robot hsim_virtual_robot mujoco::mujoco)
  ament_target_dependencies(test_virtual_robot hardware_interface rclcpp wisco_ethercat_ros2_hw)
  target_compile_definitions(test_virtual_robot PRIVATE
    REPO="${CMAKE_CURRENT_SOURCE_DIR}/../../..")
endif()

ament_package()
