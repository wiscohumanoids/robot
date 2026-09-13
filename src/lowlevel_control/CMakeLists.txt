cmake_minimum_required(VERSION 3.8)
project(lowlevel_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(humanoid_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(lowlevel_control SHARED
  src/joint_impedance_controller.cpp
)
target_include_directories(lowlevel_control PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(lowlevel_control
  rclcpp
  rclcpp_lifecycle
  controller_interface
  hardware_interface
  pluginlib
  realtime_tools
  humanoid_interfaces
  std_msgs
)
target_link_libraries(lowlevel_control Eigen3::Eigen)

pluginlib_export_plugin_description_file(controller_interface lowlevel_control.xml)

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS lowlevel_control
  EXPORT export_lowlevel_control
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
ament_export_include_directories(include)
ament_export_libraries(lowlevel_control)
ament_export_targets(export_lowlevel_control HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp rclcpp_lifecycle controller_interface hardware_interface pluginlib
  realtime_tools humanoid_interfaces std_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
