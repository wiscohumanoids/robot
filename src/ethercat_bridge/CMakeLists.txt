cmake_minimum_required(VERSION 3.8)
project(ethercat_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(ethercat_bridge SHARED
  src/ethercat_hardware_interface.cpp
)
target_include_directories(ethercat_bridge PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(ethercat_bridge
  rclcpp
  rclcpp_lifecycle
  hardware_interface
  pluginlib
  ament_index_cpp
)
target_link_libraries(ethercat_bridge yaml-cpp)

pluginlib_export_plugin_description_file(hardware_interface ethercat_bridge.xml)

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS ethercat_bridge
  EXPORT export_ethercat_bridge
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(ethercat_bridge)
ament_export_targets(export_ethercat_bridge HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp rclcpp_lifecycle hardware_interface pluginlib ament_index_cpp
)

if(BUILD_TESTING)
  # Skipped on purpose: the copyright check needs a licence/copyright-holder decision
  # that has not been made, and cpplint/uncrustify enforce ROS 2's K&R brace style while
  # this repo uses Allman (like the vendored mujoco_ros2_control). The remaining linters
  # (cppcheck, lint_cmake, xmllint) still run. See STATUS.md, Known gaps.
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
