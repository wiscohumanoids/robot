cmake_minimum_required(VERSION 3.8)
project(wisco_ethercat_ros2_hw)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
# The EtherCAT pieces (ethercat_driver_ros2 packages and the IgH library) exist only on the robot computer;
# without them the controllers and tests still build, and pp_test / test_pp_pdos are skipped.
find_package(ament_index_cpp REQUIRED)
find_package(ethercat_generic_slave QUIET)
find_package(ethercat_interface QUIET)
find_library(IGH_LIBRARY NAMES ethercat PATHS /usr/local/etherlab/lib)
if(ethercat_generic_slave_FOUND AND ethercat_interface_FOUND AND IGH_LIBRARY)
  set(HAVE_ETHERCAT TRUE)
else()
  set(HAVE_ETHERCAT FALSE)
  message(STATUS "EtherCAT libraries not found: skipping pp_test and test_pp_pdos")
endif()

find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(std_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(pinocchio REQUIRED)
find_package(yaml-cpp REQUIRED)

# --- SafeCommandController: the command controller the launch file loads ---
add_library(wisco_ethercat_ros2_hw_controllers SHARED src/safe_command_controller.cpp)
target_compile_features(wisco_ethercat_ros2_hw_controllers PUBLIC cxx_std_17)
target_include_directories(wisco_ethercat_ros2_hw_controllers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
ament_target_dependencies(wisco_ethercat_ros2_hw_controllers
  controller_interface hardware_interface pluginlib rclcpp rclcpp_lifecycle realtime_tools
  std_msgs)

# --- ArmController: the arm's 1 kHz Mode P / Mode T loop (arm_law.hpp + Pinocchio gravity) ---
add_library(wisco_ethercat_ros2_hw_arm_controller SHARED src/arm_controller.cpp src/gravity_model.cpp)
target_compile_features(wisco_ethercat_ros2_hw_arm_controller PUBLIC cxx_std_17)
target_include_directories(wisco_ethercat_ros2_hw_arm_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
ament_target_dependencies(wisco_ethercat_ros2_hw_arm_controller
  controller_interface hardware_interface pluginlib rclcpp rclcpp_lifecycle realtime_tools
  std_msgs)
target_link_libraries(wisco_ethercat_ros2_hw_arm_controller pinocchio::pinocchio yaml-cpp)

pluginlib_export_plugin_description_file(controller_interface safe_command_controller_plugin.xml)
install(TARGETS wisco_ethercat_ros2_hw_controllers wisco_ethercat_ros2_hw_arm_controller
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

# --- current_sensor_node: power-rail currents (ADS7128 + ACS758 over I2C) -> /power/currents ---
add_executable(current_sensor_node src/current_sensor_node.cpp)
target_compile_features(current_sensor_node PRIVATE cxx_std_17)
ament_target_dependencies(current_sensor_node rclcpp diagnostic_msgs)
install(TARGETS current_sensor_node DESTINATION lib/${PROJECT_NAME})

# --- pp_test: profile-position drive check, talks to the IgH master directly ---
if(HAVE_ETHERCAT)
  add_executable(pp_test src/pp_test.cpp)
  target_compile_features(pp_test PRIVATE cxx_std_17)
  ament_target_dependencies(pp_test ament_index_cpp ethercat_interface rclcpp)
  target_link_libraries(pp_test ethercat_generic_slave::ethercat_generic_slave ${IGH_LIBRARY} yaml-cpp)
  install(TARGETS pp_test DESTINATION lib/${PROJECT_NAME})
endif()

# --- Python: config generation used by the launch file and scripts ---
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config urdf launch
  DESTINATION share/${PROJECT_NAME}
)
# arm stack (calibration procedures, sim, PACE fit, `arm` CLI); importable from
# share/ via scripts/arm, or from the source tree with `python3 -m arm_stack.cli`
install(DIRECTORY arm_stack DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)
install(PROGRAMS scripts/arm DESTINATION lib/${PROJECT_NAME})
install(
  PROGRAMS
    scripts/multiple_act_command.py
    scripts/multi_velocity_cmd.py
    scripts/bringup_check.py
    scripts/check_urdf.py
    scripts/mocap_check.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_gtest(test_safety_logic test/test_safety_logic.cpp)
  target_include_directories(test_safety_logic PRIVATE include)

  ament_add_gtest(test_pp_motion test/test_pp_motion.cpp)
  target_include_directories(test_pp_motion PRIVATE src)
  target_compile_features(test_pp_motion PRIVATE cxx_std_17)

  if(HAVE_ETHERCAT)
    ament_add_gtest(test_pp_pdos test/test_pp_pdos.cpp)
    ament_target_dependencies(test_pp_pdos ethercat_interface)
    target_link_libraries(test_pp_pdos ethercat_generic_slave::ethercat_generic_slave yaml-cpp)
    target_compile_definitions(test_pp_pdos PRIVATE
      PP_TEST_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/pp_test.yaml")
  endif()

  ament_add_gtest(test_arm_law test/test_arm_law.cpp)
  target_include_directories(test_arm_law PRIVATE include)

  # Mock RMD-X drives on a mock EtherCAT ring (test/mock/), so the CiA402
  # state machine and the RT core run before the drives arrive.
  add_library(wisco_mock_ethercat STATIC
    test/mock/rmdx_mock_slave.cpp
    test/mock/mock_ethercat_system.cpp)
  set_target_properties(wisco_mock_ethercat PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_features(wisco_mock_ethercat PUBLIC cxx_std_17)
  target_include_directories(wisco_mock_ethercat PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/test/mock ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_target_dependencies(wisco_mock_ethercat hardware_interface rclcpp)

  ament_add_gtest(test_mock_slave test/test_mock_slave.cpp)
  target_link_libraries(test_mock_slave wisco_mock_ethercat)

  ament_add_gtest(test_arm_controller_mock test/test_arm_controller_mock.cpp TIMEOUT 300)
  target_link_libraries(test_arm_controller_mock
    wisco_mock_ethercat wisco_ethercat_ros2_hw_arm_controller wisco_ethercat_ros2_hw_controllers)
  ament_target_dependencies(test_arm_controller_mock
    controller_interface hardware_interface rclcpp rclcpp_lifecycle std_msgs)

  # C++ GravityModel (the controller's) against the Python ArmModel (sim,
  # calibration): golden file written by tools/make_parity_golden.py.
  ament_add_gtest(test_gravity_parity test/test_gravity_parity.cpp)
  target_link_libraries(test_gravity_parity wisco_ethercat_ros2_hw_arm_controller yaml-cpp)
  target_compile_definitions(test_gravity_parity PRIVATE
    WISCO_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

  ament_add_pytest_test(test_arm_stack test/arm_stack
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} TIMEOUT 900)

  ament_add_pytest_test(test_hardware_config test/test_hardware_config.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(controller_interface hardware_interface pluginlib rclcpp
  rclcpp_lifecycle realtime_tools std_msgs pinocchio yaml-cpp)
ament_package()
