cmake_minimum_required(VERSION 3.8)
project(nav2_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config maps
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/avoidance_test.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
