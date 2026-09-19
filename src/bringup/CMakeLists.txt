cmake_minimum_required(VERSION 3.8)
project(bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/check_contract.py scripts/smoke_task.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
