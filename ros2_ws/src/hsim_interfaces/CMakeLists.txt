cmake_minimum_required(VERSION 3.8)
project(hsim_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Skill.msg"
  "msg/SkillPlan.msg"
  "action/ExecutePlan.action"
  "action/Manipulate.action"
  "srv/SetBodyPose.srv"
  DEPENDENCIES geometry_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
