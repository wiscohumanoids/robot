cmake_minimum_required(VERSION 3.8)
project(humanoid_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Skill.msg"
  "msg/SkillSequence.msg"
  "msg/ObjectPose.msg"
  "msg/ObjectPoseArray.msg"
  "msg/PolicyObservation.msg"
  "msg/JointTargets.msg"
  "msg/JointCommand.msg"
  "msg/RobotState.msg"
  "msg/SafetyStatus.msg"
  "action/ExecuteManipulation.action"
  DEPENDENCIES std_msgs geometry_msgs
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
