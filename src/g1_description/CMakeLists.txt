cmake_minimum_required(VERSION 3.8)
project(g1_description)

find_package(ament_cmake REQUIRED)

install(FILES
  g1_23dof.urdf.xacro
  g1_23dof_rev_1_0.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/images")
  install(DIRECTORY images
    DESTINATION share/${PROJECT_NAME}
  )
endif()

ament_package()
