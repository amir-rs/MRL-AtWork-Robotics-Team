cmake_minimum_required(VERSION 2.8.7)
project(atwork_pb_msgs)
find_package(catkin REQUIRED)

find_package(Protobuf REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIRS})

set(CFH_ROOT_DIR ${PROJECT_SOURCE_DIR}/../atwork_refbox)
set(ATWORK_MSGS_DIR "${CFH_ROOT_DIR}/atwork/msgs")

IF(EXISTS "${ATWORK_MSGS_DIR}" AND IS_DIRECTORY "${ATWORK_MSGS_DIR}")
  MESSAGE(STATUS "protobuf msgs directory exists")
ELSE()
  MESSAGE(FATAL_ERROR "\n"
     "Protobuf msgs directory doesn't exist.\n"
     "Check if the submodule has been initialized.\n"
     "Please see README in \"atwork_refbox_comm\"\n"
  )
ENDIF(EXISTS "${ATWORK_MSGS_DIR}" AND IS_DIRECTORY "${ATWORK_MSGS_DIR}")

file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}/${PROJECT_NAME} )

catkin_package(
  LIBRARIES
    atwork_pb_msgs
  INCLUDE_DIRS ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}
)

file(GLOB CFH_PROTOS ${CFH_ROOT_DIR}/atwork/msgs/*.proto)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CFH_PROTOS})
add_custom_target(copy ALL
  DEPENDS ${PROTO_HDRS}
)


add_library(atwork_pb_msgs SHARED ${PROTO_SRCS})
target_link_libraries(atwork_pb_msgs ${PROTOBUF_LIBRARIES})




foreach(File ${PROTO_HDRS})
  add_custom_command(TARGET copy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${File} ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}/${PROJECT_NAME} 
  )
endforeach()

install(FILES ${PROTO_HDRS}
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(TARGETS atwork_pb_msgs
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## else: using cmake
#else()
#  install(
#    TARGETS atwork_pb_msgs
#    LIBRARY DESTINATION lib
#  )
#  install(
#    FILES ${PROTO_HDRS}
#    DESTINATION include/${PROJECT_NAME}
#  )
#
#endif(DEFINED _CATKIN_ALL_INCLUDED_)

