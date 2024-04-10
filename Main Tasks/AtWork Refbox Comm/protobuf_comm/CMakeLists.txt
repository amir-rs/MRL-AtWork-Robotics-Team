cmake_minimum_required(VERSION 2.8.7)
project(protobuf_comm)

find_package(catkin REQUIRED)
find_package(Boost 1.46.1 REQUIRED thread system)
find_package(OpenSSL REQUIRED)


set(CFH_ROOT_DIR ${PROJECT_SOURCE_DIR}/../atwork_refbox)
set(PROTOBUF_COMM_SRC_DIR ${CFH_ROOT_DIR}/src/libs/protobuf_comm)
set(PROTOBUF_COMM_INCLUDE_DIR ${CFH_ROOT_DIR}/src/libs)
set(CFH_LIBS_DIR ${CFH_ROOT_DIR}/src/libs)

#link_directories(${Boost_LIBRARY_DIRS})
if(EXISTS "${PROTOBUF_COMM_SRC_DIR}"
    AND IS_DIRECTORY "${PROTOBUF_COMM_SRC_DIR}")
  message(STATUS "protobuf_comm source directory found.")
else()
  message(FATAL_ERROR "\n"
      "protobuf_comm source directory not found."
      "Check if git submodule has been initialized."
      "See README in \"atwork_refbox_comm\"\n")
endif(EXISTS "${PROTOBUF_COMM_SRC_DIR}"
    AND IS_DIRECTORY "${PROTOBUF_COMM_SRC_DIR}")


add_definitions("-std=c++0x")
add_definitions("-DHAVE_LIBCRYPTO")

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${OPENSSL_INCLUDE_DIR}
  ${CFH_LIBS_DIR}
)


catkin_package(
  LIBRARIES
    protobuf_comm
  INCLUDE_DIRS ${PROTOBUF_COMM_INCLUDE_DIR}
)
file(GLOB PROTOBUF_COMM_SOURCES ${PROTOBUF_COMM_SRC_DIR}/*.cpp)
file(GLOB PROTOBUF_COMM_HEADERS ${PROTOBUF_COMM_SRC_DIR}/*.h)
add_library(protobuf_comm SHARED ${PROTOBUF_COMM_SOURCES})

target_link_libraries(protobuf_comm ${OPENSSL_LIBRARIES} ${Boost_LIBRARIES})


file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}/${PROJECT_NAME} )

add_custom_target(${PROJECT_NAME}_copy_headers ALL)

foreach(File ${PROTOBUF_COMM_HEADERS})
  add_custom_command(TARGET ${PROJECT_NAME}_copy_headers POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${File} ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}/${PROJECT_NAME}
  )
endforeach()

install(FILES ${PROTOBUF_COMM_HEADERS}
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
 
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
