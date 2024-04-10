cmake_minimum_required(VERSION 2.8.7)
project(atwork_ros_msgs)

find_package(catkin REQUIRED
  COMPONENTS
    std_msgs
    geometry_msgs
    message_generation
)

add_message_files(
  DIRECTORY msg
  FILES
    AttentionMessage.msg
    BenchmarkState.msg
    BenchmarkFeedback.msg
    Inventory.msg
    Item.msg
    LocationIdentifier.msg
    ObjectIdentifier.msg
    NavigationTask.msg
    TransportationTask.msg
    Task.msg
    TaskInfo.msg
    TriggeredConveyorBeltStatus.msg
    TriggeredConveyorBeltCommand.msg
    BenchmarkScenario.msg
    LoggingStatus.msg
    Transaction.msg
    RobotStatus.msg
    RobotStatusReport.msg
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
)


install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

