# types of tasks
uint8 UNKNOWN        = 0 #
uint8 TRANSPORTATION = 1 # For Pick & Place type Tasks
uint8 NAVIGATION     = 2 # For BNT

# types of status
uint8 OFFERED     =  1  # The order is offered by the RefBox and can be processed by a team/robot
uint8 TIMEOUT     =  2  # The order has not been accepted in time or processing took too long
uint8 IN_PROGRESS =  3  # A team/robot is currently processing this order
uint8 PAUSED      =  4  # Temporarily pause processing this order
uint8 ABORTED     =  5  # Stop working on this order
uint8 FINISHED    =  6  # The order has been processed successfully

# A unique identifier of this order
std_msgs/UInt64 id
# The Type of Task
std_msgs/UInt64 type
# The status of this order as specified in the Status enumeration
std_msgs/UInt64 status

# One of these should be set, according to type
atwork_ros_msgs/TransportationTask transportation_task
atwork_ros_msgs/NavigationTask     navigation_task
