# orientations 
uint8 NORTH = 1
uint8 EAST  = 2
uint8 SOUTH = 3
uint8 WEST  = 4

# The location to where the robot must navigate to.
atwork_ros_msgs/LocationIdentifier location
# The orientation the robot should face at location. 
std_msgs/UInt64 orientation
# The time the robot must wait at once at the location.
std_msgs/Time wait_time
