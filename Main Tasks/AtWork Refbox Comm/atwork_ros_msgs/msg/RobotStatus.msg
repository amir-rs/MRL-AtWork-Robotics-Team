# types of Capabilities 
uint8 UNKNOWN      =  0  #
uint8 NAVIGATION   =  1  # Navigation (e.g. planning or execution)
uint8 MANIPULATION =  2  # Manipulation (e.g. grasp planning/execution or arm motion planning/execution)
uint8 PERCEPTION   =  3  # Perception (e.g. RGB or depth image Processing)
uint8 TASK         =  4  # Task planning or task execution

# The currently active capability
std_msgs/UInt64 capability
# The functionality which provides the above capability
# (e.g. a softare component)
std_msgs/String functionality
# Arbitrary information to further describe the capability
# (e.g. error codes or detected objects)
std_msgs/String meta_data
