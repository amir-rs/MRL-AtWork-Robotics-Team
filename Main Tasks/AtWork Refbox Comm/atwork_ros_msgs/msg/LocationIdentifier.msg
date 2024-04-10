# Location types
uint8    SH    = 1  # Shelf
uint8    WS    = 2  # Workstation
uint8    CB    = 3  # Conveyor belt
uint8    WP    = 4  # Way Point (additional navigation goal)
uint8    PP    = 5  # Precision Placement platform
uint8    ROBOT = 6

# The type of location
std_msgs/UInt64 type

# The instance ID of the location (e.g. if there are several shelves)
std_msgs/UInt64  instance_id

# A human-readable description of the location
std_msgs/String  description
