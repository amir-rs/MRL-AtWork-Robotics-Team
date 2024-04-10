# Object types
uint8   F20_20_B      = 1  # Small Aluminium Profile (Black)
uint8   F20_20_G      = 2  # Small Aluminium Profile (Grey)
uint8   S40_40_B      = 3  # Big Aluminium Profile (Black)
uint8   S40_40_G      = 4  # Big Aluminium Profile (Grey)
uint8   M20_100       = 5  # Screw (Bolt)
uint8   M20           = 6  # Small Nut
uint8   M30           = 7  # Large Nut
uint8   R20           = 8  # Plastic Tube
uint8   BEARING_BOX   = 9  #
uint8   BEARING       = 10 #
uint8   AXIS          = 11 #
uint8   DISTANCE_TUBE = 12 #
uint8   MOTOR         = 13 #
uint8   CONTAINER_B   = 14 # A Container (Blue)
uint8   CONTAINER_R   = 15 # A Container (Red)

# The object class (e.g. "EM-01") consists of the type ("EM") and a type ID (1).
# The type of the object
std_msgs/UInt64 type

# The type ID of the object
std_msgs/UInt64 type_id

# All labeled objects also have an instance ID. For example, there could
# exist two instances of EM-01 containers, called "EM-01-001" and
# "EM-01-002" where "EM-01" represents the item's class and "001"/"002" are
# the instance IDs.
std_msgs/UInt64 instance_id

# A human-readable description of the identifier
std_msgs/String description
