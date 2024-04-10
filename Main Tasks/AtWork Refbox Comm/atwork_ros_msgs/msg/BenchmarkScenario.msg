# Types of benchmarks
uint8 NONE = 0 # No benchmark running
uint8 BNT  = 1 # Basic Navigation Test
uint8 BMT  = 2 # Basic Manipulation Test
uint8 BTT  = 3 # Basic Transportation Test
uint8 PPT  = 4 # Precision Placement Test
uint8 CBT  = 5 # Conveyor Belt Test
uint8 AWF  = 6 # At Work Final
uint8 IRL  = 7 # Industrial Robotics League Challenge

# e.g. "BTT 2: Basic Transportation Test instance #2"
std_msgs/UInt64 type
std_msgs/UInt32 type_id
std_msgs/String description
