# Conveyor belt states
uint8 STOP  = 0 # The conveyor belt is standing still
uint8 START = 1 # The conveyor belt is moving

# The current motion state of the conveyor belt.
std_msgs/UInt32 state

# The cycle determines how often the conveyor belt has already been
# activated. Every time that the conveyor belt is restarted by a robot, the
# cycle counter is increased.
std_msgs/UInt32 cycle