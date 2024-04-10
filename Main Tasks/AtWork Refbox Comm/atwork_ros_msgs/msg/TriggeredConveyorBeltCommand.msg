# Conveyor belt commands
uint8 STOP  = 0 # The conveyor belt is standing still
uint8 START = 1 # The conveyor belt is moving
# The desired motion state of the conveyor belt.
std_msgs/UInt32 command

# The next cycle for which the command is executed. The peer must set this
# field to exactly one greater than the cycle received via the
# TriggeredConveyorBeltStatus (next_cycle = cycle + 1).
std_msgs/UInt32 next_cycle