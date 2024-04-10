# phases of benchmarks
uint8 EXECUTION   = 0
uint8 CALIBRATION = 1
uint8 PREPARATION = 2
# Current benchmark phase to terminate
std_msgs/UInt64 phase_to_terminate

# FBM1
std_msgs/String object_class_name

# FBM1+FMB2
std_msgs/String object_instance_name

# FBM2
std_msgs/Bool grasp_notification
