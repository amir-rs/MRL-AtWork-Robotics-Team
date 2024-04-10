# Time since benchmark start
std_msgs/Time benchmark_time

# States of benchmarks
uint8 RUNNING   = 1
uint8 PAUSED    = 2
uint8 FINISHED  = 3
uint8 STOPPED   = 4
# Current benchmark state
std_msgs/UInt64 state

# Phases of benchmarks
uint8 EXECUTION   = 0
uint8 CALIBRATION = 1
uint8 PREPARATION = 2
# Current benchmark phase
std_msgs/UInt64 phase

#Current benchmark scenario
atwork_ros_msgs/BenchmarkScenario scenario

# Name of all known teams
std_msgs/String[] known_teams

# Name of teams which are currently connected to the RefBox
std_msgs/String[] connected_teams
