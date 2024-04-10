# @Work RefBox Communication
Catkin Meta Package for communicating with the @Work RefBox.

Provides:

1. **protobuf comm**: library used to communicate with the refbox.
2. **atwork pb msgs**: Official protobuf messages used to communicate with the refbox.
3. **atwork ros msgs**: ROS translations of the protobuf messages. Provided as example, not as official messages.

## Installation

    cd <catkin_workspace>/src
    git clone git@github.com:industrial-robotics/atwork_refbox_comm.git
    cd atwork_refbox_comm
    git submodule init
    git submodule update

## Run catkin make

    cd <catkin_workspace>
    catkin_make

