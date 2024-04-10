# at_work_refbox_ros_client
====================

Demo ros node for demonstrating the usage of protobuf messages from the atwork refbox inside ROS nodes.

## Node Dependancy
Depends on the following packages;
```
protobuf_comm
atwork_pb_msgs
atwork_ros_msgs
Protobuf
```
### protobuf_comm

This CMake project locates the protobuf_comm library from the atwork-refbox and builds it.
It is installed into your catkin workspace for ros packages to find and use.


### atwork_pb_msgs

This CMake project will locates the protobuf messages from the atwork-refbox submodule and builds them. 
They are then installed in your catkin workspace.

### atwork_ros_msgs

ros messages defined to communicate with other clients.
NOTE: This can be totally avoided if you can work with protobuf commands.

### Protobuf
Library required for communicating to Protobuf.


## Compilation and Installation

Before installation of this we need to install all the dependency.
### Officially Supported Setup: Ubuntu 14.04, 16.04, Boost 1.54


1. Add [Tim Niemueller's PPA](https://launchpad.net/~timn/+archive/ubuntu/clips):
      
        sudo add-apt-repository ppa:timn/clips
    (Note: This PPA currently only works for Ubuntu 12.04, 12.10, 14.04 and 16.04)
    
2. Install the dependencies for both LLSFRB and CFH (12.04, 12.10, 14.04):
        
        sudo apt-get update
        sudo apt-get install libmodbus-dev libclips-dev clips libclipsmm-dev \
                             protobuf-compiler libprotobuf-dev libprotoc-dev \
                             libmodbus-dev \
                             libglibmm-2.4-dev libgtkmm-3.0-dev libncurses5-dev \
                             libncursesw5-dev libyaml-cpp-dev libavahi-client-dev git \
                             libssl-dev libelf-dev mongodb-clients \
                             mongodb libzmq3-dev

     (If using 14.04 or older, use boost1.54 and mongodb-dev)

        sudo apt-get install mongodb-dev boost1.54-all-dev

     (If using 16.04, use default boost)

        sudo apt-get install scons libboost-all-dev

3. Clone the atwork_refbox_comm repository

        cd <catkin_workspace>/src
        git clone git@github.com:industrial-robotics/atwork_refbox_comm.git
        cd atwork_refbox_comm
        git submodule init
        git submodule update

4. Compilation 

        cd <catkin_workspace>
        catkin build


5. Compiling the refbox 
The above step also gets the refree box inside folder **atwork_refbox**.
We can compile the refree box and test it.

        cd atwork_refbox
        make 

6. Running refbox

        * Start the RefBox: ./bin/refbox
        * Start the controller: ./bin/atwork-controller
        * Start the viewer: ./bin/atwork-viewer

    

7. Cloning this repository for ros topics
    
        cd catkin workspace
        git clone https://github.com/industrial-robotics/atwork_refbox_ros_client
        cd atwork_refbox_ros_client
        catkin build --this

   
## Usage

```roslaunch atwork_refbox_ros_client robot_example_ros.launch```

## Testing 

TODO
