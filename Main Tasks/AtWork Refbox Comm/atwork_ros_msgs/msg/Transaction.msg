# A unique identifier (per robot) of the transaction. Transactions in the
# log must start with the transaction_id 1 and must increase the
# transaction_id by 1 for each further transaction.
std_msgs/UInt64 transaction_id

# The unique ID of one of the active orders to which this transaction is
# associated.
std_msgs/UInt64 order_id

# A class or instance description of the object which is modified by the
# transaction.
atwork_ros_msgs/ObjectIdentifier object

# This parameter is only required for object classes (i.e. the instance_id
# is not set in the ObjectIdentifier). In other cases it is ignored.
std_msgs/UInt32 quantity

uint8 INSERT = 1
uint8 REMOVE = 2
uint8 MOVE = 3
# INSERT instantiates new objects in the inventory. Use this action e.g.
# to add a new, external delivery to the inventory. Alternatively, the
# INSERT action (after REMOVE actions) is also used to change the class
# of an object, for instance, after assembly.

# REMOVE deletes existing objects from the inventory. Use this action
# e.g. to remove objects which have been shipped. Alternatively, REMOVE
# actions (before INSERT actions) are also used to change the class of an
# object, for instance, after assembly.

# MOVE modifies the location of the specified objects. Use this action
# e.g. to describe that a robot grasped or placed an object.
# The action to be performed
std_msgs/UInt64 action

# The semantics of the source and destination parameter depend on the
# action:
# * MOVE(source, destination) i.e. source & destination required
# * INSERT(destination) i.e. destination required, source ignored
# * REMOVE(source) i.e. source required, destination ignored
atwork_ros_msgs/LocationIdentifier source
atwork_ros_msgs/LocationIdentifier destination
