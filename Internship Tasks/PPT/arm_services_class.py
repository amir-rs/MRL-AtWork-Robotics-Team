class Arm_services:
    def __init__(self) -> None:
        #    rospy.init_node("my_robot_controller")
        pass

    def f1(self):
        pass

    def f2(self):
        pass

    def f3(self):
        pass

    def f4(self):
        pass

    def f5(self):
        pass

    def execute_sequence(self, sequence):
        # code to execute a list of commands in sequence
        for command in sequence:
            if command["action"] == "forward":
                self.move_forward(command["distance"])
            elif command["action"] == "left":
                self.turn_left(command["angle"])
            elif command["action"] == "right":
                self.turn_right(command["angle"])
            elif command["action"] == "stop":
                self.stop()
            else:
                print("Invalid command:", command["action"])


if __name__ == "__main__":
    controller = Arm_services()
    controller.execute_sequence(
        [
            {"action": "forward", "distance": 10},
            {"action": "left", "angle": 90},
            {"action": "forward", "distance": 5},
            {"action": "right", "angle": 45},
            {"action": "stop"},
        ]
    )
