import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from Controllers.HoverControllers import HoverPIDController
from KRPCInterface import KRPCInterface

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, 'topic', 10)

    def PublisherCallback(self, array):
        msg = Float64MultiArray()
        msg.data = array
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

def main():
    rclpy.init()

    minimal_publisher = MinimalPublisher()
    
    krpc_interface = KRPCInterface()

    target_altitude_ground = 20
    Kp, Ki, Kd = 8e-3, 6e-4, 4e-2

    hover_pid_controller = HoverPIDController(krpc_interface, target_altitude_ground, Kp, Ki, Kd)

    active_vessel = krpc_interface.GetActiveVessel()

    while True:
        
        if active_vessel.GetAvailableThrust() == 0:
            break

        altitude, radial_velocity, altitude_error, altitude_speed_error, throttle_command = hover_pid_controller.GetControlInput()
        active_vessel.SetThrottleControl(throttle_command)

        minimal_publisher.PublisherCallback([altitude, radial_velocity, altitude_error, throttle_command])

        inertia_tensor_derivative = active_vessel.GetInertiaTensorDerivative()

        print(f"altitude: {altitude}, radial_velocity: {radial_velocity}, "
              f"altitude_error: {altitude_error}, altitude_speed_error: {altitude_speed_error}, throttle_command: {throttle_command}, surface_gravity: {krpc_interface.surface_gravity}, ")
        # print(f"available thrust: {active_vessel.GetAvailableThrust()}, available_torque: {active_vessel.GetAvailableTorque()}, "
        # print(f"position: {active_vessel.GetCOMPosition()}, rotation: {active_vessel.GetCOMRotation()}, mass: {active_vessel.GetMass()}, inertia tensor: {active_vessel.GetInertiaTensor()}")

        # for active_fuel_part in active_fuel_parts:
        #     print(f"active_fuel_part.name: {active_fuel_part.name}")

        print(f"burn time: {active_vessel.GetBurnTime()}")

        print(f"inertia_tensor_derivative: {inertia_tensor_derivative}")

    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
