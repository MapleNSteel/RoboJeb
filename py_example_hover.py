import numpy as np
import quaternion
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

from Controllers.HoverControllers import HoverPIDController
from Dynamics.Integrators import RK4
from Dynamics.DynamicModels.ModelPrimitives import RotationalDynamics, RotationalParams, RotationalState
from KRPCInterface import KRPCInterface

class MinimalPublisher(Node):
    def __init__(self, node_name, topic_name):
        super().__init__(node_name)
        self.publisher_ = self.create_publisher(Float64MultiArray, topic_name, 10)

    def PublisherCallback(self, array):
        msg = Float64MultiArray()
        msg.data = array
        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)

def QuaternionToList(quat):
    return [quat.w, quat.x, quat.y, quat.z]

def RotationalStateToList(estimated_attitude):
    quat = estimated_attitude.quaternion
    return QuaternionToList(quat)

def main():
    rclpy.init()

    minimal_publisher_1 = MinimalPublisher('random_telemetry', 'topic')
    
    krpc_interface = KRPCInterface()

    target_altitude_ground = 20
    Kp, Ki, Kd = 8e-3, 6e-4, 4e-2

    hover_pid_controller = HoverPIDController(krpc_interface, target_altitude_ground, Kp, Ki, Kd)

    active_vessel = krpc_interface.GetActiveVessel()

    start = time.time()
    initial_attitude =  active_vessel.GetCOMRotation()
    initial_angular_velocity_body_frame = active_vessel.GetCOMAngularVelocityBodyFrame()

    initial_rotation_state = RotationalState(initial_attitude, initial_angular_velocity_body_frame)
    estimated_rotational_state = initial_rotation_state

    rotational_dynamics = RotationalDynamics(RK4.Integrate)

    while True:
        if active_vessel.GetAvailableThrust() == 0:
            break

        altitude, radial_velocity, altitude_error, altitude_speed_error, throttle_command = hover_pid_controller.GetControlInput()
        active_vessel.SetThrottleControl(throttle_command)
        
        current_time = time.time()
        # dt = (current_time - start)

        start = current_time

        # print(f"dt: {dt}")

        print(f"burn time: {active_vessel.GetBurnTime()}")

        minimal_publisher_1.PublisherCallback([altitude, radial_velocity, altitude_error, altitude_speed_error, throttle_command])

    minimal_publisher_1.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()
