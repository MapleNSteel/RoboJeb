import numpy as np
import quaternion
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

from Controllers.HoverControllers import HoverPIDController
from Dynamics.Integrators import RK4
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

def QuaternionToList(quaternion):
    return [quaternion.w, quaternion.x, quaternion.y, quaternion.z]

def NormalizeQuaternion(q):
    return q / np.linalg.norm(q)

def QuaternionDerivative(q, angular_velocity):
    omega_quat = np.quaternion(0, angular_velocity[0], angular_velocity[1], angular_velocity[2])

    quat_derivative = 0.5 * q * omega_quat
    return np.array([quat_derivative.w, quat_derivative.x, quat_derivative.y, quat_derivative.z])

def UpdateRotation(q, angular_velocity, inertia_tensor, inertia_tensor_derivative, angular_momentum, tau, dt):
    q = np.array(QuaternionToList(q)).reshape(4,)

    state = np.hstack([q, angular_velocity])

    angular_acceleration = lambda state: np.linalg.inv(inertia_tensor) @ (tau - inertia_tensor_derivative @ state[4:] - np.cross(state[4:], angular_momentum))
    quaternion_derivative = lambda state: np.hstack([QuaternionDerivative(np.quaternion(*state[0:4]), state[4:]), angular_acceleration(state)])
    q_new = RK4.Integrate(state, quaternion_derivative, dt)

    return np.quaternion(*NormalizeQuaternion(q_new[0:4]))


def main():
    rclpy.init()

    minimal_publisher_1 = MinimalPublisher('random_telemetry', 'topic')
    minimal_publisher_2 = MinimalPublisher('attitude', 'attitude')
    minimal_publisher_3 = MinimalPublisher('angular_velocity', 'angular_velocity')
    minimal_publisher_4 = MinimalPublisher('estimated_attitude', 'estimated_anttitude')
    minimal_publisher_5 = MinimalPublisher('relative_quaternion', 'relative_quaternion')
    
    krpc_interface = KRPCInterface()

    target_altitude_ground = 20
    Kp, Ki, Kd = 8e-3, 6e-4, 4e-2

    hover_pid_controller = HoverPIDController(krpc_interface, target_altitude_ground, Kp, Ki, Kd)

    active_vessel = krpc_interface.GetActiveVessel()

    start = time.time()
    initial_attitude =  active_vessel.GetCOMRotation()
    estimated_attitude = initial_attitude

    minimal_publisher_2.PublisherCallback(QuaternionToList(initial_attitude))
    minimal_publisher_4.PublisherCallback(QuaternionToList(estimated_attitude))

    while True:
        if active_vessel.GetAvailableThrust() == 0:
            break

        # altitude, radial_velocity, altitude_error, altitude_speed_error, throttle_command = hover_pid_controller.GetControlInput()
        # active_vessel.SetThrottleControl(0*throttle_command)
        # active_vessel.SetAttitudeControl([1.0, 0, 0])
        
        current_time = time.time()
        dt = (current_time - start)
        
        attitude = active_vessel.GetCOMRotation()

        angular_velocity_body_frame = active_vessel.GetCOMAngularVelocityBodyFrame()

        tau = 0*angular_velocity_body_frame
        inertia_tensor = active_vessel.GetInertiaTensor()
        inertia_tensor_derivative = active_vessel.GetInertiaTensorDerivative()

        angular_momentum = inertia_tensor @ angular_velocity_body_frame

        start = current_time

        # print(f"dt: {dt}")

        estimated_attitude = UpdateRotation(estimated_attitude, angular_velocity_body_frame, inertia_tensor, inertia_tensor_derivative, angular_momentum, tau, dt)

        # print(f"attitude: {attitude}")
        # print(f"angular_velocity: {angular_velocity_body_frame}")
        # print(f"estimated_attitude: {estimated_attitude}")

        # print(attitude)

        # print(theta)
        # print(axis)

        # exit(0)

        # minimal_publisher_1.PublisherCallback([altitude, radial_velocity, altitude_error, throttle_command])
        minimal_publisher_2.PublisherCallback(QuaternionToList(attitude))
        minimal_publisher_4.PublisherCallback(QuaternionToList(estimated_attitude))
        minimal_publisher_3.PublisherCallback(list(angular_velocity_body_frame))

        # # print(f"altitude: {altitude}, radial_velocity: {radial_velocity}, "
        #     #   f"altitude_error: {altitude_error}, altitude_speed_error: {altitude_speed_error}, throttle_command: {throttle_command}, surface_gravity: {krpc_interface.surface_gravity}, ")
        # # print(f"available thrust: {active_vessel.GetAvailableThrust()}, available_torque: {active_vessel.GetAvailableTorque()}, "
        # print(f"position: {active_vessel.GetCOMPosition()}, rotation: {active_vessel.GetCOMRotation()}, mass: {active_vessel.GetMass()}, inertia tensor: {active_vessel.GetInertiaTensor()}")

        # # for active_fuel_part in active_fuel_parts:
        # #     print(f"active_fuel_part.name: {active_fuel_part.name}")

        # print(f"burn time: {active_vessel.GetBurnTime()}")

        # print(f"inertia_tensor: {inertia_tensor}\n")
        # print(f"inertia_tensor_derivative: {inertia_tensor_derivative}")

    minimal_publisher_1.destroy_node()
    minimal_publisher_2.destroy_node()
    minimal_publisher_3.destroy_node()
    minimal_publisher_4.destroy_node()
    minimal_publisher_5.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()
