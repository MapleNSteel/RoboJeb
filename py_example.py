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

def NormalizeQuaternion(q):
    return q / np.linalg.norm(q)

def QuaternionDerivative(q, angular_velocity):
    omega_quat = np.quaternion(0, angular_velocity[0], angular_velocity[1], angular_velocity[2])

    quat_derivative = 0.5 * q * omega_quat
    return np.array([quat_derivative.w, quat_derivative.x, quat_derivative.y, quat_derivative.z])

def UpdateRotation(q, angular_velocity, inertia_tensor, inertia_tensor_derivative, tau, dt):
    angular_momentum = lambda angular_velocity: inertia_tensor @ angular_velocity

    angular_acceleration = lambda state, control: np.linalg.inv(inertia_tensor) @ (control - inertia_tensor_derivative @ state[4:] - np.cross(state[4:], angular_momentum(state[4:])))
    quaternion_derivative = lambda state, control: np.hstack([QuaternionDerivative(np.quaternion(*state[0:4]), state[4:]), angular_acceleration(state, control)])

    q = np.array(QuaternionToList(q)).reshape(4,)
    state = np.hstack([q, angular_velocity])
    q_new = RK4.Integrate(state, tau, quaternion_derivative, dt)

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
    initial_angular_velocity_body_frame = active_vessel.GetCOMAngularVelocityBodyFrame()

    estimated_attitude = initial_attitude

    initial_rotation_state = RotationalState(initial_attitude, initial_angular_velocity_body_frame)
    estimated_rotational_state = initial_rotation_state

    minimal_publisher_2.PublisherCallback(RotationalStateToList(initial_rotation_state))
    minimal_publisher_4.PublisherCallback(RotationalStateToList(estimated_rotational_state))

    rotational_dynamics = RotationalDynamics(RK4.Integrate)

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

        start = current_time

        # print(f"dt: {dt}")

        estimated_attitude = UpdateRotation(estimated_attitude, angular_velocity_body_frame, inertia_tensor, inertia_tensor_derivative, tau, dt)
        rotational_state = RotationalState(estimated_rotational_state.quaternion, angular_velocity_body_frame)
        rotational_params = RotationalParams(inertia_tensor, inertia_tensor_derivative)
        estimated_rotational_state = rotational_dynamics.Update(rotational_state, tau, rotational_params, dt)

        rotational_state = RotationalState(attitude, angular_velocity_body_frame)

        # print(f"attitude: {attitude}")
        # print(f"angular_velocity: {angular_velocity_body_frame}")
        # print(f"estimated_rotational_state.quaternion: {estimated_rotational_state.quaternion}")

        # print(attitude)

        # print(theta)
        # print(axis)

        # exit(0)

        # minimal_publisher_1.PublisherCallback([altitude, radial_velocity, altitude_error, throttle_command])
        minimal_publisher_2.PublisherCallback(RotationalStateToList(rotational_state))
        minimal_publisher_4.PublisherCallback(RotationalStateToList(estimated_rotational_state))
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
