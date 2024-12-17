import numpy as np
import quaternion
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

from Dynamics.Integrators import RK4
from Dynamics.DynamicModels.ModelPrimitives import RotationalDynamics, RotationalState, RotationalControl
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

    minimal_publisher_2 = MinimalPublisher('attitude', 'attitude')
    minimal_publisher_3 = MinimalPublisher('angular_velocity', 'angular_velocity')
    minimal_publisher_4 = MinimalPublisher('estimated_attitude', 'estimated_anttitude')
    
    krpc_interface = KRPCInterface()

    active_vessel = krpc_interface.GetActiveVessel()

    start = time.time()

    initial_inertia_tensor = active_vessel.GetInertiaTensor()
    initial_inertia_tensor_derivative = active_vessel.GetInertiaTensorDerivative()
    initial_attitude =  active_vessel.GetCOMRotation()
    initial_angular_velocity_body_frame = active_vessel.GetCOMAngularVelocityBodyFrame()

    initial_rotation_state = RotationalState(initial_inertia_tensor, initial_inertia_tensor_derivative, initial_attitude, initial_angular_velocity_body_frame)
    estimated_rotational_state = initial_rotation_state

    minimal_publisher_2.PublisherCallback(RotationalStateToList(initial_rotation_state))
    minimal_publisher_3.PublisherCallback(list(initial_angular_velocity_body_frame))
    minimal_publisher_4.PublisherCallback(RotationalStateToList(estimated_rotational_state))

    rotational_dynamics = RotationalDynamics(RK4.Integrate)

    while True:
        if active_vessel.GetAvailableThrust() == 0:
            break

        current_time = time.time()
        dt = (current_time - start)
        
        inertia_tensor = active_vessel.GetInertiaTensor()
        inertia_tensor_derivative = active_vessel.GetInertiaTensorDerivative()
        attitude = active_vessel.GetCOMRotation()
        angular_velocity_body_frame = active_vessel.GetCOMAngularVelocityBodyFrame()

        tau = np.zeros(np.shape(angular_velocity_body_frame))

        start = current_time

        rotational_state = RotationalState(inertia_tensor, inertia_tensor_derivative, attitude, angular_velocity_body_frame)
        rotational_control = RotationalControl(tau)

        estimated_rotational_state = rotational_dynamics.Update(rotational_state, rotational_control, dt)

        minimal_publisher_2.PublisherCallback(RotationalStateToList(rotational_state))
        minimal_publisher_3.PublisherCallback(list(angular_velocity_body_frame))
        minimal_publisher_4.PublisherCallback(RotationalStateToList(estimated_rotational_state))

    minimal_publisher_2.destroy_node()
    minimal_publisher_3.destroy_node()
    minimal_publisher_4.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()
