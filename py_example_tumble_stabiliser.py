import numpy as np
import quaternion
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

from Controllers.TumbleStabiliser import TumbleStabiliser
from Dynamics.DynamicModels.ModelPrimitives import RotationalDynamicsAcado, RotationalState, RotationalControl
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
    return np.array([quat.w, quat.x, quat.y, quat.z])

def RotationalStateToList(estimated_attitude):
    quat = estimated_attitude.quaternion
    return QuaternionToList(quat)

def main():
    rclpy.init()

    minimal_publisher_1 = MinimalPublisher('attitude', 'attitude')
    minimal_publisher_2 = MinimalPublisher('angular_velocity', 'angular_velocity')
    minimal_publisher_3 = MinimalPublisher('torque', 'torque')
    
    krpc_interface = KRPCInterface()

    active_vessel = krpc_interface.GetActiveVessel()

    start = time.time()

    initial_inertia_tensor = active_vessel.GetInertiaTensor()
    initial_inertia_tensor_derivative = active_vessel.GetInertiaTensorDerivative()
    initial_attitude =  active_vessel.GetCOMRotation()
    initial_angular_velocity_body_frame = active_vessel.GetCOMAngularVelocityBodyFrame()

    initial_rotation_state = RotationalState(initial_inertia_tensor, initial_inertia_tensor_derivative, initial_attitude, initial_angular_velocity_body_frame)

    minimal_publisher_1.PublisherCallback(list(RotationalStateToList(initial_rotation_state)))
    minimal_publisher_2.PublisherCallback(list(initial_rotation_state.angular_velocity))

    angular_momentum = initial_inertia_tensor @ initial_angular_velocity_body_frame
    
    tau_limits = active_vessel.GetAvailableTorque()
    N_horizon = 40 
    Tf = 20
    RTI = True

    tumble_stabiliser = TumbleStabiliser(initial_rotation_state.ToList(), tau_limits, N_horizon, Tf, RTI)

    while True:
        if active_vessel.GetAvailableThrust() == 0:
            break

        current_time = time.time()
        dt = (current_time - start)

        if dt < 1/60.:
            continue

        print(dt)
        
        inertia_tensor_body_frame = active_vessel.GetInertiaTensor()
        # inertia_tensor_derivative_body_frame = active_vessel.GetInertiaTensorDerivative()
        inertia_tensor_derivative_body_frame = inertia_tensor_body_frame*0
        attitude = active_vessel.GetCOMRotation()
        angular_velocity_body_frame = active_vessel.GetCOMAngularVelocityBodyFrame()

        # print(f"inertia_tensor_body_frame @ angular_velocity_body_frame: {inertia_tensor_body_frame @ angular_velocity_body_frame}")
        estimated_tau = (inertia_tensor_body_frame @ angular_velocity_body_frame - angular_momentum)/dt
        # print(f"angular_momentum: {angular_momentum}")
        print(f"estimated_tau: {estimated_tau}")
        angular_momentum = inertia_tensor_body_frame @ angular_velocity_body_frame

        start = current_time

        rotational_state = RotationalState(inertia_tensor_body_frame, inertia_tensor_derivative_body_frame, attitude, angular_velocity_body_frame)
        rotational_control = tumble_stabiliser.GetControlInput(rotational_state.ToList())

        print(tau_limits)

        tau = rotational_control.tau
        # control = np.array((2*tau-(np.array(tau_limits[0])+np.array(tau_limits[1])))/(np.array(tau_limits[0])-np.array(tau_limits[1])))
        control = tau
        control[0], control[1], control[2] = -control[1], -control[0], -control[2] # vessel coordinates are in left-hand coordinate system, but control frame is in orbital frame, which is right-hand (may only be in case of space)

        print(tau)
        print(control)
        active_vessel.SetAttitudeControl(control)

        time.sleep(1/60.)

        print(f"angular_velocity_body_frame: {angular_velocity_body_frame}\n")

        minimal_publisher_1.PublisherCallback(list(RotationalStateToList(rotational_state)))
        minimal_publisher_2.PublisherCallback(list(angular_velocity_body_frame))
        minimal_publisher_3.PublisherCallback(list(control))

    minimal_publisher_1.destroy_node()
    minimal_publisher_2.destroy_node()
    minimal_publisher_3.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":

    main()
