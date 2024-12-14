import numpy as np
import quaternion

from ...Integrators import RK4
from ..dynamic_model import DynamicModel, ModelParams, ModelState
from .quaternion_dynamics import QuaternionDynamics

class RotationalParams(ModelParams):
    def __init__(self, inertia_tensor, inertia_tensor_derivative):
        self.inertia_tensor = inertia_tensor
        self.inertia_tensor_derivative = inertia_tensor_derivative

class RotationalState(ModelState):
    def __init__(self, quaternion, angular_velocity):
        self.quaternion = quaternion
        self.angular_velocity = angular_velocity
    
    def __add__(self, other):
        return RotationalState(self.quaternion+other.quaternion, self.angular_velocity+other.angular_velocity)
        
    def __mul__(self, alpha):
        return RotationalState(self.quaternion*alpha, self.angular_velocity*alpha)
        
    def __rmul__(self, alpha):
        return self.__mul__(alpha)  # Simply delegate to __mul__

class RotationalDynamics(DynamicModel):

    def __init__(self, integrator):
        super().__init__(integrator)
        self.quaternion_dynamics = QuaternionDynamics(integrator)

    def Derivative(self, state: RotationalState, control, params: RotationalParams):
        angular_momentum = lambda angular_velocity: params.inertia_tensor @ angular_velocity

        angular_acceleration = lambda state, control: np.linalg.inv(params.inertia_tensor) @ (control - params.inertia_tensor_derivative @ state.angular_velocity - np.cross(state.angular_velocity, angular_momentum(state.angular_velocity)))
        quaternion_derivative = lambda state, control: RotationalState(self.quaternion_dynamics.Derivative(state.quaternion, state.angular_velocity, None), angular_acceleration(state, control))

        return quaternion_derivative(state, control)

    def Update(self, state: RotationalState, control, params: RotationalParams, dt):
        rotational_derivative = lambda state, control: self.Derivative(state, control, params)
        new_rotation = RK4.Integrate(state, control, rotational_derivative, dt)

        new_rotation.quaternion = new_rotation.quaternion/new_rotation.quaternion.norm()

        return new_rotation