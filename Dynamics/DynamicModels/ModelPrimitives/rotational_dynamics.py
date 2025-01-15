import numpy as np
import quaternion

from ...Integrators import RK4
from ..dynamic_model import DynamicModel, ModelState, ModelControl
from .quaternion_dynamics import QuaternionDynamics

class RotationalState(ModelState):
    def __init__(self, inertia_tensor, inertia_tensor_derivative, quaternion, angular_velocity):
        self.inertia_tensor = inertia_tensor
        self.inertia_tensor_derivative = inertia_tensor_derivative # This is not a differential state, assuming that the rocket is rigid; it's purely 'intermediate' given this operational constraint.
        self.quaternion = quaternion
        self.angular_velocity = angular_velocity

    def ToList(self):
        quat_list = np.array([self.quaternion.w, self.quaternion.x, self.quaternion.y, self.quaternion.z])
        return np.concatenate((self.inertia_tensor.flatten(), self.inertia_tensor_derivative.flatten(), quat_list.flatten(), self.angular_velocity.flatten()))
    
    def __add__(self, other):
        return RotationalState(self.inertia_tensor+other.inertia_tensor, self.inertia_tensor_derivative+other.inertia_tensor_derivative, self.quaternion+other.quaternion, self.angular_velocity+other.angular_velocity)
        
    def __mul__(self, alpha):
        return RotationalState(self.inertia_tensor*alpha, self.inertia_tensor_derivative*alpha, self.quaternion*alpha, self.angular_velocity*alpha)
        
    def __rmul__(self, alpha):
        return self.__mul__(alpha)  # Simply delegate to __mul__
    
class RotationalControl(ModelControl):
    def __init__(self, tau):
        self.tau = tau
    
    def ToList(self):
        return self.tau.flatten()
    
    def __add__(self, other):
        return RotationalControl(self.tau+other.tau)
        
    def __mul__(self, alpha):
        return RotationalControl(self.tau*alpha)
        
    def __rmul__(self, alpha):
        return self.__mul__(alpha)  # Simply delegate to __mul__

class RotationalDynamics(DynamicModel):

    def __init__(self, integrator):
        super().__init__(integrator)
        self.quaternion_dynamics = QuaternionDynamics(integrator)

    def Derivative(self, rotational_state: RotationalState, rotational_control: RotationalControl) -> RotationalState:
        angular_momentum = lambda inertia_tensor, angular_velocity: inertia_tensor @ angular_velocity
        angular_acceleration = lambda state, control: np.linalg.inv(state.inertia_tensor) @ (control.tau - state.inertia_tensor_derivative @ state.angular_velocity - np.cross(state.angular_velocity, angular_momentum(state.inertia_tensor, state.angular_velocity)))

        rotational_derivative = lambda state, control: RotationalState(state.inertia_tensor_derivative, np.zeros(np.shape(state.inertia_tensor_derivative)), self.quaternion_dynamics.Derivative(state.quaternion, state.angular_velocity, None), angular_acceleration(state, control))

        return rotational_derivative(rotational_state, rotational_control)

    def Update(self, state: RotationalState, control: RotationalControl, dt) -> RotationalState:
        rotational_derivative = lambda state, control: self.Derivative(state, control)
        new_rotation = RK4.Integrate(state, control, rotational_derivative, dt)

        new_rotation.quaternion = new_rotation.quaternion/new_rotation.quaternion.norm()

        return new_rotation