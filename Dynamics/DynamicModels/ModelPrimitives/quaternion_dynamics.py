import numpy as np
import quaternion

from ..dynamic_model import DynamicModel
from ...Integrators import RK4

class QuaternionDynamics(DynamicModel):

    def __init__(self, integrator):
        super().__init__(integrator)

    def ToList(self, quat):
        return [quat.w, quat.x, quat.y, quat.z]

    def Normalize(self, quat):
        return quat / np.linalg.norm(quat)

    def Derivative(self, state, control, model_params):
        quat = state
        angular_velocity = control

        omega_quat = np.quaternion(0, angular_velocity[0], angular_velocity[1], angular_velocity[2])
        quat_derivative = 0.5 * quat * omega_quat
        return quat_derivative
    
    def Update(self, state, control, model_params, dt):

        quaternion_derivative = lambda state, control: self.Derivative(state, control)
        q_new = RK4.Integrate(state, control, quaternion_derivative, dt)

        self.quaternion = np.quaternion(*q_new)
        self.Normalize()