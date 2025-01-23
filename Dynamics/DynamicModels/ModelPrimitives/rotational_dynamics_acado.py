from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
from casadi import SX, horzcat, vertcat, inv
import numpy as np
import quaternion

from Dynamics.DynamicModels.ModelPrimitives import RotationalState, RotationalControl

def export_rotational_ode_model() -> AcadosModel:

    model_name = 'rotational_ode'

    # set up states & controls
    inertia_tensor = SX.sym('inertia_tensor', 3, 3)
    inertia_tensor_derivative = SX.sym('inertia_tensor_derivative', 3, 3)
    quaternion = SX.sym('quaternion', 4, 1)
    angular_velocity = SX.sym('angular_velocity', 3, 1)

    torque_limits = SX.sym('torque_limits', 3, 2)

    x = vertcat(inertia_tensor.reshape((9, 1)), inertia_tensor_derivative.reshape((9, 1)), quaternion, angular_velocity)

    torque_control = SX.sym('u', 3, 1)
    u = vertcat(torque_control)
    
    tau = 0.5*((torque_limits[:, 0]+torque_limits[:, 1]) + ((torque_limits[:, 0]-torque_limits[:, 1]) * torque_control))

    # xdot
    inertia_tensor_dot = SX.sym('inertia_tensor_dot', 9 , 1)
    inertia_tensor_derivative_dot = SX.sym('inertia_tensor_derivative_dot', 9, 1)
    quaternion_dot = SX.sym('quaternion_dot', 4, 1)
    angular_velocity_dot = SX.sym('angular_velocity_dot', 3, 1)

    xdot = vertcat(inertia_tensor_dot, inertia_tensor_derivative_dot, quaternion_dot, angular_velocity_dot)

    # dynamics
    
    omega_skew = SX.zeros(3, 3)
    omega_skew[0, 1] = -angular_velocity[2]
    omega_skew[0, 2] = angular_velocity[1]
    omega_skew[1, 0] = angular_velocity[2]
    omega_skew[1, 2] = -angular_velocity[0]
    omega_skew[2, 0] = -angular_velocity[1]
    omega_skew[2, 1] = angular_velocity[0]

    # Omega matrix as a block matrix
    omega_matrix = vertcat(
        horzcat(SX.zeros(1, 1), -angular_velocity.T),
        horzcat(angular_velocity, -omega_skew)
    )

    quaternion_derivative = 0.5 * omega_matrix @ quaternion

    f_expl = vertcat(inertia_tensor_derivative.reshape((9, 1)), SX.zeros(3, 3).reshape((9, 1)) , quaternion_derivative, inv(inertia_tensor) @ (tau - inertia_tensor_derivative @ angular_velocity - omega_skew @ inertia_tensor @ angular_velocity))
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = torque_limits
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model

def GetIntegrator():
    rotational_model = export_rotational_ode_model()
    
    sim = AcadosSim()
    sim.model = rotational_model

    Tf = 0.1
    nx = sim.model.x.rows()

    # set simulation time
    sim.solver_options.T = Tf
    # set options
    sim.solver_options.integrator_type = 'ERK'
    # sim.solver_options.num_stages = 3
    # sim.solver_options.num_steps = 3
    # sim.solver_options.newton_iter = 3 # for implicit integrator
    # sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

    # create
    acados_integrator = AcadosSimSolver(sim)

    return acados_integrator

def GetSimulatedRotationalState(sim_solver: AcadosSimSolver, estimated_rotational_state: RotationalState, control: RotationalControl, dt):
    
    sim_solver.set("T", dt)
    new_state = sim_solver.simulate(estimated_rotational_state.ToList(), control.ToList(), dt)

    inertia_tensor, inertia_tensor_derivative, quat_list, angular_velocity = new_state[0:9], new_state[9:18], new_state[18:22], new_state[22:25]

    inertia_tensor = inertia_tensor.reshape(3, 3)
    inertia_tensor_derivative = inertia_tensor_derivative.reshape(3, 3)
    quat = np.quaternion(quat_list[0], quat_list[1], quat_list[2], quat_list[3])

    new_rotational_state = RotationalState(inertia_tensor=inertia_tensor, inertia_tensor_derivative=inertia_tensor_derivative, quaternion=quat, angular_velocity=angular_velocity)

    return new_rotational_state
