from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
from casadi import SX, horzcat, vertcat

def export_rotational_ode_model() -> AcadosModel:

    model_name = 'rotational_ode'

    # set up states & controls
    inertia_tensor = SX.sym('inertia_tensor', 3, 3)
    inertia_tensor_derivative = SX.sym('inertia_tensor_derivative', 3, 3)
    quaternion = SX.sym('quaternion', 4, 1)
    angular_velocity = SX.sym('angular_velocity', 3, 1)

    x = vertcat(inertia_tensor.reshape((9, 1)), inertia_tensor_derivative.reshape((9, 1)), quaternion, angular_velocity)

    tau = SX.sym('tau', 3, 1)
    u = vertcat(tau)

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

    f_expl = vertcat(inertia_tensor_derivative.reshape((9, 1)), SX.zeros(3, 3).reshape((9, 1)) , quaternion_derivative, (tau - inertia_tensor_derivative @ angular_velocity - omega_skew @ inertia_tensor @ angular_velocity))
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
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
    N_sim = 200

    # set simulation time
    sim.solver_options.T = Tf
    # set options
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages = 3
    sim.solver_options.num_steps = 3
    sim.solver_options.newton_iter = 3 # for implicit integrator
    sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

    # create
    acados_integrator = AcadosSimSolver(sim)

    return acados_integrator