from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from Dynamics.DynamicModels.ModelPrimitives import RotationalDynamicsAcado, RotationalState, RotationalControl
import numpy as np
import scipy.linalg
from casadi import vertcat

class TumbleStabiliser:
    def __init__(self, x0, tau_limits, N_horizon, Tf, RTI=False):
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # set model
        model = RotationalDynamicsAcado.export_rotational_ode_model()
        ocp.model = model

        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = nx


        # set cost module
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        Q_mat = 2*np.diag([0] * 22 + [1e5, 1e5, 1e5])
        R_mat = 2*np.diag([1e-7, 1e-7, 1e-7])

        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.W_e = Q_mat

        ocp.model.cost_y_expr = vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x
        ocp.cost.yref  = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        self.tau_limits = tau_limits

        # set constraints
        ocp.constraints.lbu = np.array(tau_limits[1])
        ocp.constraints.ubu = np.array(tau_limits[0])
        ocp.constraints.idxbu = np.arange(nu)

        ocp.constraints.x0 = x0

        # set prediction horizon
        ocp.solver_options.N_horizon = N_horizon
        ocp.solver_options.tf = Tf

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.sim_method_newton_iter = 10

        if RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'
            ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
            ocp.solver_options.nlp_solver_max_iter = 150

        self.RTI = RTI

        ocp.solver_options.qp_solver_cond_N = N_horizon

        solver_json = 'acados_ocp_' + model.name + '.json'
        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

        # do some initial iterations to start with a good initial guess
        num_iter_initial = 5
        for _ in range(num_iter_initial):
            self.acados_ocp_solver.solve_for_x0(x0_bar = x0)

        # create an integrator with the same settings as used in the OCP solver.
        self.acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    def GetControlInput(self, rotational_state: RotationalState):
        if self.RTI:
            rotational_state_list = rotational_state
            # preparation phase
            self.acados_ocp_solver.options_set('rti_phase', 1)
            status = self.acados_ocp_solver.solve()
            print(status)

            # set initial state
            self.acados_ocp_solver.set(0, "lbx", rotational_state_list)
            self.acados_ocp_solver.set(0, "ubx", rotational_state_list)

            # feedback phase
            self.acados_ocp_solver.options_set('rti_phase', 2)
            status = self.acados_ocp_solver.solve()
            print(status)

            torque = self.acados_ocp_solver.get(0, "u")

        else:
            # solve ocp and get next control input
            torque = self.acados_ocp_solver.solve_for_x0(x0_bar = rotational_state)

        return RotationalControl(torque)