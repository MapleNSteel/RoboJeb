import matplotlib.pyplot as plt
import numpy as np

from Controllers.TumbleStabiliser import TumbleStabiliser
def main(use_RTI=True):
    initial_rotation_state = np.array([
        1.43504762e+06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.12281555e+05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.44209525e+06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, -7.23439340e-01, 9.56159939e-02,
        6.81701350e-01, 5.26925592e-02, 2.91506211e-04,  1.03701703e+00,
        -4.33832358e-04
    ])

    tau_limits = ((39000.0, 39000.0, 39000.0), (-39000.0, -39000.0, -39000.0))

    Tf = 20
    N_horizon = 40

    hover_pid_controller = TumbleStabiliser(initial_rotation_state, tau_limits, N_horizon, Tf, use_RTI)
    ocp_solver, integrator = hover_pid_controller.acados_ocp_solver, hover_pid_controller.acados_integrator

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    Nsim = 40
    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim, nu))
    errU = np.zeros((Nsim, nu))

    simX[0,:] = initial_rotation_state

    if use_RTI:
        t_preparation = np.zeros((Nsim))
        t_feedback = np.zeros((Nsim))

    else:
        t = np.zeros((Nsim))

    # do some initial iterations to start with a good initial guess
    num_iter_initial = 5
    for _ in range(num_iter_initial):
        ocp_solver.solve_for_x0(x0_bar = initial_rotation_state)

    # closed loop
    for i in range(Nsim):

        if use_RTI:
            # preparation phase
            ocp_solver.options_set('rti_phase', 1)
            status = ocp_solver.solve()
            t_preparation[i] = ocp_solver.get_stats('time_tot')

            # set initial state
            ocp_solver.set(0, "lbx", simX[i, :])
            ocp_solver.set(0, "ubx", simX[i, :])

            # feedback phase
            ocp_solver.options_set('rti_phase', 2)
            status = ocp_solver.solve()
            t_feedback[i] = ocp_solver.get_stats('time_tot')

            simU[i, :] = ocp_solver.get(0, "u")
            import math
            
            errU[i, :] = hover_pid_controller.GetControlInput(simX[i, :]).tau-simU[i, :]
            print(simX[i, :])
            print(simU[i,:])
            print(errU[i, :])
            import pdb; pdb.set_trace()

        else:
            # solve ocp and get next control input
            simU[i, :] = ocp_solver.solve_for_x0(x0_bar = simX[i, :])

            t[i] = ocp_solver.get_stats('time_tot')

        # simulate system
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

    # evaluate timings
    if use_RTI:
        # scale to milliseconds
        t_preparation *= 1000
        t_feedback *= 1000
        print(f'Computation time in preparation phase in ms: \
                min {np.min(t_preparation):.3f} median {np.median(t_preparation):.3f} max {np.max(t_preparation):.3f}')
        print(f'Computation time in feedback phase in ms:    \
                min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}')
    else:
        # scale to milliseconds
        t *= 1000
        print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')

    # plot results
    model = ocp_solver.acados_ocp.model

    print(simX[:, 22:])
    print(simU)

    plt.plot(simX[:, 22], 'r')
    plt.plot(simX[:, 23], 'b')
    plt.plot(simX[:, 24], 'g')
    plt.savefig('plot.png')
    plt.clf()

    plt.plot(simU[:, 0], 'rx')
    plt.plot(simU[:, 1], 'gx')
    plt.plot(simU[:, 2], 'bx')
    plt.savefig('plot1.png')
    plt.clf()

    plt.plot(errU[:, 0], 'r.')
    plt.plot(errU[:, 1], 'g.')
    plt.plot(errU[:, 2], 'b.')
    plt.savefig('plot2.png')

    ocp_solver = None

if __name__ == "__main__":

    main()
