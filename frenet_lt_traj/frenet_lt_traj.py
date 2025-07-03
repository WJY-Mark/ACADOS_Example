from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
import scipy.linalg
import os
import logging

import os
import sys
import shutil
import errno
import timeit

import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

import numpy as np
import scipy.linalg
from typing import List

import numpy as np
import casadi as ca
import math


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

THETA_UPPER_BOUND = 0.5*np.pi
KAPPA_UPPER_BOUND = 0.1
DKAPPA_UPPER_BOUND = 0.5
L_UPPER_BOUND = 2.0

THETA_LOWER_BOUND = -THETA_UPPER_BOUND
KAPPA_LOWER_BOUND = -KAPPA_UPPER_BOUND
DKAPPA_LOWER_BOUND = -DKAPPA_UPPER_BOUND
L_LOWER_BOUND = -L_UPPER_BOUND

L_INIT = 1.0
THETA_INIT = 0.0
KAPPA_INIT = -0.02

VEL = 10.0
KR = 0.01

ADD_C_D_CONSTRAINT = False
ADD_BOUND_CONSTRAINT = True


def export_simple_frenet_model():
    model_name = "frenet_lt"
    l = ca.SX.sym("l")
    delta_theta = ca.SX.sym("delta_theta")
    k = ca.SX.sym("k")
    x = ca.vertcat(l, delta_theta, k)

    # control
    dk = ca.SX.sym("dk")
    u = ca.vertcat(dk)

    # state
    l_dot = ca.SX.sym("l_dot")
    delta_theta_dot = ca.SX.sym("delta_theta_dot")
    k_dot = ca.SX.sym("k_dot")
    xdot = ca.vertcat(l_dot, delta_theta_dot, k_dot)

    vel = ca.SX.sym("vel")
    kr = ca.SX.sym("kr")

    p = ca.vertcat(vel, kr)

    f_expl = ca.vertcat(vel*delta_theta,
                        vel*(k-kr),
                        dk)
    f_impl = xdot - f_expl
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name
    z = ca.vertcat([])
    model.z = z

    return model


def output_debug_info_ocp(ocp):
    logger.debug('[Size of the Problem............]')
    logger.debug('state dimension of the problem: %d', ocp.dims.nx)
    logger.debug('control dimension of the problem: %d', ocp.dims.nu)
    logger.debug('dimension of the parameters: %d', ocp.dims.np)
    logger.debug('dimension of the lagrange term: %d', ocp.dims.ny)
    logger.debug('number of state bounds: %d', ocp.dims.nbx)
    logger.debug('number of control bounds: %d', ocp.dims.nbu)
    logger.debug('number of nonlinear constraints: %d', ocp.dims.nh)
    logger.debug('number of soft nonlinear constraints: %d', ocp.dims.nsh)
    logger.debug('total number of slacks: %d', ocp.dims.ns)
    logger.debug('cost type: %s', ocp.cost.cost_type)
    logger.debug('default coefficient for objectives: \n %s',
                 ocp.cost.W.__str__())
    logger.debug('init parameters: %s', ocp.model.p.__str__())

    logger.debug('[Solver general settings.............]')
    logger.debug('Stages numbers: %d', ocp.dims.N)
    logger.debug('horizon of the problem: %d', ocp.solver_options.tf)
    logger.debug('step size:%f', ocp.solver_options.tf/ocp.dims.N)

    logger.debug('qp solver type:%s', ocp.solver_options.qp_solver)
    logger.debug('nlp solver type:%s', ocp.solver_options.nlp_solver_type)
    logger.debug('hessian_approx method:%s', ocp.solver_options.hessian_approx)
    logger.debug('integrator_type:%s', ocp.solver_options.integrator_type)
    logger.debug('sim_method_num_stages:%d',
                 ocp.solver_options.sim_method_num_stages)
    logger.debug('sim_method_num_steps:%d',
                 ocp.solver_options.sim_method_num_steps)
    logger.debug('print_level:%d', ocp.solver_options.print_level)
    logger.debug('tollerance:%f', ocp.solver_options.tol)

    logger.debug('init state:%s', ocp.constraints.x0.__str__())


def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print("Error while removing directory {}".format(directory))


def set_acados_model(stage_n, tf):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    acados_models_dir = "./acados_models"
    safe_mkdir_recursive(os.path.join(os.getcwd(), acados_models_dir))
    acados_source_path = os.environ["ACADOS_SOURCE_DIR"]
    sys.path.insert(0, acados_source_path)
    # default state boundaries

    dk_lower = -0.5
    dk_upper = 0.5

    ocp = AcadosOcp()
    model = export_simple_frenet_model()
    ocp.model = model

    # set the dimension of the problem
    ocp.dims.N = stage_n
    ocp.dims.nx = ocp.model.x.size()[0]
    ocp.dims.nu = ocp.model.u.size()[0]
    ocp.dims.ny = ocp.dims.nx + ocp.dims.nu
    ocp.dims.ny_e = 0  # not set for now
    ocp.dims.nbx = ocp.dims.nx  # number of state bounds
    ocp.dims.nbu = ocp.dims.nu  # number of control bounds
    ocp.dims.np = ocp.model.p.size()[0]  # parameter size
    ocp.dims.nh = 0     # number of nonlinear constraints
    ocp.dims.nsh = 0    # number of soft nonlinear constraints
    ocp.dims.ns = 0     # total number of slacks
    ocp.dims.nsg = 0     # total number of slacks
    l_coeff = 100.0
    dtheta_coeff = 0.1
    k_coeff = 1.0
    dk_coeff = 10.0
    # cost functions and weights setting
    ocp.cost.cost_type = "LINEAR_LS"  # cost type, default format
    Q = np.eye(ocp.dims.nx)
    Q[0, 0] = l_coeff
    Q[1, 1] = dtheta_coeff
    Q[2, 2] = k_coeff
    R = np.eye(ocp.dims.nu)
    R[0, 0] = dk_coeff
    # set the cost function form
    Vx = np.zeros((ocp.dims.ny, ocp.dims.nx))
    Vx[:ocp.dims.nx, :ocp.dims.nx] = np.eye(ocp.dims.nx)
    Vu = np.zeros((ocp.dims.ny, ocp.dims.nu))
    Vu[ocp.dims.ny-1, 0] = 1

    ocp.cost.W_0 = scipy.linalg.block_diag(Q, R)
    ocp.cost.Vx_0 = Vx
    ocp.cost.Vu_0 = Vu
    ocp.cost.yref_0 = np.zeros((ocp.dims.ny, 1))

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.Vu = Vu
    ocp.cost.Vx = Vx
    ocp.cost.yref = np.zeros((ocp.dims.ny, 1))

    ocp.cost.Vx_e = np.eye(ocp.dims.nx)
    ocp.cost.W_e = Q
    ocp.cost.yref_e = np.zeros((ocp.dims.nx, 1))

    # # for slack variables
    # # L2 penenaty
    # ocp.cost.Zl = np.array([0])
    # ocp.cost.Zu = np.array([0])
    # # L1 penenaty
    # ocp.cost.zl = np.array([100])
    # ocp.cost.zu = np.array([100])

    # # path constraints, C , D matrix and corresponding lower and upper bounds
    # # lg <= C*X + D*U <= ug
    if ADD_C_D_CONSTRAINT:
        C = np.zeros((ocp.dims.ny, ocp.dims.nx))
        C[:ocp.dims.nx, :ocp.dims.nx] = np.eye(ocp.dims.nx)
        D = np.zeros((ocp.dims.ny, ocp.dims.nu))
        # in reality, these variables change according to the referenceline
        D[ocp.dims.ny - 1, 0] = 1
        ocp.constraints.D = D
        ocp.constraints.C = C
        ocp.constraints.lg = np.array(
            [L_LOWER_BOUND, THETA_LOWER_BOUND, KAPPA_LOWER_BOUND, DKAPPA_LOWER_BOUND])
        ocp.constraints.ug = np.array(
            [L_UPPER_BOUND, THETA_UPPER_BOUND, KAPPA_UPPER_BOUND, DKAPPA_UPPER_BOUND])

        ocp.constraints.C_e = np.eye(ocp.dims.nx)
        ocp.constraints.lg_e = np.array(
            [L_LOWER_BOUND, THETA_LOWER_BOUND, KAPPA_LOWER_BOUND])
        ocp.constraints.ug_e = np.array(
            [L_UPPER_BOUND, THETA_UPPER_BOUND, KAPPA_UPPER_BOUND])

    # ocp.constraints.lsg = np.array([1.0])

    # state constraints
    if ADD_BOUND_CONSTRAINT:
        ocp.constraints.lbx_0 = np.array(
            [L_LOWER_BOUND, THETA_LOWER_BOUND, KAPPA_LOWER_BOUND])
        ocp.constraints.ubx_0 = np.array(
            [L_UPPER_BOUND, THETA_UPPER_BOUND, KAPPA_UPPER_BOUND])
        ocp.constraints.idxbx_0 = np.array([0, 1, 2])

        ocp.constraints.lbx = np.array(
            [L_LOWER_BOUND, THETA_LOWER_BOUND, KAPPA_LOWER_BOUND])
        ocp.constraints.ubx = np.array(
            [L_UPPER_BOUND, THETA_UPPER_BOUND, KAPPA_UPPER_BOUND])
        ocp.constraints.idxbx = np.array([0, 1, 2])

        ocp.constraints.lbx_e = np.array(
            [L_LOWER_BOUND, THETA_LOWER_BOUND, KAPPA_LOWER_BOUND])
        ocp.constraints.ubx_e = np.array(
            [L_UPPER_BOUND, THETA_UPPER_BOUND, KAPPA_UPPER_BOUND])
        ocp.constraints.idxbx_e = np.array([0, 1, 2])
        # control bounds
        ocp.constraints.lbu = np.array([dk_lower])
        ocp.constraints.ubu = np.array([dk_upper])
        ocp.constraints.idxbu = np.array([0])

    # ocp.constraints.x0 = np.array([0, 0, 0])

    # ocp parameters default
    ocp.parameter_values = np.array([10.0, 0.01])

    # solver settings
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 1
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.print_level = 0
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = stage_n

    output_debug_info_ocp(ocp)
    json_file = os.path.join("./" + model.name + "_acados_ocp.json")
    acados_solver = AcadosOcpSolver(ocp, json_file=json_file)
    integrator = AcadosSimSolver(ocp, json_file=json_file)
    return acados_solver, integrator


def get_bounds():
    x_ub = []
    x_lb = []
    u_ub = []
    u_lb = []
    xu_lb = []
    xu_ub = []
    for i in range(0, N+1):
        if i == 0:
            x_lb.append(np.array(
                [L_INIT, THETA_INIT, KAPPA_INIT]))
            x_ub.append(np.array(
                [L_INIT, THETA_INIT, KAPPA_INIT]))
            u_lb.append(np.array([DKAPPA_LOWER_BOUND]))
            u_ub.append(np.array([DKAPPA_UPPER_BOUND]))
            xu_lb.append(
                np.array([L_INIT, THETA_INIT, KAPPA_INIT, DKAPPA_LOWER_BOUND]))
            xu_ub.append(
                np.array([L_INIT, THETA_INIT, KAPPA_INIT, DKAPPA_UPPER_BOUND]))
        elif i < N*0.8:
            x_lb.append(np.array(
                [L_LOWER_BOUND, THETA_LOWER_BOUND, KAPPA_LOWER_BOUND]))
            x_ub.append(np.array(
                [L_UPPER_BOUND, THETA_UPPER_BOUND, KAPPA_UPPER_BOUND]))
            u_lb.append(np.array([DKAPPA_LOWER_BOUND]))
            u_ub.append(np.array([DKAPPA_UPPER_BOUND]))
            xu_lb.append(np.array(
                [L_LOWER_BOUND, THETA_LOWER_BOUND, KAPPA_LOWER_BOUND, DKAPPA_LOWER_BOUND]))
            xu_ub.append(np.array(
                [L_UPPER_BOUND, THETA_UPPER_BOUND, KAPPA_UPPER_BOUND, DKAPPA_UPPER_BOUND]))
        elif i < N:
            x_lb.append(np.array(
                [NEW_L_BOUND, THETA_LOWER_BOUND, KAPPA_LOWER_BOUND]))
            x_ub.append(np.array(
                [L_UPPER_BOUND, THETA_UPPER_BOUND, KAPPA_UPPER_BOUND]))
            u_lb.append(np.array([DKAPPA_LOWER_BOUND]))
            u_ub.append(np.array([DKAPPA_UPPER_BOUND]))
            xu_lb.append(np.array(
                [NEW_L_BOUND, THETA_LOWER_BOUND, KAPPA_LOWER_BOUND, DKAPPA_LOWER_BOUND]))
            xu_ub.append(np.array(
                [L_UPPER_BOUND, THETA_UPPER_BOUND, KAPPA_UPPER_BOUND, DKAPPA_UPPER_BOUND]))
        else:
            x_lb.append(np.array(
                [NEW_L_BOUND, THETA_LOWER_BOUND, KAPPA_LOWER_BOUND]))
            x_ub.append(np.array(
                [L_UPPER_BOUND, THETA_UPPER_BOUND, KAPPA_UPPER_BOUND]))
    return x_lb, x_ub, u_lb, u_ub, xu_lb, xu_ub


def plot_acados_results(x, u, x_lb, x_ub, u_lb, u_ub, N, tf):
    """
    Plot the results from Acados solver including states and control with bounds

    Args:
        x: List of state vectors [l, theta, kappa] for each time step
        u: List of control vectors [dkappa] for each time step
        x_lb: List of lower bounds for states
        x_ub: List of upper bounds for states
        u_lb: List of lower bounds for controls
        u_ub: List of upper bounds for controls
        N: Number of control intervals
        tf: Final time
    """
    # Convert lists to numpy arrays
    x = np.array(x)
    u = np.array(u)
    x_lb = np.array(x_lb)
    x_ub = np.array(x_ub)
    u_lb = np.array(u_lb)
    u_ub = np.array(u_ub)

    # Create time vectors
    t_x = np.linspace(0, tf, N+1)  # Time for states
    t_u = np.linspace(0, tf, N)    # Time for controls

    plt.figure(figsize=(12, 10))

    # Plot lateral deviation (l)
    plt.subplot(4, 1, 1)
    plt.plot(t_x, x[:, 0], 'b-', label='l')
    plt.plot(t_x, x_lb[:, 0], 'r--', label='l lower bound')
    plt.plot(t_x, x_ub[:, 0], 'g--', label='l upper bound')
    plt.ylabel('l (m)')
    plt.title('Lateral Deviation')
    plt.legend()
    plt.grid(True)

    # Plot heading angle (theta)
    plt.subplot(4, 1, 2)
    plt.plot(t_x, x[:, 1], 'b-', label='theta')
    plt.plot(t_x, x_lb[:, 1], 'r--', label='theta lower bound')
    plt.plot(t_x, x_ub[:, 1], 'g--', label='theta upper bound')
    plt.ylabel('theta (rad)')
    plt.title('Heading Angle')
    plt.legend()
    plt.grid(True)

    # Plot curvature (kappa)
    plt.subplot(4, 1, 3)
    plt.plot(t_x, x[:, 2], 'b-', label='kappa')
    plt.plot(t_x, x_lb[:, 2], 'r--', label='kappa lower bound')
    plt.plot(t_x, x_ub[:, 2], 'g--', label='kappa upper bound')
    plt.ylabel('kappa (1/m)')
    plt.title('Curvature')
    plt.legend()
    plt.grid(True)

    # Plot curvature rate (dkappa)
    plt.subplot(4, 1, 4)
    plt.step(t_u, u[:, 0], 'b-', where='post', label='dkappa')
    plt.plot(t_u, u_lb[:, 0], 'r--', label='dkappa lower bound')
    plt.plot(t_u, u_ub[:, 0], 'g--', label='dkappa upper bound')
    plt.xlabel('Time (s)')
    plt.ylabel('dkappa (1/m^2)')
    plt.title('Curvature Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()


def visualize_in_cartesian(x, u, N, tf, vel, kr):
    """
    Visualize the reference line and vehicle trajectory in Cartesian coordinates

    Args:
        x: List of state vectors [l, theta, kappa] for each time step
        u: List of control vectors [dkappa] for each time step
        N: Number of control intervals
        tf: Final time
        vel: Vehicle velocity (constant)
        kr: Reference line curvature
    """
    # Convert lists to numpy arrays
    x = np.array(x)
    u = np.array(u)

    # Time vectors
    t_x = np.linspace(0, tf, N+1)  # Time for states
    dt = tf/N  # Time step

    # Initialize arrays for reference line and vehicle trajectory
    ref_x = np.zeros(N+1)
    ref_y = np.zeros(N+1)
    ref_theta = np.zeros(N+1)

    veh_x = np.zeros(N+1)
    veh_y = np.zeros(N+1)
    veh_theta = np.zeros(N+1)

    # Initial conditions
    ref_x[0] = 0
    ref_y[0] = 0
    ref_theta[0] = 0

    veh_x[0] = 0
    veh_y[0] = x[0, 0]  # Initial lateral offset
    veh_theta[0] = x[0, 1]  # Initial heading angle

    # Simulate reference line and vehicle trajectory
    for i in range(1, N+1):
        # Reference line dynamics (constant curvature kr)
        ref_theta[i] = ref_theta[i-1] + vel[i-1] * kr[i-1] * dt
        ref_x[i] = ref_x[i-1] + vel[i-1] * np.cos(ref_theta[i-1]) * dt
        ref_y[i] = ref_y[i-1] + vel[i-1] * np.sin(ref_theta[i-1]) * dt

        # Vehicle dynamics
        kappa = x[i-1, 2]
        delta_theta = x[i-1, 1]

        # Update vehicle heading (theta = ref_theta + delta_theta)
        veh_theta[i] = veh_theta[i-1] + vel[i-1] * kappa * dt

        # Update vehicle position (Frenet to Cartesian)
        veh_x[i] = veh_x[i-1] + vel[i-1] * np.cos(veh_theta[i-1]) * dt
        veh_y[i] = veh_y[i-1] + vel[i-1] * np.sin(veh_theta[i-1]) * dt

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot reference line
    plt.plot(ref_x, ref_y, 'b-', label='Reference line', linewidth=2)

    # Plot vehicle trajectory
    plt.plot(veh_x, veh_y, 'r-', label='Vehicle trajectory', linewidth=2)

    # Plot initial and final positions
    plt.plot(ref_x[0], ref_y[0], 'bo', markersize=8, label='Ref start')
    plt.plot(ref_x[-1], ref_y[-1], 'bs', markersize=8, label='Ref end')
    plt.plot(veh_x[0], veh_y[0], 'ro', markersize=8, label='Veh start')
    plt.plot(veh_x[-1], veh_y[-1], 'rs', markersize=8, label='Veh end')

    # Add arrows to show direction
    arrow_interval = N//10  # Show 10 arrows along the path
    for i in range(0, N+1, arrow_interval):
        plt.arrow(ref_x[i], ref_y[i],
                  0.5*np.cos(ref_theta[i]), 0.5*np.sin(ref_theta[i]),
                  head_width=0.5, head_length=0.7, fc='blue', ec='blue')
        plt.arrow(veh_x[i], veh_y[i],
                  0.5*np.cos(veh_theta[i]), 0.5*np.sin(veh_theta[i]),
                  head_width=0.5, head_length=0.7, fc='red', ec='red')

    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Vehicle Trajectory vs Reference Line in Cartesian Coordinates')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Plot curvature comparison
    plt.figure(figsize=(12, 4))
    plt.plot(t_x, x[:, 2], 'r-', label='Vehicle curvature (kappa)')
    plt.plot(t_x, kr, 'b-', label='Reference curvature (kappa)')

    plt.xlabel('Time (s)')
    plt.ylabel('Curvature (1/m)')
    plt.title('Vehicle Curvature vs Reference Curvature')
    plt.legend()
    plt.grid(True)

# Call the visualization function


if __name__ == "__main__":

    matplotlib.set_loglevel("warning")
    N = 50
    # N = 250
    tf = 5.0
    acados_solver, sim_solver = set_acados_model(N, tf)
    xx = np.array([0.1, 0.1, 0.01])
    uu = np.array([0.1])
    sim_solver.set('T', 0.1)
    xx_next = sim_solver.simulate(x=xx, u=uu, z=None, xdot=None, p=None)
    print(f"xx_next: {xx_next}")
    print(f"N = {N}, tf = {tf}")

    NEW_L_BOUND = 0.5

    x_lb, x_ub, u_lb, u_ub, xu_lb, xu_ub = get_bounds()

    params = [np.array([VEL+i*1*tf/N, KR+i*0.01*tf/N]) for i in range(0, N+1)]

    for i in range(0, N+1):
        acados_solver.set(i, 'p', params[i])
        if ADD_BOUND_CONSTRAINT:
            # set the bounds for the control variables
            if i < N:
                acados_solver.constraints_set(i, "lbu", u_lb[i])
                acados_solver.constraints_set(i, "ubu", u_ub[i])
            # set the bounds for the state variables
            acados_solver.constraints_set(i, "lbx", x_lb[i])
            acados_solver.constraints_set(i, "ubx", x_ub[i])
        if ADD_C_D_CONSTRAINT:
            if i < N:
                acados_solver.constraints_set(i, "lg", xu_lb[i])
                acados_solver.constraints_set(i, "ug", xu_ub[i])
            else:
                acados_solver.constraints_set(i, "lg", x_lb[i])
                acados_solver.constraints_set(i, "ug", x_ub[i])

    start_time = time.perf_counter()
    status = acados_solver.solve()
    end_time = time.perf_counter()
    nlp_iter = acados_solver.get_stats("nlp_iter")
    sqp_iter = acados_solver.get_stats("sqp_iter")
    elapsed_time = (end_time - start_time) * 1000
    x = [acados_solver.get(i, "x") for i in range(N + 1)]
    u = [acados_solver.get(i, "u") for i in range(N)]
    print(
        f"Elapsed time: {elapsed_time:.2f} ms status: {status} nlp_iter: {nlp_iter} sqp_iter: {sqp_iter}")
    for i in range(N):
        print(f"x[{i}]: {x[i]}")
        print(f"u[{i}]: {u[i]}")
    print(f"x[{N}]: {x[N]}")
    plot_acados_results(x, u, x_lb, x_ub, u_lb, u_ub, N, tf)

    vel = [params[i][0] for i in range(N+1)]
    kr = [params[i][1] for i in range(N+1)]
    visualize_in_cartesian(x, u, N, tf, vel, kr)
    plt.show()
    pass
