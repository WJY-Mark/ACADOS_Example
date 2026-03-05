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

# State indices (matching esa_constants.h)
IDX_BETA = 0
IDX_YAW_RATE = 1
IDX_HEADING_ERROR = 2
IDX_LATERAL_ERROR = 3
IDX_FYF = 4
IDX_FYF_CMD = 5

# Normalization scale for control only (matching norm_range_dFyf in C++)
FORCE_SCALE = 1.0e4            # dFyf_cmd_n = dFyf_cmd / FORCE_SCALE

# State bounds (all in physical units)
BETA_UPPER = 0.2               # ~11.5 deg
YAW_RATE_UPPER = 0.5           # rad/s
HEADING_ERROR_UPPER = 0.3      # rad (~17 deg)
LATERAL_ERROR_UPPER = 2.0      # m
FYF_UPPER = 1.0e4              # N
FYF_CMD_UPPER = 1.0e4          # N

BETA_LOWER = -BETA_UPPER
YAW_RATE_LOWER = -YAW_RATE_UPPER
HEADING_ERROR_LOWER = -HEADING_ERROR_UPPER
LATERAL_ERROR_LOWER = -LATERAL_ERROR_UPPER
FYF_LOWER = -FYF_UPPER
FYF_CMD_LOWER = -FYF_CMD_UPPER

# Control bounds (normalized by FORCE_SCALE)
DFYF_CMD_N_UPPER = 10.0        # = 1e5 N/s / FORCE_SCALE
DFYF_CMD_N_LOWER = -DFYF_CMD_N_UPPER

# Initial state (physical units)
BETA_INIT = 0.0
YAW_RATE_INIT = 0.0
HEADING_ERROR_INIT = 0.0
LATERAL_ERROR_INIT = 0.0
FYF_INIT = 0.0                 # N
FYF_CMD_INIT = 0.0             # N

# Vehicle parameters (default values)
VEHICLE_MASS = 1800.0          # kg
VEHICLE_LF = 1.2               # m, CG to front axle
VEHICLE_LR = 1.6               # m, CG to rear axle
VEHICLE_IZ = 3000.0            # kg*m^2, yaw inertia
FYF_TAU = 1e-2                 # s, first-order lag time constant

# Nominal operating parameters
VX_NOMINAL = 10.0              # m/s
KAPPA_REF_NOMINAL = 0.0        # 1/m
SLOPE_R_NOMINAL = 8.0e4        # N/rad, rear tire cornering stiffness
ALPHA_R_BAR_NOMINAL = 0.0      # rad
FYR_BAR_NOMINAL = 0.0          # N

ADD_INIT_CONSTRAINT = True
ADD_BOUND_CONSTRAINT = False
USE_STATE_REF = True

# Cost weights (Bryson's rule inspired)
W_BETA = 0.0
W_YAW_RATE = 1.5e4
W_HEADING_ERROR = 2.0e4
W_LATERAL_ERROR = 1.5e3
W_FYF = 0.0
W_FYF_CMD = 0.0
W_DFYF_CMD_N = 1e-5


def export_esa_mpc_model():
    """
    ESA MPC lateral control model with 6 states and 1 control input.
    Only the control input dFyf_cmd is normalized by FORCE_SCALE
    (matching norm_range_dFyf in C++). All states are in physical units.

    State x = [beta, yaw_rate, delta_theta, lat_error, Fyf, Fyf_cmd]

    Normalized control u = [dFyf_cmd_n]
      where dFyf_cmd_n = dFyf_cmd / FORCE_SCALE

    Parameters p = [vx, kappa_ref, slope_r, alpha_r_bar, Fyr_bar,
                    mass, lf, lr, Iz, tau]

    To recover physical control:
      dFyf_cmd = dFyf_cmd_n * FORCE_SCALE
    """
    model_name = "esa_mpc"

    # states (all physical units)
    beta = ca.SX.sym("beta")
    yaw_rate = ca.SX.sym("yaw_rate")
    delta_theta = ca.SX.sym("delta_theta")
    lat_error = ca.SX.sym("lat_error")
    Fyf = ca.SX.sym("Fyf")
    Fyf_cmd = ca.SX.sym("Fyf_cmd")
    x = ca.vertcat(beta, yaw_rate, delta_theta, lat_error, Fyf, Fyf_cmd)

    # control (normalized)
    dFyf_cmd_n = ca.SX.sym("dFyf_cmd_n")
    u = ca.vertcat(dFyf_cmd_n)

    # state derivatives
    beta_dot = ca.SX.sym("beta_dot")
    yaw_rate_dot = ca.SX.sym("yaw_rate_dot")
    delta_theta_dot = ca.SX.sym("delta_theta_dot")
    lat_error_dot = ca.SX.sym("lat_error_dot")
    Fyf_dot = ca.SX.sym("Fyf_dot")
    Fyf_cmd_dot = ca.SX.sym("Fyf_cmd_dot")
    xdot = ca.vertcat(beta_dot, yaw_rate_dot, delta_theta_dot,
                       lat_error_dot, Fyf_dot, Fyf_cmd_dot)

    # parameters
    vx = ca.SX.sym("vx")
    kappa_ref = ca.SX.sym("kappa_ref")
    slope_r = ca.SX.sym("slope_r")
    alpha_r_bar = ca.SX.sym("alpha_r_bar")
    Fyr_bar = ca.SX.sym("Fyr_bar")
    mass = ca.SX.sym("mass")
    lf = ca.SX.sym("lf")
    lr = ca.SX.sym("lr")
    Iz = ca.SX.sym("Iz")
    tau = ca.SX.sym("tau")
    p = ca.vertcat(vx, kappa_ref, slope_r, alpha_r_bar, Fyr_bar,
                   mass, lf, lr, Iz, tau)

    S = FORCE_SCALE
    d_rear = slope_r * alpha_r_bar + Fyr_bar

    # Row 5: d(Fyf_cmd)/dt = dFyf_cmd = S * dFyf_cmd_n
    f_expl = ca.vertcat(
        -slope_r / (mass * vx) * beta
        + (slope_r * lr / (mass * vx**2) - 1) * yaw_rate
        + 1 / (mass * vx) * Fyf
        + d_rear / (mass * vx),

        slope_r * lr / Iz * beta
        - slope_r * lr**2 / (Iz * vx) * yaw_rate
        + lf / Iz * Fyf
        - d_rear * lr / Iz,

        yaw_rate - vx * kappa_ref,

        vx * beta + vx * delta_theta,

        -1 / tau * Fyf + 1 / tau * Fyf_cmd,

        S * dFyf_cmd_n,
    )

    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name
    model.z = ca.vertcat([])

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

    ocp = AcadosOcp()
    model = export_esa_mpc_model()
    ocp.model = model

    nx = model.x.size()[0]   # 6
    nu = model.u.size()[0]   # 1
    np_ = model.p.size()[0]  # 10
    ny = nx + nu             # 7

    ocp.dims.N = stage_n

    # cost: LINEAR_LS  y = Vx*x + Vu*u,  cost = ||y - yref||_W
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Q = np.diag([W_BETA, W_YAW_RATE, W_HEADING_ERROR,
                 W_LATERAL_ERROR, W_FYF, W_FYF_CMD])
    R = np.diag([W_DFYF_CMD_N])

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx, 0] = 1.0

    W = scipy.linalg.block_diag(Q, R)

    ocp.cost.W_0 = W
    ocp.cost.Vx_0 = Vx
    ocp.cost.Vu_0 = Vu
    ocp.cost.yref_0 = np.zeros(ny)

    ocp.cost.W = W
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.yref = np.zeros(ny)

    ocp.cost.W_e = Q
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.zeros(nx)

    # initial state constraint (all 6 states pinned)
    if ADD_INIT_CONSTRAINT:
        x_lb_0 = np.array([BETA_LOWER, YAW_RATE_LOWER, HEADING_ERROR_LOWER,
                           LATERAL_ERROR_LOWER, FYF_LOWER, FYF_CMD_LOWER])
        x_ub_0 = np.array([BETA_UPPER, YAW_RATE_UPPER, HEADING_ERROR_UPPER,
                           LATERAL_ERROR_UPPER, FYF_UPPER, FYF_CMD_UPPER])
        ocp.constraints.lbx_0 = x_lb_0
        ocp.constraints.ubx_0 = x_ub_0
        ocp.constraints.idxbx_0 = np.arange(nx)

    if ADD_BOUND_CONSTRAINT:
        x_lb = np.array([BETA_LOWER, YAW_RATE_LOWER, HEADING_ERROR_LOWER,
                         LATERAL_ERROR_LOWER, FYF_LOWER, FYF_CMD_LOWER])
        x_ub = np.array([BETA_UPPER, YAW_RATE_UPPER, HEADING_ERROR_UPPER,
                         LATERAL_ERROR_UPPER, FYF_UPPER, FYF_CMD_UPPER])

        ocp.constraints.lbx_0 = x_lb
        ocp.constraints.ubx_0 = x_ub
        ocp.constraints.idxbx_0 = np.arange(nx)

        ocp.constraints.lbx = x_lb
        ocp.constraints.ubx = x_ub
        ocp.constraints.idxbx = np.arange(nx)

        ocp.constraints.lbx_e = x_lb
        ocp.constraints.ubx_e = x_ub
        ocp.constraints.idxbx_e = np.arange(nx)

        ocp.constraints.lbu = np.array([DFYF_CMD_N_LOWER])
        ocp.constraints.ubu = np.array([DFYF_CMD_N_UPPER])
        ocp.constraints.idxbu = np.array([0])

    # default parameter values: [vx, kappa_ref, slope_r, alpha_r_bar, Fyr_bar,
    #                             mass, lf, lr, Iz, tau]
    ocp.parameter_values = np.array([
        VX_NOMINAL, KAPPA_REF_NOMINAL,
        SLOPE_R_NOMINAL, ALPHA_R_BAR_NOMINAL, FYR_BAR_NOMINAL,
        VEHICLE_MASS, VEHICLE_LF, VEHICLE_LR, VEHICLE_IZ, FYF_TAU
    ])

    # solver settings
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
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

def get_reference(N):
    """Generate reference trajectory for the 6-state ESA MPC model.
    y_ref: [beta, yaw_rate, delta_theta, lat_error, Fyf, Fyf_cmd, dFyf_cmd]
    y_ref_e: [beta, yaw_rate, delta_theta, lat_error, Fyf, Fyf_cmd]
    """
    ny = 7  # nx(6) + nu(1)
    nx = 6
    y_ref = []
    for i in range(N):
        ref = np.zeros(ny)
        if i < N * 0.2:
            ref[IDX_LATERAL_ERROR] = 0.0
        elif i < N * 0.6:
            ref[IDX_LATERAL_ERROR] = 1.0
        else:
            ref[IDX_LATERAL_ERROR] = 0.0
        y_ref.append(ref)

    y_ref_e = np.zeros(nx)
    y_ref_e[IDX_LATERAL_ERROR] = 0.0
    return y_ref, y_ref_e

def get_bounds(N):
    """Generate state/control bounds. States are physical, control is normalized."""
    x_lb_vec = np.array([BETA_LOWER, YAW_RATE_LOWER, HEADING_ERROR_LOWER,
                         LATERAL_ERROR_LOWER, FYF_LOWER, FYF_CMD_LOWER])
    x_ub_vec = np.array([BETA_UPPER, YAW_RATE_UPPER, HEADING_ERROR_UPPER,
                         LATERAL_ERROR_UPPER, FYF_UPPER, FYF_CMD_UPPER])
    u_lb_vec = np.array([DFYF_CMD_N_LOWER])
    u_ub_vec = np.array([DFYF_CMD_N_UPPER])

    x_lb = []
    x_ub = []
    u_lb = []
    u_ub = []
    for i in range(N + 1):
        x_lb.append(x_lb_vec.copy())
        x_ub.append(x_ub_vec.copy())
        if i < N:
            u_lb.append(u_lb_vec.copy())
            u_ub.append(u_ub_vec.copy())
    return x_lb, x_ub, u_lb, u_ub


def plot_acados_results(x, u, N, tf, y_ref=None, y_ref_e=None):
    """
    Plot the results from Acados solver for the 6-state ESA MPC model.

    Args:
        x: array (N+1, 6) - [beta, yaw_rate, delta_theta, lat_error, Fyf, Fyf_cmd]
        u: array (N, 1) - [dFyf_cmd]
        N: Number of control intervals
        tf: Final time
        y_ref: list of (N,) refs, each shape (7,)
        y_ref_e: terminal ref shape (6,)
    """
    x = np.array(x)
    u = np.array(u)
    t_x = np.linspace(0, tf, N + 1)
    t_u = np.linspace(0, tf, N)

    has_ref = y_ref is not None and y_ref_e is not None
    if has_ref:
        yr = np.array(y_ref)
        yre = np.array(y_ref_e)

    state_labels = [
        (IDX_BETA, "beta (rad)", "Sideslip Angle"),
        (IDX_YAW_RATE, "yaw_rate (rad/s)", "Yaw Rate"),
        (IDX_HEADING_ERROR, "delta_theta (rad)", "Heading Error"),
        (IDX_LATERAL_ERROR, "lat_error (m)", "Lateral Error"),
        (IDX_FYF, "Fyf (N)", "Front Lateral Force"),
        (IDX_FYF_CMD, "Fyf_cmd (N)", "Front Lateral Force Cmd"),
    ]

    fig, axes = plt.subplots(len(state_labels) + 1, 1, figsize=(12, 16), sharex=True)

    for ax, (idx, ylabel, title) in zip(axes, state_labels):
        ax.plot(t_x, x[:, idx], 'b-', linewidth=2, label=ylabel.split()[0])
        if has_ref:
            ref_vals = np.concatenate([yr[:, idx], [yre[idx]]])
            ax.step(t_x, ref_vals, 'm--', linewidth=1.2, where='post', label='ref')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    axes[-1].step(t_u, u[:, 0], 'b-', linewidth=2, where='post', label='dFyf_cmd')
    if has_ref:
        axes[-1].step(t_u, yr[:, -1], 'm--', linewidth=1.2, where='post', label='ref')
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_ylabel('dFyf_cmd_n (normalized)')
    axes[-1].set_title('Control: dFyf_cmd_n (x FORCE_SCALE = physical N/s)')
    axes[-1].legend()
    axes[-1].grid(True)

    plt.tight_layout()


def visualize_in_cartesian(x, N, tf, vx_list, kappa_list):
    """
    Visualize vehicle trajectory vs reference line in Cartesian coordinates.

    Args:
        x: array (N+1, 6) - states [beta, yaw_rate, delta_theta, lat_error, Fyf, Fyf_cmd]
        N: Number of control intervals
        tf: Final time
        vx_list: longitudinal velocity per step (N+1,)
        kappa_list: reference curvature per step (N+1,)
    """
    x = np.array(x)
    dt = tf / N

    ref_x = np.zeros(N + 1)
    ref_y = np.zeros(N + 1)
    ref_theta = np.zeros(N + 1)
    veh_x = np.zeros(N + 1)
    veh_y = np.zeros(N + 1)

    veh_y[0] = x[0, IDX_LATERAL_ERROR]
    veh_theta = ref_theta.copy()
    veh_theta[0] = x[0, IDX_HEADING_ERROR]

    for i in range(1, N + 1):
        vx = vx_list[i - 1]
        kr = kappa_list[i - 1]

        ref_theta[i] = ref_theta[i - 1] + vx * kr * dt
        ref_x[i] = ref_x[i - 1] + vx * np.cos(ref_theta[i - 1]) * dt
        ref_y[i] = ref_y[i - 1] + vx * np.sin(ref_theta[i - 1]) * dt

        heading = ref_theta[i - 1] + x[i - 1, IDX_HEADING_ERROR]
        veh_x[i] = veh_x[i - 1] + vx * np.cos(heading) * dt
        veh_y[i] = veh_y[i - 1] + vx * np.sin(heading) * dt

    plt.figure(figsize=(12, 6))
    plt.plot(ref_x, ref_y, 'b-', label='Reference line', linewidth=2)
    plt.plot(veh_x, veh_y, 'r-', label='Vehicle trajectory', linewidth=2)
    plt.plot(ref_x[0], ref_y[0], 'bo', markersize=8, label='Ref start')
    plt.plot(veh_x[0], veh_y[0], 'ro', markersize=8, label='Veh start')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Vehicle Trajectory vs Reference Line')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')


if __name__ == "__main__":

    matplotlib.set_loglevel("warning")
    N = 50
    tf = 5.0
    acados_solver, sim_solver = set_acados_model(N, tf)

    # quick integrator test (force states are normalized)
    xx = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    uu = np.array([0.01])
    sim_solver.set('T', tf / N)
    xx_next = sim_solver.simulate(x=xx, u=uu)
    print(f"sim test: x_next = {xx_next}")

    # parameter vector: [vx, kappa_ref, slope_r, alpha_r_bar, Fyr_bar,
    #                     mass, lf, lr, Iz, tau]
    default_p = np.array([
        VX_NOMINAL, KAPPA_REF_NOMINAL,
        SLOPE_R_NOMINAL, ALPHA_R_BAR_NOMINAL, FYR_BAR_NOMINAL,
        VEHICLE_MASS, VEHICLE_LF, VEHICLE_LR, VEHICLE_IZ, FYF_TAU
    ])
    params = [default_p.copy() for _ in range(N + 1)]

    y_ref, y_ref_e = get_reference(N)
    x_lb, x_ub, u_lb, u_ub = get_bounds(N)

    x0 = np.array([BETA_INIT, YAW_RATE_INIT, HEADING_ERROR_INIT,
                    LATERAL_ERROR_INIT, FYF_INIT, FYF_CMD_INIT])
    x0[IDX_LATERAL_ERROR] = 0.5  # start with 0.5m offset

    for i in range(N + 1):
        acados_solver.set(i, 'p', params[i])
        if ADD_BOUND_CONSTRAINT:
            if i < N:
                acados_solver.constraints_set(i, "lbu", u_lb[i])
                acados_solver.constraints_set(i, "ubu", u_ub[i])
            acados_solver.constraints_set(i, "lbx", x_lb[i])
            acados_solver.constraints_set(i, "ubx", x_ub[i])
        if USE_STATE_REF:
            if i < N:
                acados_solver.cost_set(i, "yref", y_ref[i])
            else:
                acados_solver.cost_set(i, "yref", y_ref_e)

    start_time = time.perf_counter()
    status = acados_solver.solve_for_x0(x0)
    elapsed_time = (time.perf_counter() - start_time) * 1000
    nlp_iter = acados_solver.get_stats("nlp_iter")
    sqp_iter = acados_solver.get_stats("sqp_iter")

    x_sol = [acados_solver.get(i, "x") for i in range(N + 1)]
    u_sol = [acados_solver.get(i, "u") for i in range(N)]
    print(f"Elapsed: {elapsed_time:.2f} ms  status: {status}  "
          f"nlp_iter: {nlp_iter}  sqp_iter: {sqp_iter}")
    print(f"x[0] = {x_sol[0]}")
    print(f"x[{N}] = {x_sol[N]}")

    plot_acados_results(x_sol, u_sol, N, tf, y_ref, y_ref_e)

    vx_list = [params[i][0] for i in range(N + 1)]
    kappa_list = [params[i][1] for i in range(N + 1)]
    visualize_in_cartesian(x_sol, N, tf, vx_list, kappa_list)
    plt.show()
