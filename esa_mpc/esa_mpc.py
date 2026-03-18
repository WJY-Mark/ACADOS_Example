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

from scipy.linalg import expm


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# State indices
IDX_BETA = 0
IDX_YAW_RATE = 1
IDX_HEADING_ERROR = 2
IDX_LATERAL_ERROR = 3
IDX_FYF = 4

# Normalization: x_or_u_normalized = x_or_u_physical / SCALE
# Set SCALE = 1.0 to disable normalization for that variable
FYF_SCALE = 1.0e4              # Fyf_n = Fyf / FYF_SCALE
DF_SCALE = 1.0e4               # dFyf_n = dFyf / DF_SCALE

# State bounds (physical values, Fyf auto-scaled by FYF_SCALE)
BETA_UPPER = 0.2               # ~11.5 deg
YAW_RATE_UPPER = 0.5           # rad/s
HEADING_ERROR_UPPER = 1.0      # rad (~17 deg)
LATERAL_ERROR_UPPER = 5.0      # m
FYF_PHYSICAL_MAX = 1.0e4       # N
FYF_N_UPPER = FYF_PHYSICAL_MAX / FYF_SCALE
FYF_N_LOWER = -FYF_N_UPPER

BETA_LOWER = -BETA_UPPER
YAW_RATE_LOWER = -YAW_RATE_UPPER
HEADING_ERROR_LOWER = -HEADING_ERROR_UPPER
LATERAL_ERROR_LOWER = -LATERAL_ERROR_UPPER

# Control bounds (physical value, auto-scaled by DF_SCALE)
DFYF_PHYSICAL_MAX = 1.0e5     # N/s
DFYF_N_UPPER = DFYF_PHYSICAL_MAX / DF_SCALE
DFYF_N_LOWER = -DFYF_N_UPPER

# Initial state (physical values, Fyf auto-scaled)
BETA_INIT = 0.0
YAW_RATE_INIT = 0.0
HEADING_ERROR_INIT = 0.0
LATERAL_ERROR_INIT = 0.0
FYF_N_INIT = 0.0               # = Fyf_physical_init / FYF_SCALE

# Vehicle parameters (default values)
VEHICLE_MASS = 2594.0          # kg
VEHICLE_LF = 1.588               # m, CG to front axle
VEHICLE_LR = 1.451               # m, CG to rear axle
VEHICLE_IZ = 2500.0            # kg*m^2, yaw inertia

# Nominal operating parameters
VX_NOMINAL = 33.39            # m/s
KAPPA_REF_NOMINAL = 0.0        # 1/m
SLOPE_R_NOMINAL = 149738        # N/rad, rear tire cornering stiffness
ALPHA_R_BAR_NOMINAL = 0.0      # rad
FYR_BAR_NOMINAL = 0.0          # N

# Discretization mode: True = ZOH (pre-computed, exact for LTI),
#                      False = acados IRK (implicit Runge-Kutta)
USE_ZOH = False

ADD_INIT_CONSTRAINT = True
ADD_BOUND_CONSTRAINT = True
USE_STATE_REF = True

# Cost weights (Bryson's rule inspired)
W_BETA = 0.0
W_YAW_RATE = 15.0
W_HEADING_ERROR = 20.0
W_LATERAL_ERROR = 15
# Physical weight for Fyf, auto-scaled by FYF_SCALE^2
W_FYF_PHYSICAL = 0.0
W_FYF_N = W_FYF_PHYSICAL / (FYF_SCALE * FYF_SCALE)
# Physical weight for dFyf, auto-scaled by DF_SCALE^2
W_DFYF_PHYSICAL = 1.0e8
W_DFYF_N = W_DFYF_PHYSICAL / (DF_SCALE * DF_SCALE)


NX = 5
NU = 1
# Discrete parameter layout: [A_d(25), Bu_d(5), Bd_d(5)] = 35
N_PARAM_DISC = NX * NX + NX * NU + NX  # 35


def compute_continuous_matrices(vx, kappa_ref, slope_r, alpha_r_bar, Fyr_bar,
                                mass, lf, lr, Iz):
    """Build A_con, Bu_con, Bd_con from physical parameters (after normalization)."""
    Sf = FYF_SCALE
    Sd = DF_SCALE
    d_rear = slope_r * alpha_r_bar + Fyr_bar

    A_con = np.array([
        [-slope_r / (mass * vx), slope_r * lr / (mass * vx**2) - 1, 0, 0, Sf / (mass * vx)],
        [slope_r * lr / Iz,     -slope_r * lr**2 / (Iz * vx),       0, 0, Sf * lf / Iz],
        [0,                      1,                                   0, 0, 0],
        [vx,                     0,                                   vx, 0, 0],
        [0,                      0,                                   0, 0, 0],
    ])

    Bu_con = np.array([
        [0],
        [0],
        [0],
        [0],
        [Sd / Sf],
    ])

    Bd_con = np.array([
        d_rear / (mass * vx),
        -d_rear * lr / Iz,
        -vx * kappa_ref,
        0,
        0,
    ])

    return A_con, Bu_con, Bd_con


def c2d_zoh(A_con, Bu_con, Bd_con, dt):
    """Zero-order hold discretization via matrix exponential (exact for LTI)."""
    nx = A_con.shape[0]
    nu = Bu_con.shape[1]

    M = np.zeros((nx + nu + 1, nx + nu + 1))
    M[:nx, :nx] = A_con * dt
    M[:nx, nx:nx + nu] = Bu_con * dt
    M[:nx, nx + nu] = Bd_con * dt

    eM = expm(M)

    A_d = eM[:nx, :nx]
    Bu_d = eM[:nx, nx:nx + nu]
    Bd_d = eM[:nx, nx + nu]

    return A_d, Bu_d, Bd_d


def discretize_stage(phys_params, dt):
    """Compute discrete matrices for one stage and pack into parameter vector.
    phys_params: [vx, kappa_ref, slope_r, alpha_r_bar, Fyr_bar, mass, lf, lr, Iz]
    Returns: flat array of length N_PARAM_DISC = 35
    """
    A_con, Bu_con, Bd_con = compute_continuous_matrices(*phys_params)
    A_d, Bu_d, Bd_d = c2d_zoh(A_con, Bu_con, Bd_con, dt)
    return np.concatenate([A_d.flatten(order='F'), Bu_d.flatten(order='F'), Bd_d])


N_PARAM_PHYS = 9   # [vx, kappa_ref, slope_r, alpha_r_bar, Fyr_bar, mass, lf, lr, Iz]


def build_kkt_matrix(A_list, Bu_list, Q, R, Q_e, N, nx, nu, sigma=0.0):
    """
    Build the KKT matrix of the OCP QP, optionally with IPM barrier diagonal.

    HPIPM solves at each IPM iteration:
        [H + Σ   G^T] [Δz]     [−r_z ]
        [G       0  ] [Δλ]  =  [−r_λ ]

    where:
        H = blkdiag(Q,R, Q,R, ..., Q,R, Q_e)   (cost Hessian)
        G = [A_k  B_k  -I]                       (dynamics Jacobian)
        Σ = diag(μ_i/s_i)                        (barrier term from box constraints)

    Since Σ depends on the IPM iterate (not accessible from Python API),
    sigma provides a uniform scalar approximation: Σ ≈ sigma * I.
        sigma = 0   → equality-only KKT (lower bound on conditioning)
        sigma > 0   → approximate first IPM iteration (e.g. sigma = 1e-2)

    Returns (KKT_matrix, condition_number).
    """
    nz = (N + 1) * nx + N * nu
    nc = N * nx
    dim = nz + nc

    KKT = np.zeros((dim, dim))

    def x_idx(k):
        return k * (nx + nu)

    def u_idx(k):
        return k * (nx + nu) + nx

    for k in range(N):
        ix = x_idx(k)
        iu = u_idx(k)
        KKT[ix:ix+nx, ix:ix+nx] = Q + sigma * np.eye(nx)
        KKT[iu:iu+nu, iu:iu+nu] = R + sigma * np.eye(nu)

    ix_N = x_idx(N)
    KKT[ix_N:ix_N+nx, ix_N:ix_N+nx] = Q_e + sigma * np.eye(nx)

    for k in range(N):
        row = nz + k * nx
        ix_k = x_idx(k)
        iu_k = u_idx(k)
        ix_k1 = x_idx(k + 1)

        KKT[row:row+nx, ix_k:ix_k+nx] = A_list[k]
        KKT[ix_k:ix_k+nx, row:row+nx] = A_list[k].T

        KKT[row:row+nx, iu_k:iu_k+nu] = Bu_list[k]
        KKT[iu_k:iu_k+nu, row:row+nx] = Bu_list[k].T

        KKT[row:row+nx, ix_k1:ix_k1+nx] = -np.eye(nx)
        KKT[ix_k1:ix_k1+nx, row:row+nx] = -np.eye(nx)

    svs = np.linalg.svd(KKT, compute_uv=False)
    cond = svs[0] / svs[svs > 1e-15][-1] if np.any(svs > 1e-15) else np.inf

    return KKT, cond


def export_esa_mpc_model():
    """
    ESA MPC lateral control model (5 states, 1 control).

    When USE_ZOH=True  (DISCRETE):
        Parameters p = [A_d_flat(25), Bu_d_flat(5), Bd_d(5)]  (total 35)
        Dynamics: x_{k+1} = A_d @ x_k + Bu_d @ u_k + Bd_d

    When USE_ZOH=False (IRK continuous):
        Parameters p = [vx, kappa_ref, slope_r, alpha_r_bar, Fyr_bar,
                        mass, lf, lr, Iz]  (total 9)
        Dynamics: xdot = f_expl(x, u, p)
    """
    model_name = "esa_mpc"
    nx = NX
    nu = NU

    beta = ca.SX.sym("beta")
    yaw_rate = ca.SX.sym("yaw_rate")
    delta_theta = ca.SX.sym("delta_theta")
    lat_error = ca.SX.sym("lat_error")
    Fyf_n = ca.SX.sym("Fyf_n")
    x = ca.vertcat(beta, yaw_rate, delta_theta, lat_error, Fyf_n)

    dFyf_n = ca.SX.sym("dFyf_n")
    u = ca.vertcat(dFyf_n)

    model = AcadosModel()

    if USE_ZOH:
        A_d_flat = ca.SX.sym("A_d", nx * nx)
        Bu_d_flat = ca.SX.sym("Bu_d", nx * nu)
        Bd_d = ca.SX.sym("Bd_d", nx)
        p = ca.vertcat(A_d_flat, Bu_d_flat, Bd_d)

        A_d = ca.reshape(A_d_flat, nx, nx)
        Bu_d = ca.reshape(Bu_d_flat, nx, nu)

        model.disc_dyn_expr = A_d @ x + Bu_d @ u + Bd_d
    else:
        vx = ca.SX.sym("vx")
        kappa_ref = ca.SX.sym("kappa_ref")
        slope_r = ca.SX.sym("slope_r")
        alpha_r_bar = ca.SX.sym("alpha_r_bar")
        Fyr_bar = ca.SX.sym("Fyr_bar")
        mass = ca.SX.sym("mass")
        lf = ca.SX.sym("lf")
        lr = ca.SX.sym("lr")
        Iz = ca.SX.sym("Iz")
        p = ca.vertcat(vx, kappa_ref, slope_r, alpha_r_bar, Fyr_bar,
                       mass, lf, lr, Iz)

        Sf = FYF_SCALE
        Sd = DF_SCALE
        d_rear = slope_r * alpha_r_bar + Fyr_bar

        f_expl = ca.vertcat(
            -slope_r / (mass * vx) * beta
            + (slope_r * lr / (mass * vx**2) - 1) * yaw_rate
            + Sf / (mass * vx) * Fyf_n
            + d_rear / (mass * vx),

            slope_r * lr / Iz * beta
            - slope_r * lr**2 / (Iz * vx) * yaw_rate
            + Sf * lf / Iz * Fyf_n
            - d_rear * lr / Iz,

            yaw_rate - vx * kappa_ref,

            vx * beta + vx * delta_theta,

            Sd / Sf * dFyf_n,
        )

        beta_dot = ca.SX.sym("beta_dot")
        yaw_rate_dot = ca.SX.sym("yaw_rate_dot")
        delta_theta_dot = ca.SX.sym("delta_theta_dot")
        lat_error_dot = ca.SX.sym("lat_error_dot")
        Fyf_n_dot = ca.SX.sym("Fyf_n_dot")
        xdot = ca.vertcat(beta_dot, yaw_rate_dot, delta_theta_dot,
                           lat_error_dot, Fyf_n_dot)

        model.f_expl_expr = f_expl
        model.f_impl_expr = xdot - f_expl
        model.xdot = xdot

    model.x = x
    model.u = u
    model.p = p
    model.name = model_name

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
    if ocp.solver_options.integrator_type != "DISCRETE":
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


def set_acados_model(stage_n, tf, time_steps=None):
    """
    time_steps: optional array of length stage_n specifying per-stage dt.
                If None, uniform dt = tf/stage_n is used.
                If provided, tf is ignored and computed from sum(time_steps).
    """
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    acados_models_dir = "./acados_models"
    safe_mkdir_recursive(os.path.join(os.getcwd(), acados_models_dir))
    acados_source_path = os.environ["ACADOS_SOURCE_DIR"]
    sys.path.insert(0, acados_source_path)

    if time_steps is not None:
        time_steps = np.array(time_steps, dtype=float)
        assert len(time_steps) == stage_n
        tf = float(np.sum(time_steps))

    ocp = AcadosOcp()
    model = export_esa_mpc_model()
    ocp.model = model

    nx = model.x.size()[0]   # 5
    nu = model.u.size()[0]   # 1
    ny = nx + nu             # 6

    ocp.dims.N = stage_n

    # cost: LINEAR_LS  y = Vx*x + Vu*u,  cost = ||y - yref||_W
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Q = np.diag([W_BETA, W_YAW_RATE, W_HEADING_ERROR,
                 W_LATERAL_ERROR, W_FYF_N])
    R = np.diag([W_DFYF_N])

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

    if ADD_INIT_CONSTRAINT:
        x_lb_0 = np.array([BETA_LOWER, YAW_RATE_LOWER, HEADING_ERROR_LOWER,
                           LATERAL_ERROR_LOWER, FYF_N_LOWER])
        x_ub_0 = np.array([BETA_UPPER, YAW_RATE_UPPER, HEADING_ERROR_UPPER,
                           LATERAL_ERROR_UPPER, FYF_N_UPPER])
        ocp.constraints.lbx_0 = x_lb_0
        ocp.constraints.ubx_0 = x_ub_0
        ocp.constraints.idxbx_0 = np.arange(nx)

    if ADD_BOUND_CONSTRAINT:
        x_lb = np.array([BETA_LOWER, YAW_RATE_LOWER, HEADING_ERROR_LOWER,
                         LATERAL_ERROR_LOWER, FYF_N_LOWER])
        x_ub = np.array([BETA_UPPER, YAW_RATE_UPPER, HEADING_ERROR_UPPER,
                         LATERAL_ERROR_UPPER, FYF_N_UPPER])

        ocp.constraints.lbx_0 = x_lb
        ocp.constraints.ubx_0 = x_ub
        ocp.constraints.idxbx_0 = np.arange(nx)

        ocp.constraints.lbx = x_lb
        ocp.constraints.ubx = x_ub
        ocp.constraints.idxbx = np.arange(nx)

        ocp.constraints.lbx_e = x_lb
        ocp.constraints.ubx_e = x_ub
        ocp.constraints.idxbx_e = np.arange(nx)

        ocp.constraints.lbu = np.array([DFYF_N_LOWER])
        ocp.constraints.ubu = np.array([DFYF_N_UPPER])
        ocp.constraints.idxbu = np.array([0])

    default_phys = np.array([
        VX_NOMINAL, KAPPA_REF_NOMINAL,
        SLOPE_R_NOMINAL, ALPHA_R_BAR_NOMINAL, FYR_BAR_NOMINAL,
        VEHICLE_MASS, VEHICLE_LF, VEHICLE_LR, VEHICLE_IZ
    ])

    default_dt = tf / stage_n
    if USE_ZOH:
        ocp.parameter_values = discretize_stage(default_phys, default_dt)
    else:
        ocp.parameter_values = default_phys

    # solver settings
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    if USE_ZOH:
        ocp.solver_options.integrator_type = "DISCRETE"
    else:
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.print_level = 0
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = stage_n
    if time_steps is not None:
        ocp.solver_options.time_steps = time_steps

    output_debug_info_ocp(ocp)
    json_file = os.path.join("./" + model.name + "_acados_ocp.json")
    acados_solver = AcadosOcpSolver(ocp, json_file=json_file)

    sim_solver = None
    if not USE_ZOH:
        sim_solver = AcadosSimSolver(ocp, json_file=json_file)
    return acados_solver, sim_solver

def get_reference(N):
    """Generate reference trajectory for the 5-state ESA MPC model.
    y_ref: [beta, yaw_rate, delta_theta, lat_error, Fyf, dFyf_n]
    y_ref_e: [beta, yaw_rate, delta_theta, lat_error, Fyf]
    """
    ny = 6  # nx(5) + nu(1)
    nx = 5
    y_ref = []
    for i in range(N):
        ref = np.zeros(ny)
        if i < N * 0.2:
            ref[IDX_LATERAL_ERROR] = 3.0
        elif i < N * 0.6:
            ref[IDX_LATERAL_ERROR] = 3.0
        else:
            ref[IDX_LATERAL_ERROR] = 3.0
        y_ref.append(ref)

    y_ref_e = np.zeros(nx)
    y_ref_e[IDX_LATERAL_ERROR] = 3.0
    return y_ref, y_ref_e

def get_bounds(N):
    """Generate state/control bounds. States are physical, control is normalized."""
    x_lb_vec = np.array([BETA_LOWER, YAW_RATE_LOWER, HEADING_ERROR_LOWER,
                         LATERAL_ERROR_LOWER, FYF_N_LOWER])
    x_ub_vec = np.array([BETA_UPPER, YAW_RATE_UPPER, HEADING_ERROR_UPPER,
                         LATERAL_ERROR_UPPER, FYF_N_UPPER])
    u_lb_vec = np.array([DFYF_N_LOWER])
    u_ub_vec = np.array([DFYF_N_UPPER])

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


def plot_acados_results(x, u, N, tf, y_ref=None, y_ref_e=None, time_steps=None):
    """
    Plot the results from Acados solver for the 5-state ESA MPC model.

    Args:
        x: array (N+1, 5)
        u: array (N, 1)
        N: Number of control intervals
        tf: Final time
        y_ref: list of (N,) refs, each shape (6,)
        y_ref_e: terminal ref shape (5,)
        time_steps: optional per-stage dt array (length N) for non-uniform grids
    """
    x = np.array(x)
    u = np.array(u)
    if time_steps is not None:
        t_x = np.concatenate([[0.0], np.cumsum(time_steps)])
        t_u = t_x[:-1]
    else:
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
        (IDX_FYF, "Fyf_n (normalized)", "Front Lateral Force (x FYF_SCALE = N)"),
    ]

    fig, axes = plt.subplots(len(state_labels) + 1, 1, figsize=(12, 14), sharex=True)

    for ax, (idx, ylabel, title) in zip(axes, state_labels):
        ax.plot(t_x, x[:, idx], 'b-', linewidth=2, label=ylabel.split()[0])
        if has_ref:
            ref_vals = np.concatenate([yr[:, idx], [yre[idx]]])
            ax.step(t_x, ref_vals, 'm--', linewidth=1.2, where='post', label='ref')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    axes[-1].step(t_u, u[:, 0], 'b-', linewidth=2, where='post', label='dFyf_n')
    if has_ref:
        axes[-1].step(t_u, yr[:, -1], 'm--', linewidth=1.2, where='post', label='ref')
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_ylabel('dFyf_n (normalized)')
    axes[-1].set_title('Control: dFyf_n (x DF_SCALE = physical N/s)')
    axes[-1].legend()
    axes[-1].grid(True)

    plt.tight_layout()


def visualize_in_cartesian(x, N, tf, vx_list, kappa_list, time_steps=None):
    """
    Visualize vehicle trajectory vs reference line in Cartesian coordinates.

    Args:
        x: array (N+1, 5) - states [beta, yaw_rate, delta_theta, lat_error, Fyf]
        N: Number of control intervals
        tf: Final time
        vx_list: longitudinal velocity per step (N+1,)
        kappa_list: reference curvature per step (N+1,)
        time_steps: optional per-stage dt array (length N) for non-uniform grids
    """
    x = np.array(x)
    if time_steps is None:
        time_steps = np.full(N, tf / N)

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
        dt_i = time_steps[i - 1]

        ref_theta[i] = ref_theta[i - 1] + vx * kr * dt_i
        ref_x[i] = ref_x[i - 1] + vx * np.cos(ref_theta[i - 1]) * dt_i
        ref_y[i] = ref_y[i - 1] + vx * np.sin(ref_theta[i - 1]) * dt_i

        heading = ref_theta[i - 1] + x[i - 1, IDX_HEADING_ERROR]
        veh_x[i] = veh_x[i - 1] + vx * np.cos(heading) * dt_i
        veh_y[i] = veh_y[i - 1] + vx * np.sin(heading) * dt_i

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
    N = 40
    time_steps = np.concatenate([
        np.full(20, 0.02),   # first 20 steps:  dt = 0.02 s
        np.full(20, 0.2),    # last  20 steps:  dt = 0.2  s
    ])
    tf = float(np.sum(time_steps))  # 0.4 + 4.0 = 4.4 s
    acados_solver, sim_solver = set_acados_model(N, tf, time_steps=time_steps)

    # Physical parameters: [vx, kappa_ref, slope_r, alpha_r_bar, Fyr_bar,
    #                        mass, lf, lr, Iz]
    default_phys = np.array([
        VX_NOMINAL, KAPPA_REF_NOMINAL,
        SLOPE_R_NOMINAL, ALPHA_R_BAR_NOMINAL, FYR_BAR_NOMINAL,
        VEHICLE_MASS, VEHICLE_LF, VEHICLE_LR, VEHICLE_IZ
    ])
    phys_params = [default_phys.copy() for _ in range(N)]

    if USE_ZOH:
        solver_params = [discretize_stage(phys_params[i], time_steps[i])
                         for i in range(N)]
    else:
        solver_params = phys_params

    y_ref, y_ref_e = get_reference(N)
    x_lb, x_ub, u_lb, u_ub = get_bounds(N)

    x0 = np.array([BETA_INIT, YAW_RATE_INIT, HEADING_ERROR_INIT,
                    LATERAL_ERROR_INIT, FYF_N_INIT])
    x0[IDX_LATERAL_ERROR] = 0.5  # start with 0.5m offset

    for i in range(N):
        acados_solver.set(i, 'p', solver_params[i])
        if ADD_BOUND_CONSTRAINT:
            acados_solver.constraints_set(i, "lbu", u_lb[i])
            acados_solver.constraints_set(i, "ubu", u_ub[i])
            acados_solver.constraints_set(i, "lbx", x_lb[i])
            acados_solver.constraints_set(i, "ubx", x_ub[i])
        if USE_STATE_REF:
            acados_solver.cost_set(i, "yref", y_ref[i])
    acados_solver.set(N, 'p', solver_params[-1])
    if ADD_BOUND_CONSTRAINT:
        acados_solver.constraints_set(N, "lbx", x_lb[N])
        acados_solver.constraints_set(N, "ubx", x_ub[N])
    if USE_STATE_REF:
        acados_solver.cost_set(N, "yref", y_ref_e)

    np.set_printoptions(precision=6, linewidth=120, suppress=True)
    if USE_ZOH:
        p0 = acados_solver.get(0, 'p')
        A_d = p0[:NX*NX].reshape(NX, NX, order='F')
        Bu_d = p0[NX*NX:NX*NX+NX*NU].reshape(NX, NU, order='F')
        Bd_d = p0[NX*NX+NX*NU:]
        print("=== Stage 0 discrete matrices (ZOH) ===")
        print(f"A_d ({NX}x{NX}):\n{A_d}")
        print(f"Bu_d ({NX}x{NU}):\n{Bu_d}")
        print(f"Bd_d ({NX}x1):\n{Bd_d}")
        eigs = np.linalg.eigvals(A_d)
        print(f"eig(A_d): {eigs}")
        print(f"|eig|:    {np.abs(eigs)}")
        print()
    else:
        sim_solver.set("T", time_steps[0])
        sim_solver.set("p", solver_params[0])
        x_zero = np.zeros(NX)
        u_zero = np.zeros(NU)
        sim_solver.set("x", x_zero)
        sim_solver.set("u", u_zero)
        sim_solver.solve()
        Bd_d = sim_solver.get("x")
        A_d = sim_solver.get("Sx")
        Bu_d = sim_solver.get("Su")
        print("=== Stage 0 discrete matrices (IRK sensitivities) ===")
        print(f"A_d = Sx ({NX}x{NX}):\n{A_d}")
        print(f"Bu_d = Su ({NX}x{NU}):\n{Bu_d}")
        print(f"Bd_d = f(0,0) ({NX}x1):\n{Bd_d}")
        eigs = np.linalg.eigvals(A_d)
        print(f"eig(A_d): {eigs}")
        print(f"|eig|:    {np.abs(eigs)}")
        print()

    # Extract per-stage A_d, Bu_d for KKT analysis
    A_list = []
    Bu_list = []
    if USE_ZOH:
        for i in range(N):
            pi = acados_solver.get(i, 'p')
            A_list.append(pi[:NX*NX].reshape(NX, NX, order='F'))
            Bu_list.append(pi[NX*NX:NX*NX+NX*NU].reshape(NX, NU, order='F'))
    else:
        for i in range(N):
            sim_solver.set("T", time_steps[i])
            sim_solver.set("p", solver_params[i])
            sim_solver.set("x", np.zeros(NX))
            sim_solver.set("u", np.zeros(NU))
            sim_solver.solve()
            A_list.append(sim_solver.get("Sx"))
            Bu_list.append(sim_solver.get("Su"))

    Q_cost = np.diag([W_BETA, W_YAW_RATE, W_HEADING_ERROR,
                       W_LATERAL_ERROR, W_FYF_N])
    R_cost = np.diag([W_DFYF_N])

    _, cond_eq = build_kkt_matrix(A_list, Bu_list, Q_cost, R_cost,
                                  Q_cost, N, NX, NU, sigma=0.0)
    _, cond_ipm = build_kkt_matrix(A_list, Bu_list, Q_cost, R_cost,
                                   Q_cost, N, NX, NU, sigma=1e-2)
    print("=== KKT condition number (2-norm) ===")
    print(f"  Σ=0   (equality-only):     cond = {cond_eq:.2e}  log10 = {np.log10(cond_eq):.1f}")
    print(f"  Σ=0.01 (approx IPM init):  cond = {cond_ipm:.2e}  log10 = {np.log10(cond_ipm):.1f}")
    print(f"  (cond > ~1e12 → HPIPM likely fails with MINSTEP)")
    print()

    start_time = time.perf_counter()
    status = acados_solver.solve_for_x0(x0)
    status = acados_solver.get_status()
    elapsed_time = (time.perf_counter() - start_time) * 1000
    nlp_iter = acados_solver.get_stats("nlp_iter")
    sqp_iter = acados_solver.get_stats("sqp_iter")

    x_sol = [acados_solver.get(i, "x") for i in range(N + 1)]
    u_sol = [acados_solver.get(i, "u") for i in range(N)]
    print(f"Elapsed: {elapsed_time:.2f} ms  status: {status}  "
          f"nlp_iter: {nlp_iter}  sqp_iter: {sqp_iter}")
    print(f"x[0] = {x_sol[0]}")
    print(f"x[{N}] = {x_sol[N]}")
    print(f"u = {u_sol}")
    acados_solver.print_statistics()

    plot_acados_results(x_sol, u_sol, N, tf, y_ref, y_ref_e, time_steps=time_steps)

    vx_list = [phys_params[min(i, N - 1)][0] for i in range(N + 1)]
    kappa_list = [phys_params[min(i, N - 1)][1] for i in range(N + 1)]
    visualize_in_cartesian(x_sol, N, tf, vx_list, kappa_list, time_steps=time_steps)
    plt.show()
