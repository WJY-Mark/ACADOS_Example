from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import numpy as np
import scipy.linalg
import os
import sys
import shutil
import errno
import time
import logging

import casadi as ca
import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Inertia time constant for a_cmd -> a (first-order lag)
TAU_A = 0.1  # seconds

# Number of shooting intervals
N_STEPS = 40
N_SHORT = 20   # first 20 steps with dt_short
N_LONG = 20    # last 20 steps with dt_long
DT_SHORT = 0.02  # seconds
DT_LONG = 0.2    # seconds

# Default bounds
S_LOWER = -1e8       # s lower: no constraint (very large negative)
S_UPPER = 2.0       # s upper: < 40
V_LOWER = 0.0        # v lower: >= 0
V_UPPER = 1e8        # v upper: no constraint (very large positive)
A_LOWER = -6.0
A_UPPER = 4.0
A_CMD_LOWER = -9.0   # a_cmd lower: > -9
A_CMD_UPPER = 9.0    # a_cmd upper: < 9
J_CMD_LOWER = -10.0  # jerk: -10 ~ 10
J_CMD_UPPER = 10.0

# Cost weights (Bryson's rule style)
W_S = 1.0
W_V = 1.0
W_A = 10.0
W_A_CMD = 5.0
W_J_CMD = 1.0

# Slack penalty weights for soft box constraint on s only
# L2 penalty (quadratic)
ZL_S = 0.0         # s lower is -inf, no need to penalize
ZU_S = 100.0       # s upper = 40, soft penalty
# L1 penalty (linear)
ZL1_S = 0.0        # s lower is -inf
ZU1_S = 1000.0     # s upper = 40


def export_lon_model():
    """
    Longitudinal dynamics model with inertia element.
    States: [s, v, a, a_cmd]
    Control: [j_cmd]
    Dynamics:
        s_dot = v
        v_dot = a
        a_dot = (a_cmd - a) / tau
        a_cmd_dot = j_cmd
    """
    model_name = "lon_pwj"

    # States
    s = ca.SX.sym("s")
    v = ca.SX.sym("v")
    a = ca.SX.sym("a")
    a_cmd = ca.SX.sym("a_cmd")
    x = ca.vertcat(s, v, a, a_cmd)

    # Control
    j_cmd = ca.SX.sym("j_cmd")
    u = ca.vertcat(j_cmd)

    # State derivatives
    s_dot = ca.SX.sym("s_dot")
    v_dot = ca.SX.sym("v_dot")
    a_dot = ca.SX.sym("a_dot")
    a_cmd_dot = ca.SX.sym("a_cmd_dot")
    xdot = ca.vertcat(s_dot, v_dot, a_dot, a_cmd_dot)

    # Parameters: tau (inertia time constant)
    tau = ca.SX.sym("tau")
    p = ca.vertcat(tau)

    # Explicit dynamics
    f_expl = ca.vertcat(
        v,
        a,
        (a_cmd - a) / tau,
        j_cmd
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

    return model


def build_time_steps():
    """Build non-uniform time steps: first N_SHORT steps at DT_SHORT, last N_LONG at DT_LONG."""
    dt_vec = np.zeros(N_STEPS)
    dt_vec[:N_SHORT] = DT_SHORT
    dt_vec[N_SHORT:] = DT_LONG
    return dt_vec


def set_acados_model():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    acados_models_dir = "./acados_models"
    safe_mkdir_recursive(os.path.join(os.getcwd(), acados_models_dir))
    acados_source_path = os.environ["ACADOS_SOURCE_DIR"]
    sys.path.insert(0, acados_source_path)

    ocp = AcadosOcp()
    model = export_lon_model()
    ocp.model = model

    nx = model.x.size()[0]  # 4: [s, v, a, a_cmd]
    nu = model.u.size()[0]  # 1: [j_cmd]
    ny = nx + nu            # 5: tracking cost dimension
    ny_e = nx               # 4: terminal cost dimension
    np_val = model.p.size()[0]  # 1: [tau]

    # Time steps (non-uniform)
    dt_vec = build_time_steps()
    tf = float(np.sum(dt_vec))

    ocp.dims.N = N_STEPS
    ocp.solver_options.time_steps = dt_vec
    ocp.solver_options.tf = tf

    # ---- Cost: LINEAR_LS type ----
    # y = Vx * x + Vu * u, cost = (y - yref)^T W (y - yref)
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Stage cost matrices
    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx, 0] = 1.0

    Q = np.diag([W_S, W_V, W_A, W_A_CMD])
    R = np.diag([W_J_CMD])
    W = scipy.linalg.block_diag(Q, R)

    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.W = W
    ocp.cost.yref = np.zeros(ny)

    # Initial stage cost (same as intermediate)
    ocp.cost.Vx_0 = Vx
    ocp.cost.Vu_0 = Vu
    ocp.cost.W_0 = W
    ocp.cost.yref_0 = np.zeros(ny)

    # Terminal cost
    Vx_e = np.eye(ny_e, nx)
    Q_e = np.diag([W_S, W_V, W_A, W_A_CMD]) * 10.0  # heavier terminal weight
    ocp.cost.Vx_e = Vx_e
    ocp.cost.W_e = Q_e
    ocp.cost.yref_e = np.zeros(ny_e)

    # ---- Constraints ----
    # Initial state constraints (all 4 states fixed at start)
    ocp.constraints.x0 = np.zeros(nx)

    # State bounds for intermediate stages: constrain s, v, a_cmd (indices 0, 1, 3)
    ocp.constraints.idxbx = np.array([0, 1, 3])
    ocp.constraints.lbx = np.array([S_LOWER, V_LOWER, A_CMD_LOWER])
    ocp.constraints.ubx = np.array([S_UPPER, V_UPPER, A_CMD_UPPER])

    # Soft state box constraints: only s (index 0 in idxbx) is softened
    ocp.constraints.idxsbx = np.array([0])

    # L2 penalty (quadratic slack cost)
    ocp.cost.Zl = np.array([ZL_S])
    ocp.cost.Zu = np.array([ZU_S])
    # L1 penalty (linear slack cost)
    ocp.cost.zl = np.array([ZL1_S])
    ocp.cost.zu = np.array([ZU1_S])

    # Control bounds: j_cmd (hard constraint, no slack)
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([J_CMD_LOWER])
    ocp.constraints.ubu = np.array([J_CMD_UPPER])

    # Terminal state bounds (also with soft constraints)
    ocp.constraints.idxbx_e = np.array([0, 1, 3])
    ocp.constraints.lbx_e = np.array([S_LOWER, V_LOWER, A_CMD_LOWER])
    ocp.constraints.ubx_e = np.array([S_UPPER, V_UPPER, A_CMD_UPPER])

    # Soft terminal state box constraints: only s
    ocp.constraints.idxsbx_e = np.array([0])
    ocp.cost.Zl_e = np.array([ZL_S])
    ocp.cost.Zu_e = np.array([ZU_S])
    ocp.cost.zl_e = np.array([ZL1_S])
    ocp.cost.zu_e = np.array([ZU1_S])

    # ---- Parameters default value ----
    ocp.parameter_values = np.array([TAU_A])

    # ---- Solver options ----
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.hpipm_mode = "BALANCE"
    ocp.solver_options.hpipm_mode = "SPEED_ABS"
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.nlp_solver_warm_start_first_qp = True
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.print_level = 0
    ocp.solver_options.tol = 1e-5
    ocp.solver_options.N_horizon = N_STEPS

    json_file = os.path.join("./" + model.name + "_acados_ocp.json")
    acados_solver = AcadosOcpSolver(ocp, json_file=json_file)
    return acados_solver, ocp


def get_reference(N):
    """
    Generate reference trajectory.
    yref shape: (ny,) = (5,) -> [s_ref, v_ref, a_ref, a_cmd_ref, j_cmd_ref]
    yref_e shape: (ny_e,) = (4,) -> [s_ref, v_ref, a_ref, a_cmd_ref]
    """
    s_ref = 40.0
    v_ref = 0.0
    a_ref = 0.0
    a_cmd_ref = 0.0
    j_cmd_ref = 0.0

    y_ref = []
    for i in range(N):
        ref = np.array([s_ref, v_ref, a_ref, a_cmd_ref, j_cmd_ref])
        y_ref.append(ref)

    y_ref_e = np.array([s_ref, v_ref, a_ref, a_cmd_ref])
    return y_ref, y_ref_e


def get_bounds(N):
    """
    Get per-step bounds for states and controls.
    Returns lists of bounds that can be set per shooting node.
    State bounds: [s, v, a_cmd] (idxbx = [0, 1, 3])
    Control bounds: [j_cmd]
    """
    x_lb_list = []
    x_ub_list = []
    u_lb_list = []
    u_ub_list = []

    for i in range(N + 1):
        if i == 0:
            # Initial constraints are handled by ocp.constraints.x0
            x_lb_list.append(np.array([S_LOWER, V_LOWER, A_CMD_LOWER]))
            x_ub_list.append(np.array([S_UPPER, V_UPPER, A_CMD_UPPER]))
        else:
            x_lb_list.append(np.array([S_LOWER, V_LOWER, A_CMD_LOWER]))
            x_ub_list.append(np.array([S_UPPER, V_UPPER, A_CMD_UPPER]))

        if i < N:
            u_lb_list.append(np.array([J_CMD_LOWER]))
            u_ub_list.append(np.array([J_CMD_UPPER]))

    return x_lb_list, x_ub_list, u_lb_list, u_ub_list


def plot_results(acados_solver, x_traj, u_traj, dt_vec):
    """
    Plot states and control trajectories with per-step references and bounds
    retrieved directly from the acados solver.
    """
    x_arr = np.array(x_traj)
    u_arr = np.array(u_traj)
    N = len(u_traj)

    # Retrieve per-step references and bounds from solver
    yref_list = [acados_solver.cost_get(i, "yref") for i in range(N)]
    yref_e = acados_solver.cost_get(N, "yref")

    # idxbx = [0, 1, 3] -> lbx/ubx maps to [s, v, a_cmd]
    lbx_list = [acados_solver.constraints_get(i, "lbx") for i in range(1, N + 1)]
    ubx_list = [acados_solver.constraints_get(i, "ubx") for i in range(1, N + 1)]
    # idxbu = [0] -> lbu/ubu maps to [j_cmd]
    lbu_list = [acados_solver.constraints_get(i, "lbu") for i in range(N)]
    ubu_list = [acados_solver.constraints_get(i, "ubu") for i in range(N)]

    # Extract per-step references: yref = [s_ref, v_ref, a_ref, a_cmd_ref, j_cmd_ref]
    s_ref = [yref_list[i][0] for i in range(N)]
    v_ref = [yref_list[i][1] for i in range(N)]
    a_cmd_ref = [yref_list[i][3] for i in range(N)]

    # Extract per-step bounds (mask out values with abs >= 1e7 as "no constraint")
    BOUND_THRESH = 1e7

    def mask_bounds(vals):
        """Replace values with abs >= threshold with NaN so they won't be plotted."""
        return [v if abs(v) < BOUND_THRESH else np.nan for v in vals]

    s_lb = mask_bounds([lbx_list[i][0] for i in range(N)])
    s_ub = mask_bounds([ubx_list[i][0] for i in range(N)])
    v_lb = mask_bounds([lbx_list[i][1] for i in range(N)])
    v_ub = mask_bounds([ubx_list[i][1] for i in range(N)])
    a_cmd_lb = mask_bounds([lbx_list[i][2] for i in range(N)])
    a_cmd_ub = mask_bounds([ubx_list[i][2] for i in range(N)])
    j_cmd_lb = mask_bounds([lbu_list[i][0] for i in range(N)])
    j_cmd_ub = mask_bounds([ubu_list[i][0] for i in range(N)])

    # Build time vector from non-uniform dt
    t_x = np.zeros(N + 1)
    for i in range(N):
        t_x[i + 1] = t_x[i] + dt_vec[i]
    t_u = t_x[:N]

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    # s
    axes[0].plot(t_x, x_arr[:, 0], 'b-', linewidth=2, label='s')
    axes[0].step(t_u, s_ref, 'm--', linewidth=1.5, where='post', label='s_ref')
    axes[0].step(t_u, s_ub, 'r--', linewidth=1, where='post', label='s_ub')
    axes[0].step(t_u, s_lb, 'g--', linewidth=1, where='post', label='s_lb')
    axes[0].set_ylabel('s (m)')
    axes[0].set_title('Distance')
    axes[0].legend()
    axes[0].grid(True)

    # v
    axes[1].plot(t_x, x_arr[:, 1], 'b-', linewidth=2, label='v')
    axes[1].step(t_u, v_ref, 'm--', linewidth=1.5, where='post', label='v_ref')
    axes[1].step(t_u, v_lb, 'r--', linewidth=1, where='post', label='v_lb')
    axes[1].step(t_u, v_ub, 'g--', linewidth=1, where='post', label='v_ub')
    axes[1].set_ylabel('v (m/s)')
    axes[1].set_title('Velocity')
    axes[1].legend()
    axes[1].grid(True)

    # a
    axes[2].plot(t_x, x_arr[:, 2], 'b-', linewidth=2, label='a (actual)')
    axes[2].plot(t_x, x_arr[:, 3], 'g--', linewidth=1.5, label='a_cmd')
    axes[2].set_ylabel('a (m/s²)')
    axes[2].set_title('Acceleration (actual vs command)')
    axes[2].legend()
    axes[2].grid(True)

    # a_cmd
    axes[3].plot(t_x, x_arr[:, 3], 'b-', linewidth=2, label='a_cmd')
    axes[3].step(t_u, a_cmd_lb, 'r--', linewidth=1, where='post', label='a_cmd_lb')
    axes[3].step(t_u, a_cmd_ub, 'r--', linewidth=1, where='post', label='a_cmd_ub')
    axes[3].step(t_u, a_cmd_ref, 'm--', linewidth=1.5, where='post', label='a_cmd_ref')
    axes[3].set_ylabel('a_cmd (m/s²)')
    axes[3].set_title('Acceleration Command')
    axes[3].legend()
    axes[3].grid(True)

    # j_cmd (control)
    axes[4].step(t_u, u_arr[:, 0], 'b-', linewidth=2, where='post', label='j_cmd')
    axes[4].step(t_u, j_cmd_lb, 'r--', linewidth=1, where='post', label='j_cmd_lb')
    axes[4].step(t_u, j_cmd_ub, 'r--', linewidth=1, where='post', label='j_cmd_ub')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('j_cmd (m/s³)')
    axes[4].set_title('Jerk Command (Control)')
    axes[4].legend()
    axes[4].grid(True)

    plt.tight_layout()


def compute_continuous_matrices(tau: float):
    """
    Continuous-time linear model:
        x = [s, v, a, a_cmd], u = [j_cmd]
        s_dot     = v
        v_dot     = a
        a_dot     = (a_cmd - a) / tau
        a_cmd_dot = j_cmd
    """
    nx = 4
    nu = 1
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    A[0, 1] = 1.0
    A[1, 2] = 1.0
    A[2, 2] = -1.0 / tau
    A[2, 3] = 1.0 / tau
    B[3, 0] = 1.0

    return A, B


def c2d_zoh(A, B, dt: float):
    """
    Exact ZOH discretization via matrix exponential:
        [A_d  B_d] = expm([A B; 0 0] * dt)
    """
    nx = A.shape[0]
    nu = B.shape[1]
    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = A
    M[:nx, nx:nx + nu] = B
    Md = scipy.linalg.expm(M * dt)
    A_d = Md[:nx, :nx]
    B_d = Md[:nx, nx:nx + nu]
    return A_d, B_d


def build_kkt_matrix(A_list, Bu_list, Q, R, Q_e, N, nx, nu, sigma=0.0):
    """
    Build the KKT matrix of the OCP QP (equality constraints only),
    optionally with a uniform diagonal 'sigma' term as an approximation of
    IPM barrier diagonal contribution.

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
        KKT[ix:ix + nx, ix:ix + nx] = Q + sigma * np.eye(nx)
        KKT[iu:iu + nu, iu:iu + nu] = R + sigma * np.eye(nu)

    ix_N = x_idx(N)
    KKT[ix_N:ix_N + nx, ix_N:ix_N + nx] = Q_e + sigma * np.eye(nx)

    for k in range(N):
        row = nz + k * nx
        ix_k = x_idx(k)
        iu_k = u_idx(k)
        ix_k1 = x_idx(k + 1)

        KKT[row:row + nx, ix_k:ix_k + nx] = A_list[k]
        KKT[ix_k:ix_k + nx, row:row + nx] = A_list[k].T

        KKT[row:row + nx, iu_k:iu_k + nu] = Bu_list[k]
        KKT[iu_k:iu_k + nu, row:row + nx] = Bu_list[k].T

        KKT[row:row + nx, ix_k1:ix_k1 + nx] = -np.eye(nx)
        KKT[ix_k1:ix_k1 + nx, row:row + nx] = -np.eye(nx)

    svs = np.linalg.svd(KKT, compute_uv=False)
    cond = svs[0] / svs[svs > 1e-15][-1] if np.any(svs > 1e-15) else np.inf
    return KKT, cond


def print_dynamics_and_kkt_diagnostics(acados_solver, N, time_steps):
    """
    Print per-stage continuous/discrete A matrices and eigenvalues, then
    overall KKT condition number. Prints every 10th stage + last 5 stages.

    Also attempts to print A/B that acados QP used (get_from_qp_in) if available.
    """
    nx = 4
    nu = 1

    print_stages = set(range(0, N, 10))
    print_stages.update(range(max(0, N - 4), N))
    print_stages = sorted(print_stages)

    A_disc_list = []
    Bu_disc_list = []

    logger.info("=" * 72)
    logger.info("  DYNAMICS & EIGENVALUE DIAGNOSTICS (lon_pwj)")
    logger.info("=" * 72)

    for i in range(N):
        dt = float(time_steps[i])
        tau = TAU_A
        try:
            p_i = acados_solver.get(i, "p")
            if p_i is not None and len(p_i) >= 1:
                tau = float(np.array(p_i).reshape((-1,))[0])
        except Exception:
            pass

        A_con, Bu_con = compute_continuous_matrices(tau)
        A_d, Bu_d = c2d_zoh(A_con, Bu_con, dt)
        A_disc_list.append(A_d)
        Bu_disc_list.append(Bu_d)

        if i in print_stages:
            eig_con = np.linalg.eigvals(A_con)
            eig_disc = np.linalg.eigvals(A_d)
            lam_fast = eig_con[np.argmin(np.real(eig_con))]
            lam_h = abs(np.real(lam_fast)) * dt
            z = np.real(lam_fast) * dt
            rk4_growth = abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)

            logger.info("-" * 72)
            logger.info("Stage %2d  |  tau=%.4f s  dt=%.4f s", i, tau, dt)
            logger.info("  A_con (continuous):")
            for row in range(nx):
                logger.info("    [%s]",
                            "  ".join(f"{A_con[row, c]:12.4e}" for c in range(nx)))
            logger.info("  eig(A_con)  = [%s]",
                        ", ".join(f"{e.real:+.6f}{e.imag:+.6f}j" for e in eig_con))
            logger.info("  A_disc (ZOH exact):")
            for row in range(nx):
                logger.info("    [%s]",
                            "  ".join(f"{A_d[row, c]:12.4e}" for c in range(nx)))
            logger.info("  eig(A_disc) = [%s]",
                        ", ".join(f"{e.real:+.6f}{e.imag:+.6f}j" for e in eig_disc))

            try:
                A_qp = acados_solver.get_from_qp_in(i, "A")
                B_qp = acados_solver.get_from_qp_in(i, "B")
                diff_A = float(np.max(np.abs(A_qp - A_d)))
                diff_B = float(np.max(np.abs(B_qp - Bu_d)))
                eig_qp = np.linalg.eigvals(A_qp)
                logger.info("  A_acados (from QP):")
                for row in range(nx):
                    logger.info("    [%s]",
                                "  ".join(f"{A_qp[row, c]:12.4e}" for c in range(nx)))
                logger.info("  eig(A_acados) = [%s]",
                            ", ".join(f"{e.real:+.6f}{e.imag:+.6f}j" for e in eig_qp))
                logger.info("  max|A_acados - A_zoh| = %.4e  %s", diff_A,
                            "" if diff_A < 1e-6 else ">>> LARGE DIFF <<<")
                logger.info("  max|B_acados - B_zoh| = %.4e  %s", diff_B,
                            "" if diff_B < 1e-6 else ">>> LARGE DIFF <<<")
            except Exception as exc:
                logger.warning("  A/B from QP not available: %s", exc)

            logger.info("  |lambda_fast * dt| = %.3f  (RK4 limit=2.785)  RK4_growth=%.2e  %s",
                        lam_h, rk4_growth,
                        "STABLE" if lam_h < 2.785 else ">>> UNSTABLE <<<")

    Q_cost = np.diag([W_S, W_V, W_A, W_A_CMD])
    R_cost = np.diag([W_J_CMD])
    Q_e = np.diag([W_S, W_V, W_A, W_A_CMD]) * 10.0

    logger.info("=" * 72)
    logger.info("  KKT CONDITION NUMBER (based on ZOH A)")
    logger.info("=" * 72)
    for sigma in [0.0, 1e-4, 1e-2]:
        _, cond = build_kkt_matrix(
            A_disc_list, Bu_disc_list, Q_cost, R_cost, Q_e, N, nx, nu,
            sigma=sigma)
        logger.info("  sigma=%.0e  =>  cond(KKT) = %.4e", sigma, cond)

    try:
        A_qp_list = []
        Bu_qp_list = []
        for i in range(N):
            A_qp_list.append(acados_solver.get_from_qp_in(i, "A"))
            Bu_qp_list.append(acados_solver.get_from_qp_in(i, "B"))
        logger.info("  KKT CONDITION NUMBER (based on acados QP A)")
        for sigma in [0.0, 1e-4, 1e-2]:
            _, cond_qp = build_kkt_matrix(
                A_qp_list, Bu_qp_list, Q_cost, R_cost, Q_e, N, nx, nu,
                sigma=sigma)
            logger.info("  sigma=%.0e  =>  cond(KKT_qp) = %.4e", sigma, cond_qp)
    except Exception as exc:
        logger.warning("  KKT from acados QP A not available: %s", exc)

    logger.info("=" * 72)

def _safe_get_stats_ms(acados_solver, name: str) -> float:
    try:
        v = acados_solver.get_stats(name)
        if v is None:
            return float("nan")
        return float(v) * 1000.0
    except Exception:
        return float("nan")


def run_single_solve_lon(acados_solver, N, time_steps, x0, tau,
                         y_ref, y_ref_e,
                         x_lb_list, x_ub_list, u_lb_list, u_ub_list,
                         do_diagnostics=False):
    """
    One solve call for lon_pwj with per-stage overriding of:
      - time_steps
      - x0
      - tau (parameter)
      - yref / bounds
    """
    # allow changing time grid without re-exporting
    try:
        acados_solver.set_new_time_steps(np.array(time_steps, dtype=float))
    except Exception:
        pass

    # stage 0 x0
    acados_solver.constraints_set(0, "lbx", x0)
    acados_solver.constraints_set(0, "ubx", x0)

    # set per-stage params/refs/bounds
    for i in range(N + 1):
        acados_solver.set(i, "p", np.array([tau], dtype=float))
        if i == 0:
            acados_solver.cost_set(i, "yref", y_ref[0])
            if i < N:
                acados_solver.constraints_set(i, "lbu", u_lb_list[i])
                acados_solver.constraints_set(i, "ubu", u_ub_list[i])
        elif i < N:
            acados_solver.cost_set(i, "yref", y_ref[i])
            acados_solver.constraints_set(i, "lbx", x_lb_list[i])
            acados_solver.constraints_set(i, "ubx", x_ub_list[i])
            acados_solver.constraints_set(i, "lbu", u_lb_list[i])
            acados_solver.constraints_set(i, "ubu", u_ub_list[i])
        else:
            acados_solver.cost_set(i, "yref", y_ref_e)
            acados_solver.constraints_set(i, "lbx", x_lb_list[i])
            acados_solver.constraints_set(i, "ubx", x_ub_list[i])

    # warm start: simple x0 propagation
    acados_solver.reset()
    for i in range(N + 1):
        acados_solver.set(i, "x", x0.copy())
    for i in range(N):
        acados_solver.set(i, "u", np.zeros((1,)))

    t0 = time.perf_counter()
    status = acados_solver.solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    x_sol = [acados_solver.get(i, "x") for i in range(N + 1)]
    u_sol = [acados_solver.get(i, "u") for i in range(N)]

    res = {
        "status": int(status),
        "elapsed_ms": float(elapsed_ms),
        "time_tot_ms": _safe_get_stats_ms(acados_solver, "time_tot"),
        "time_qp_ms": _safe_get_stats_ms(acados_solver, "time_qp"),
        "time_lin_ms": _safe_get_stats_ms(acados_solver, "time_lin"),
        "sqp_iter": int(acados_solver.get_stats("sqp_iter")) if hasattr(acados_solver, "get_stats") else -1,
        "x0": x0.copy(),
        "tau": float(tau),
        "time_steps": np.array(time_steps, dtype=float),
        "x_sol": x_sol,
        "u_sol": u_sol,
        "y_ref": y_ref,
        "y_ref_e": y_ref_e,
        "x_lb_list": x_lb_list,
        "x_ub_list": x_ub_list,
        "u_lb_list": u_lb_list,
        "u_ub_list": u_ub_list,
    }

    if do_diagnostics:
        logger.info("Diagnostics for one call:")
        print_dynamics_and_kkt_diagnostics(acados_solver, N, np.array(time_steps, dtype=float))

    return res


def run_benchmark(n_calls=1, plot_call_idx=17):
    """
    Run lon_pwj solver n_calls times with randomized inputs.

    Each call varies:
      - time_steps: first 20 = 0.02s, one middle step in [0.05, 0.2], rest = 0.2s
      - x0: small perturbations
      - yref: target s_ref and v_ref randomized
      - tau: inertia time constant randomized
      - bounds: s upper and jerk bounds randomized slightly

    Args:
        n_calls: number of solve calls
        plot_call_idx: which call to plot (0-based); set -1 to skip plotting
    """
    rng = np.random.default_rng(seed=42)
    N = N_STEPS

    base_time_steps = build_time_steps()
    base_tf = float(np.sum(base_time_steps))
    acados_solver, _ = set_acados_model()

    results = []

    for k in range(n_calls):
        # --- 1) Randomize time_steps ---
        dt_mid = float(rng.uniform(0.05, 0.2))
        ts = np.concatenate([
            np.full(N_SHORT, DT_SHORT),
            np.array([dt_mid]),
            np.full(N_LONG - 1, DT_LONG),
        ])
        if len(ts) != N:
            ts = base_time_steps.copy()

        # --- 2) Randomize x0 ---
        s0 = float(rng.uniform(-0.5, 0.5))
        v0 = float(rng.uniform(0.0, 30.0))
        a0 = float(rng.uniform(-2.0, 2.0))
        a_cmd0 = float(rng.uniform(-2.0, 2.0))
        x0 = np.array([s0, v0, a0, a_cmd0], dtype=float)

        # --- 3) Randomize tau ---
        tau = float(np.clip(TAU_A * (1.0 + rng.uniform(-0.5, 0.5)), 0.02, 0.5))

        # --- 4) Randomize reference ---
        s_ref = float(rng.uniform(10.0, 60.0))
        v_ref = float(rng.uniform(0.0, 5.0))
        a_ref = 0.0
        a_cmd_ref = 0.0
        j_ref = 0.0
        y_ref = [np.array([s_ref, v_ref, a_ref, a_cmd_ref, j_ref], dtype=float) for _ in range(N)]
        y_ref_e = np.array([s_ref, v_ref, a_ref, a_cmd_ref], dtype=float)

        # --- 5) Randomize constraints (per-step) ---
        s_upper = float(S_UPPER + rng.uniform(-1.0, 1.0))
        s_lower = float(S_LOWER)  # treated as -inf in plotting
        v_lower = float(V_LOWER)
        v_upper = float(V_UPPER)
        a_cmd_lower = float(A_CMD_LOWER)
        a_cmd_upper = float(A_CMD_UPPER)
        j_lower = float(J_CMD_LOWER * (1.0 + rng.uniform(-0.2, 0.2)))
        j_upper = float(J_CMD_UPPER * (1.0 + rng.uniform(-0.2, 0.2)))

        x_lb_list = [np.array([s_lower, v_lower, a_cmd_lower], dtype=float) for _ in range(N + 1)]
        x_ub_list = [np.array([s_upper, v_upper, a_cmd_upper], dtype=float) for _ in range(N + 1)]
        u_lb_list = [np.array([j_lower], dtype=float) for _ in range(N)]
        u_ub_list = [np.array([j_upper], dtype=float) for _ in range(N)]

        # --- 6) Solve ---
        res = run_single_solve_lon(
            acados_solver, N, ts, x0, tau,
            y_ref, y_ref_e,
            x_lb_list, x_ub_list, u_lb_list, u_ub_list,
            do_diagnostics=(k==plot_call_idx),
        )
        results.append(res)

        is_fail = res["status"] != 0
        verbose = k < 2 or k == n_calls - 1 or is_fail
        tag = "FAIL" if is_fail else "OK"
        summary = (f"[{k:4d}] {tag}  status={res['status']}  "
                   f"sqp_iter={res['sqp_iter']}  "
                   f"wall={res['elapsed_ms']:.3f}  "
                   f"tot={res['time_tot_ms']:.3f}  "
                   f"qp={res['time_qp_ms']:.3f} ms")
        if verbose:
            print(summary)
            print(f"  x0: s={x0[0]:.3f}  v={x0[1]:.3f}  a={x0[2]:.3f}  a_cmd={x0[3]:.3f}")
            print(f"  tau={tau:.4f}  ref: s_ref={s_ref:.2f}  v_ref={v_ref:.2f}")
            print(f"  dt_mid={dt_mid:.3f}  s_ub={s_upper:.3f}  "
                  f"j=[{j_lower:.2f},{j_upper:.2f}]")

    # --- Summary statistics ---
    elapsed_arr = np.array([r["elapsed_ms"] for r in results], dtype=float)
    tot_arr = np.array([r["time_tot_ms"] for r in results], dtype=float)
    qp_arr = np.array([r["time_qp_ms"] for r in results], dtype=float)
    iter_arr = np.array([r["sqp_iter"] for r in results], dtype=float)
    status_arr = np.array([r["status"] for r in results], dtype=int)

    n_fail = int(np.sum(status_arr != 0))
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY  ({n_calls} calls)")
    print("=" * 70)
    print(f"  Failures:  {n_fail} / {n_calls}  ({100*n_fail/n_calls:.1f}%)")
    print(f"  elapsed   (Python wall): mean={np.mean(elapsed_arr):.3f}  "
          f"median={np.median(elapsed_arr):.3f}  "
          f"p95={np.percentile(elapsed_arr, 95):.3f}  "
          f"max={np.max(elapsed_arr):.3f} ms")
    if np.all(np.isfinite(tot_arr)):
        print(f"  time_tot  (acados C):    mean={np.mean(tot_arr):.3f}  "
              f"median={np.median(tot_arr):.3f}  "
              f"p95={np.percentile(tot_arr, 95):.3f}  "
              f"max={np.max(tot_arr):.3f} ms")
    if np.all(np.isfinite(qp_arr)):
        print(f"  time_qp:                 mean={np.mean(qp_arr):.3f}  "
              f"median={np.median(qp_arr):.3f}  "
              f"p95={np.percentile(qp_arr, 95):.3f}  "
              f"max={np.max(qp_arr):.3f} ms")
    print(f"  sqp_iter:                mean={np.mean(iter_arr):.2f}  "
          f"median={np.median(iter_arr):.0f}  "
          f"max={np.max(iter_arr):.0f}")
    print("=" * 70)

    # --- Timing distribution plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"lon_pwj Benchmark ({n_calls} calls)", fontsize=14)

    axes[0, 0].hist(elapsed_arr, bins=40, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.median(elapsed_arr), color='r', linestyle='--',
                       label=f"median={np.median(elapsed_arr):.3f}")
    axes[0, 0].set_xlabel('elapsed (ms)')
    axes[0, 0].set_ylabel('count')
    axes[0, 0].set_title('Python wall time')
    axes[0, 0].legend()

    if np.any(np.isfinite(tot_arr)):
        axes[0, 1].hist(tot_arr[np.isfinite(tot_arr)], bins=40,
                        edgecolor='black', alpha=0.7, color='tab:green')
        axes[0, 1].axvline(np.nanmedian(tot_arr), color='r', linestyle='--',
                           label=f"median={np.nanmedian(tot_arr):.3f}")
        axes[0, 1].set_xlabel('time_tot (ms)')
        axes[0, 1].set_ylabel('count')
        axes[0, 1].set_title('Total solve time (acados C)')
        axes[0, 1].legend()

    axes[1, 0].plot(iter_arr, 'o-', markersize=2, linewidth=0.5)
    axes[1, 0].set_xlabel('call index')
    axes[1, 0].set_ylabel('sqp_iter')
    axes[1, 0].set_title('SQP iterations per call')

    colors = ['tab:blue' if s == 0 else 'tab:red' for s in status_arr]
    axes[1, 1].bar(range(n_calls), elapsed_arr, color=colors, width=1.0)
    axes[1, 1].set_xlabel('call index')
    axes[1, 1].set_ylabel('elapsed (ms)')
    axes[1, 1].set_title('Per-call wall time (red = failure)')

    plt.tight_layout()

    # --- Plot the selected call ---
    if 0 <= plot_call_idx < n_calls:
        r = results[plot_call_idx]
        print(
            f"\nPlotting call #{plot_call_idx}:  status={r['status']}  "
            f"elapsed={r['elapsed_ms']:.3f} ms  sqp_iter={r['sqp_iter']}"
        )
        x_traj = r["x_sol"]
        u_traj = r["u_sol"]
        plot_results(acados_solver, x_traj, u_traj, r["time_steps"])

    return results


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


if __name__ == "__main__":
    matplotlib.set_loglevel("warning")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true",
                        help="Run randomized benchmark instead of single solve")

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(n_calls=100, plot_call_idx=17)
        plt.show()
        raise SystemExit(0)

    # Build solver
    acados_solver, ocp = set_acados_model()
    dt_vec = build_time_steps()
    tf = float(np.sum(dt_vec))

    # Initial state: [s0, v0, a0, a_cmd0]
    s0 = 0.0
    v0 = 25.0
    a0 = 0.0
    a_cmd0 = 0.0
    x0 = np.array([s0, v0, a0, a_cmd0])

    # Set initial state
    acados_solver.constraints_set(0, "lbx", x0)
    acados_solver.constraints_set(0, "ubx", x0)

    # Get references and bounds
    y_ref, y_ref_e = get_reference(N_STEPS)
    x_lb, x_ub, u_lb, u_ub = get_bounds(N_STEPS)

    # Set per-step parameters, references, and bounds
    for i in range(N_STEPS + 1):
        acados_solver.set(i, 'p', np.array([TAU_A]))

        if i == 0:
            # Stage 0: x0 constraint has dimension nx=4 (all states fixed)
            acados_solver.cost_set(i, "yref", y_ref[i])
            acados_solver.constraints_set(i, "lbx", x0)
            acados_solver.constraints_set(i, "ubx", x0)
            acados_solver.constraints_set(i, "lbu", u_lb[i])
            acados_solver.constraints_set(i, "ubu", u_ub[i])
        elif i < N_STEPS:
            acados_solver.cost_set(i, "yref", y_ref[i])
            acados_solver.constraints_set(i, "lbx", x_lb[i])
            acados_solver.constraints_set(i, "ubx", x_ub[i])
            acados_solver.constraints_set(i, "lbu", u_lb[i])
            acados_solver.constraints_set(i, "ubu", u_ub[i])
        else:
            acados_solver.cost_set(i, "yref", y_ref_e)
            acados_solver.constraints_set(i, "lbx", x_lb[i])
            acados_solver.constraints_set(i, "ubx", x_ub[i])

    # Solve
    start_time = time.perf_counter()
    status = acados_solver.solve()
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) * 1000

    # Get solver stats
    sqp_iter = acados_solver.get_stats("sqp_iter")
    print(f"Solver status: {status}, SQP iterations: {sqp_iter}, "
          f"Elapsed time: {elapsed_time:.2f} ms")

    if status != 0:
        print(f"WARNING: Solver returned status {status}")

    # Extract solution
    x_traj = [acados_solver.get(i, "x") for i in range(N_STEPS + 1)]
    u_traj = [acados_solver.get(i, "u") for i in range(N_STEPS)]

    print(f"Initial state: {x_traj[0]}")
    print(f"Final state:   {x_traj[-1]}")
    print(f"Total horizon: {tf:.3f} s")
    print(f"Time steps: first {N_SHORT} x {DT_SHORT}s + last {N_LONG} x {DT_LONG}s")

    print_dynamics_and_kkt_diagnostics(acados_solver, N_STEPS, dt_vec)

    # Plot
    plot_results(acados_solver, x_traj, u_traj, dt_vec)
    plt.show()
