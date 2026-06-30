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

    # Plot
    plot_results(acados_solver, x_traj, u_traj, dt_vec)
    plt.show()
