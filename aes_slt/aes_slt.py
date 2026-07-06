"""
AES-SLT: Frenet 坐标系轨迹优化 (acados)

State:   x = [l, delta_theta, kappa, s, vx, ax]
Control: u = [dk, jx]
Params:  p = [kr, ax_max, ay_max, l_ref, vx_ref]

Friction ellipse handled via relaxed log-barrier in the cost:
    g = ax^2/ax_max^2 + (vx^2*kappa)^2/ay_max^2 - 1
    B(g) = -mu*ln(-g)                           if g <= -delta
    B(g) = mu*(t^2/2 + t - ln(delta))           if g >  -delta,  t=(g+delta)/delta
"""
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import numpy as np
import casadi as ca
import os
import time
import logging
import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
)

# ────────────────────────── indices ──────────────────────────
IDX_L      = 0
IDX_DTHETA = 1
IDX_KAPPA  = 2
IDX_S      = 3
IDX_VX     = 4
IDX_AX     = 5
NX = 6

IDX_DK = 0
IDX_JX = 1
NU = 2

IDX_P_KR     = 0
IDX_P_AX_MAX = 1
IDX_P_AY_MAX = 2
IDX_P_L_REF  = 3
IDX_P_VX_REF = 4
NP = 5

# ────────────────────────── default bounds ───────────────────
L_MIN  = -4.0;    L_MAX  = 4.0
S_MAX  = 200.0
VX_MAX = 40.0
DK_MIN = -0.1;    DK_MAX = 0.1
JX_MIN = -10.0;   JX_MAX = 10.0

# ────────────────────────── slack weights ────────────────────
W_SLACK_L_L1 = 0.0;   W_SLACK_L_L2 = 10.0
W_SLACK_S_L1 = 0.0;   W_SLACK_S_L2 = 10.0

# ────────────────────────── cost weights ─────────────────────
W_L      = 5.0
W_DTHETA = 1.0
W_KAPPA  = 10.0
W_VX     = 2.0
W_AX     = 1.0
W_JX     = 1.0
W_DK     = 10.0

# ────────────────────────── barrier parameters ───────────────
BARRIER_MU    = 10.0   # barrier weight
BARRIER_DELTA = 0.05   # relaxation zone width

# ────────────────────────── default parameters ───────────────
DEFAULT_KR     = 0.0
DEFAULT_AX_MAX = 5.0
DEFAULT_AY_MAX = 4.0
L_REF  = 3.0
VX_REF = 10.0

# ═══════════════════════════════════════════════════════════════
#  Relaxed log-barrier (CasADi)
# ═══════════════════════════════════════════════════════════════

def _relaxed_log_barrier(g, mu, delta):
    """
    C1-continuous relaxed log-barrier for constraint g <= 0.

    g < -delta : standard barrier  -mu * ln(-g)
    g >= -delta: quadratic extension (keeps gradient continuous)

    Returns a scalar CasADi expression.
    """
    t = (g + delta) / delta
    return ca.if_else(
        g <= -delta,
        -mu * ca.log(-g),
        mu * (t**2 / 2.0 + t - ca.log(delta)),
    )

# ═══════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════

def export_frenet_model(barrier_mu=BARRIER_MU, barrier_delta=BARRIER_DELTA):
    """Build Frenet model with EXTERNAL cost containing relaxed log-barrier."""
    model_name = "aes_slt"

    l      = ca.SX.sym("l")
    dtheta = ca.SX.sym("delta_theta")
    kappa  = ca.SX.sym("kappa")
    s      = ca.SX.sym("s")
    vx     = ca.SX.sym("vx")
    ax     = ca.SX.sym("ax")
    x = ca.vertcat(l, dtheta, kappa, s, vx, ax)

    dk = ca.SX.sym("dk")
    jx = ca.SX.sym("jx")
    u = ca.vertcat(dk, jx)

    xdot = ca.SX.sym("xdot", NX)

    kr       = ca.SX.sym("kr")
    ax_max_p = ca.SX.sym("ax_max")
    ay_max_p = ca.SX.sym("ay_max")
    l_ref_p  = ca.SX.sym("l_ref")
    vx_ref_p = ca.SX.sym("vx_ref")
    p = ca.vertcat(kr, ax_max_p, ay_max_p, l_ref_p, vx_ref_p)

    # ── dynamics ──
    denom = 1.0 - kr * l
    f_expl = ca.vertcat(
        vx * dtheta,
        kappa * vx - kr * vx / denom,
        dk,
        vx / denom,
        ax,
        jx,
    )
    f_impl = xdot - f_expl

    # ── friction ellipse barrier ──
    ay = vx**2 * kappa
    g = ax**2 / ax_max_p**2 + ay**2 / ay_max_p**2 - 1.0
    barrier = _relaxed_log_barrier(g, barrier_mu, barrier_delta)

    # ── quadratic tracking cost ──
    cost_states = (W_L * (l - l_ref_p)**2 +
                   W_DTHETA * dtheta**2 +
                   W_KAPPA * kappa**2 +
                   W_VX * (vx - vx_ref_p)**2 +
                   W_AX * ax**2)

    cost_controls = W_DK * dk**2 + W_JX * jx**2

    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.x    = x
    model.xdot = xdot
    model.u    = u
    model.p    = p
    model.z    = ca.vertcat([])
    model.name = model_name

    model.cost_expr_ext_cost   = cost_states + cost_controls + barrier
    model.cost_expr_ext_cost_e = cost_states + barrier

    return model

# ═══════════════════════════════════════════════════════════════
#  OCP setup
# ═══════════════════════════════════════════════════════════════

def set_acados_ocp(N: int, tf: float,
                   l_min=L_MIN, l_max=L_MAX, s_max=S_MAX,
                   barrier_mu=BARRIER_MU, barrier_delta=BARRIER_DELTA):
    """Build AcadosOcpSolver with EXTERNAL cost (barrier in cost)."""
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    ocp = AcadosOcp()
    model = export_frenet_model(barrier_mu, barrier_delta)
    ocp.model = model

    ocp.dims.N = N

    # ── cost type ──
    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ── constraints ──────────────────────────────────────────

    # stage 0: all states fixed
    ocp.constraints.idxbx_0 = np.arange(NX)
    ocp.constraints.lbx_0   = np.zeros(NX)
    ocp.constraints.ubx_0   = np.zeros(NX)

    # stages 1..N-1: l bounds (soft) + vx>=0 (hard)
    ocp.constraints.idxbx  = np.array([IDX_L, IDX_VX])
    ocp.constraints.lbx    = np.array([l_min, 0.0])
    ocp.constraints.ubx    = np.array([l_max, VX_MAX])
    ocp.constraints.idxsbx = np.array([0])
    ocp.constraints.lsbx   = np.array([0.0])
    ocp.constraints.usbx   = np.array([0.0])

    # stage N: l (soft) + s<=s_max (soft) + vx>=0 (hard)
    ocp.constraints.idxbx_e  = np.array([IDX_L, IDX_S, IDX_VX])
    ocp.constraints.lbx_e    = np.array([l_min, -1e8, 0.0])
    ocp.constraints.ubx_e    = np.array([l_max, s_max, VX_MAX])
    ocp.constraints.idxsbx_e = np.array([0, 1])
    ocp.constraints.lsbx_e   = np.array([0.0, 0.0])
    ocp.constraints.usbx_e   = np.array([0.0, 0.0])

    # control bounds
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu   = np.array([DK_MIN, JX_MIN])
    ocp.constraints.ubu   = np.array([DK_MAX, JX_MAX])

    # ── slack costs ──
    ocp.cost.Zl = np.array([W_SLACK_L_L2])
    ocp.cost.Zu = np.array([W_SLACK_L_L2])
    ocp.cost.zl = np.array([W_SLACK_L_L1])
    ocp.cost.zu = np.array([W_SLACK_L_L1])

    ocp.cost.Zl_e = np.array([W_SLACK_L_L2, W_SLACK_S_L2])
    ocp.cost.Zu_e = np.array([W_SLACK_L_L2, W_SLACK_S_L2])
    ocp.cost.zl_e = np.array([W_SLACK_L_L1, W_SLACK_S_L1])
    ocp.cost.zu_e = np.array([W_SLACK_L_L1, W_SLACK_S_L1])

    # ── default parameter values ──
    ocp.parameter_values = np.array([
        DEFAULT_KR, DEFAULT_AX_MAX, DEFAULT_AY_MAX, L_REF, VX_REF,
    ])

    # ── solver options ──
    ocp.solver_options.qp_solver        = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type  = "SQP"
    ocp.solver_options.hessian_approx   = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type  = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 1
    ocp.solver_options.print_level      = 0
    ocp.solver_options.tol              = 1e-4
    ocp.solver_options.tf               = tf
    ocp.solver_options.N_horizon        = N
    # ocp.solver_options.nlp_solver_max_iter = 200
    # ocp.solver_options.globalization    = "MERIT_BACKTRACKING"
    # ocp.solver_options.regularize_method = "CONVEXIFY"

    json_file = os.path.join(".", model.name + "_acados_ocp.json")
    acados_solver = AcadosOcpSolver(ocp, json_file=json_file)
    return acados_solver

# ═══════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════

def plot_results(x_arr, u_arr, N, tf,
                 l_ref=L_REF, vx_ref=VX_REF,
                 l_min=L_MIN, l_max=L_MAX, s_max=S_MAX,
                 ax_max=DEFAULT_AX_MAX, ay_max=DEFAULT_AY_MAX):
    x_arr = np.array(x_arr)
    u_arr = np.array(u_arr)
    dt = tf / N
    t_x = np.arange(N + 1) * dt
    t_u = np.arange(N) * dt

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex="col")
    fig.suptitle("AES-SLT Frenet OCP (log-barrier)", fontsize=14)

    # l
    ax = axes[0, 0]
    ax.plot(t_x, x_arr[:, IDX_L], "b-", lw=2, label="$l$")
    ax.axhline(l_min, color="r", ls="--", lw=1, label="bounds")
    ax.axhline(l_max, color="r", ls="--", lw=1)
    ax.axhline(l_ref, color="m", ls=":", lw=1.2, label="$l_{ref}$")
    ax.set_ylabel("$l$ (m)");  ax.set_title("Lateral Offset")
    ax.legend(loc="best", fontsize=8);  ax.grid(True)

    # delta_theta
    ax = axes[0, 1]
    ax.plot(t_x, np.rad2deg(x_arr[:, IDX_DTHETA]), "b-", lw=2)
    ax.set_ylabel(r"$\delta\theta$ (deg)");  ax.set_title("Heading Error");  ax.grid(True)

    # kappa
    ax = axes[1, 0]
    ax.plot(t_x, x_arr[:, IDX_KAPPA], "b-", lw=2)
    ax.set_ylabel(r"$\kappa$ (1/m)");  ax.set_title("Curvature");  ax.grid(True)

    # s
    ax = axes[1, 1]
    ax.plot(t_x, x_arr[:, IDX_S], "b-", lw=2)
    ax.axhline(s_max, color="r", ls="--", lw=1, label="$S_{max}$")
    ax.set_ylabel("$s$ (m)");  ax.set_title("Arc Length")
    ax.legend(loc="best", fontsize=8);  ax.grid(True)

    # vx
    ax = axes[2, 0]
    ax.plot(t_x, x_arr[:, IDX_VX], "b-", lw=2, label="$v_x$")
    ax.axhline(0.0, color="r", ls="--", lw=1)
    ax.axhline(vx_ref, color="m", ls=":", lw=1.2, label="$v_{x,ref}$")
    ax.set_ylabel("$v_x$ (m/s)");  ax.set_title("Longitudinal Velocity")
    ax.legend(loc="best", fontsize=8);  ax.grid(True)

    # ax
    ax = axes[2, 1]
    ax.plot(t_x, x_arr[:, IDX_AX], "b-", lw=2)
    ax.axhline(ax_max, color="r", ls="--", lw=1, alpha=0.5)
    ax.axhline(-ax_max, color="r", ls="--", lw=1, alpha=0.5)
    ax.set_ylabel("$a_x$ (m/s²)");  ax.set_title("Longitudinal Accel");  ax.grid(True)

    # dk
    ax = axes[3, 0]
    ax.step(t_u, u_arr[:, IDX_DK], "b-", lw=2, where="post")
    ax.axhline(DK_MIN, color="r", ls="--", lw=1);  ax.axhline(DK_MAX, color="r", ls="--", lw=1)
    ax.set_xlabel("Time (s)");  ax.set_ylabel(r"$d\kappa$ (1/m/s)")
    ax.set_title("Curvature Rate");  ax.grid(True)

    # jx
    ax = axes[3, 1]
    ax.step(t_u, u_arr[:, IDX_JX], "b-", lw=2, where="post")
    ax.axhline(JX_MIN, color="r", ls="--", lw=1);  ax.axhline(JX_MAX, color="r", ls="--", lw=1)
    ax.set_xlabel("Time (s)");  ax.set_ylabel("$j_x$ (m/s³)")
    ax.set_title("Jerk");  ax.grid(True)

    fig.tight_layout()

    # ── friction ellipse post-hoc ──
    ax_vals = x_arr[:, IDX_AX]
    vx_vals = x_arr[:, IDX_VX]
    kappa_vals = x_arr[:, IDX_KAPPA]
    ay_vals = vx_vals**2 * kappa_vals
    fe_vals = ax_vals**2 / ax_max**2 + ay_vals**2 / ay_max**2

    fig2, (ax_fe, ax_ay) = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("Friction Ellipse (log-barrier in cost)", fontsize=13)

    ax_fe.plot(t_x, fe_vals, "b-", lw=2)
    ax_fe.axhline(1.0, color="r", ls="--", lw=1.5, label="limit = 1")
    ax_fe.set_xlabel("Time (s)")
    ax_fe.set_ylabel("$a_x^2/a_{x,max}^2 + a_y^2/a_{y,max}^2$")
    ax_fe.set_title("Friction Ellipse Utilisation")
    ax_fe.legend(loc="best");  ax_fe.grid(True)

    ax_ay.plot(t_x, ay_vals, "g-", lw=2, label="$a_y = v_x^2 \\kappa$")
    ax_ay.plot(t_x, ax_vals, "r-", lw=2, label="$a_x$")
    ax_ay.set_xlabel("Time (s)");  ax_ay.set_ylabel("Accel (m/s²)")
    ax_ay.set_title("Longitudinal & Lateral Acceleration")
    ax_ay.legend(loc="best");  ax_ay.grid(True)
    fig2.tight_layout()

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def run_ocp(
    N: int = 50,
    tf: float = 5.0,
    l_ref: float = L_REF,
    vx_ref: float = VX_REF,
    l_min: float = L_MIN,
    l_max: float = L_MAX,
    s_max: float = S_MAX,
    x0: np.ndarray = None,
    kr_list=None,
    ax_max: float = DEFAULT_AX_MAX,
    ay_max: float = DEFAULT_AY_MAX,
    barrier_mu: float = BARRIER_MU,
    barrier_delta: float = BARRIER_DELTA,
):
    if x0 is None:
        x0 = np.array([0.0, 0.0, 0.0, 0.0, vx_ref, 0.0])
    if kr_list is None:
        kr_list = [DEFAULT_KR] * (N + 1)

    logger.info("Building OCP  N=%d  tf=%.2f  barrier_mu=%.1f  barrier_delta=%.3f",
                N, tf, barrier_mu, barrier_delta)
    solver = set_acados_ocp(N, tf, l_min=l_min, l_max=l_max, s_max=s_max,
                            barrier_mu=barrier_mu, barrier_delta=barrier_delta)

    # ── set parameters per stage: [kr, ax_max, ay_max, l_ref, vx_ref] ──
    for i in range(N + 1):
        solver.set(i, "p", np.array([kr_list[i], ax_max, ay_max, l_ref, vx_ref]))

    # ── warm start: smooth S-curve ──
    dt = tf / N
    vx0 = x0[IDX_VX]
    l0  = x0[IDX_L]
    dl  = l_ref - l0
    dvx = vx_ref - vx0

    ws_x = np.zeros((N + 1, NX))
    for i in range(N + 1):
        t = i * dt
        frac = t / tf
        poly  = 3 * frac**2 - 2 * frac**3
        dpoly = (6 * frac - 6 * frac**2) / tf

        ws_x[i, IDX_L]  = l0 + dl * poly
        ws_x[i, IDX_VX] = vx0 + dvx * frac
        ws_x[i, IDX_AX] = dvx / tf if tf > 0 else 0.0

        vx_i = max(ws_x[i, IDX_VX], 0.1)
        ws_x[i, IDX_DTHETA] = dl * dpoly / vx_i

    ws_x[0, IDX_S] = x0[IDX_S]
    for i in range(1, N + 1):
        ws_x[i, IDX_S] = ws_x[i - 1, IDX_S] + ws_x[i - 1, IDX_VX] * dt

    for i in range(N + 1):
        if i == 0:
            dd = (ws_x[1, IDX_DTHETA] - ws_x[0, IDX_DTHETA]) / dt
        elif i == N:
            dd = (ws_x[N, IDX_DTHETA] - ws_x[N - 1, IDX_DTHETA]) / dt
        else:
            dd = (ws_x[i + 1, IDX_DTHETA] - ws_x[i - 1, IDX_DTHETA]) / (2 * dt)
        vx_i = max(ws_x[i, IDX_VX], 0.1)
        ws_x[i, IDX_KAPPA] = dd / vx_i + kr_list[i]

    for i in range(N + 1):
        solver.set(i, "x", ws_x[i])
    for i in range(N):
        solver.set(i, "u", np.zeros(NU))

    # ── solve ──
    t_start = time.perf_counter()
    solver.constraints_set(0, "lbx", x0)
    solver.constraints_set(0, "ubx", x0)
    status = solver.solve()
    t_elapsed = (time.perf_counter() - t_start) * 1000.0
    if status != 0:
        logger.warning("Solver status %d (non-zero), extracting result anyway", status)

    sqp_iter = solver.get_stats("sqp_iter")
    cost_val = solver.get_cost()
    logger.info("Solve: status=%d  sqp_iter=%s  cost=%.4f  time=%.2f ms",
                status, sqp_iter, cost_val, t_elapsed)

    x_sol = [solver.get(i, "x") for i in range(N + 1)]
    u_sol = [solver.get(i, "u") for i in range(N)]

    logger.info("x(0) = %s", x_sol[0])
    logger.info("x(N) = %s", x_sol[-1])
    logger.info("s(N) = %.4f  (s_max=%.1f)", x_sol[-1][IDX_S], s_max)

    x_arr = np.array(x_sol)
    ay_post = x_arr[:, IDX_VX]**2 * x_arr[:, IDX_KAPPA]
    fe_post = x_arr[:, IDX_AX]**2 / ax_max**2 + ay_post**2 / ay_max**2
    logger.info("Friction ellipse max = %.4f  (limit 1.0)", np.max(fe_post))

    return {
        "status": status, "x": x_sol, "u": u_sol,
        "N": N, "tf": tf,
        "l_ref": l_ref, "vx_ref": vx_ref,
        "l_min": l_min, "l_max": l_max, "s_max": s_max,
        "ax_max": ax_max, "ay_max": ay_max,
        "cost": cost_val, "sqp_iter": sqp_iter, "elapsed_ms": t_elapsed,
    }


if __name__ == "__main__":
    matplotlib.set_loglevel("warning")

    result = run_ocp(
        N=50,
        tf=5.0,
        l_ref=3.0,
        vx_ref=0.1,
        l_min=-4.0,
        l_max=4.0,
        s_max=25.0,
        x0=np.array([0.0, 0.0, 0.0, 0.0, 20.0, 0.0]),
        kr_list=None,
        ax_max=9.0,
        ay_max=8.0,
        barrier_mu=10.0,
        barrier_delta=0.05,
    )

    plot_results(
        result["x"], result["u"], result["N"], result["tf"],
        l_ref=result["l_ref"], vx_ref=result["vx_ref"],
        l_min=result["l_min"], l_max=result["l_max"],
        s_max=result["s_max"],
        ax_max=result["ax_max"], ay_max=result["ay_max"],
    )
    plt.show()
