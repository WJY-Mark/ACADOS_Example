#!/usr/bin/env python
# coding=UTF-8

import os
import sys
import shutil
import errno
import timeit

from spline_ocp_model import *
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import time
import matplotlib.pyplot as plt

# import casadi as ca
import numpy as np
import scipy.linalg
from typing import List


X0_InitConstraint = False


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


class Spline2dSeg:
    def __init__(self, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3):
        self.a0 = _a0
        self.a1 = _a1
        self.a2 = _a2
        self.a3 = _a3

        self.b0 = _b0
        self.b1 = _b1
        self.b2 = _b2
        self.b3 = _b3

    def evaluate(self, t):
        t2 = t * t
        t3 = t2 * t
        x = self.a0 + self.a1 * t + self.a2 * t2 + self.a3 * t3
        y = self.b0 + self.b1 * t + self.b2 * t2 + self.b3 * t3
        return x, y

    def evaluate_derivative(self, t):
        t2 = t * t
        x_dot = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t2
        y_dot = self.b1 + 2 * self.b2 * t + 3 * self.b3 * t2
        return x_dot, y_dot

    def evaluate_second_derivative(self, t):
        t2 = t * t
        t3 = t2 * t
        x_ddot = 2 * self.a2 + 6 * self.a3 * t
        y_ddot = 2 * self.b2 + 6 * self.b3 * t
        return x_ddot, y_ddot


class Spline2d:
    def __init__(self):
        self.segs: List[Spline2dSeg] = []
        self.t_knots: List[float] = []
        self.t_span: List[float] = []


class SplineOcpOpt(object):
    def __init__(self, spline_ocp_model: SplineOcpModel, t_horizon, n_stage):
        model = spline_ocp_model.model

        # Ensure current working directory is current folder
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.acados_models_dir = "./acados_models"
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))
        acados_source_path = os.environ["ACADOS_SOURCE_DIR"]
        sys.path.insert(0, acados_source_path)

        # create OCP
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.nx = ocp.model.x.size()[0]
        ocp.dims.nu = ocp.model.u.size()[0]
        ocp.dims.ny = ocp.dims.nx + ocp.dims.nu
        ocp.dims.ny_e = 0  # not set for now
        ocp.dims.nbx = ocp.dims.nx  # number of state bounds
        ocp.dims.nbu = ocp.dims.nu  # number of control bounds
        ocp.acados_include_path = acados_source_path + "/include"
        ocp.acados_lib_path = acados_source_path + "/lib"
        ocp.dims.N = n_stage
        ocp.solver_options.tf = t_horizon
        n_params = len(model.p)
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        # initialize parameters
        pos_weight = 0.1
        d_weight = 1.0
        dd_weight = 0.1
        ddd_weight = 0.1
        # cost type
        Q = np.eye(ocp.dims.nx)
        Q[X_XIDX, X_XIDX] = pos_weight
        Q[DX_XIDX, DX_XIDX] = d_weight
        Q[DDX_XIDX, DDX_XIDX] = dd_weight
        Q[Y_XIDX, Y_XIDX] = pos_weight
        Q[DY_XIDX, DY_XIDX] = d_weight
        Q[DDY_XIDX, DDY_XIDX] = dd_weight
        R = np.eye(ocp.dims.nu)
        R[DDDX_UIDX, DDDX_UIDX] = ddd_weight
        R[DDDY_UIDX, DDDY_UIDX] = ddd_weight

        ocp.cost.cost_type = "LINEAR_LS"  # cost type, default format
        ocp.cost.cost_type_e = "LINEAR_LS"
        # set the cost function form

        Vx = np.zeros((ocp.dims.ny, ocp.dims.nx))
        Vx[: ocp.dims.nx, : ocp.dims.nx] = np.eye(ocp.dims.nx)
        Vu = np.zeros((ocp.dims.ny, ocp.dims.nu))
        Vu[ocp.dims.ny - 1, 0] = 1

        ocp.cost.W_0 = scipy.linalg.block_diag(Q, R)
        ocp.cost.Vx_0 = Vx
        ocp.cost.Vu_0 = Vu
        ocp.cost.yref_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.Vu = Vu
        ocp.cost.Vx = Vx
        ocp.cost.yref = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        ocp.cost.Vx_e = np.eye(ocp.dims.nx)
        ocp.cost.W_e = Q
        ocp.cost.yref_e = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # set constraints
        point = spline_ocp_model.ref_points[0]

        if X0_InitConstraint:
            ocp.constraints.idxbx_0 = np.array(
                [X_XIDX, DX_XIDX, DDX_XIDX, Y_XIDX, DY_XIDX, DDY_XIDX])
            ocp.constraints.lbx_0 = np.array(
                [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10]
            )
            ocp.constraints.ubx_0 = np.array(
                [1e10, 1e10, 1e10, 1e10, 1e10, 1e10]
            )
        else:
            ocp.constraints.idxbx_0 = np.array(
                [X_XIDX, DX_XIDX, Y_XIDX, DY_XIDX])
            ocp.constraints.lbx_0 = np.array(
                [-1e10, -1e10,  -1e10, -1e10,]
            )
            ocp.constraints.ubx_0 = np.array(
                [1e10, 1e10, 1e10, 1e10]
            )

        ocp.constraints.idxbx = np.array([X_XIDX, Y_XIDX])
        ocp.constraints.lbx = np.array([-1e10, -1e10])
        ocp.constraints.ubx = np.array([1e10, 1e10])

        ocp.constraints.idxbx_e = np.array([X_XIDX, Y_XIDX])
        ocp.constraints.lbx_e = np.array([-1e10, -1e10])
        ocp.constraints.ubx_e = np.array([1e10, 1e10])

        # solver options

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        # explicit Runge-Kutta integrator
        ocp.solver_options.integrator_type = "ERK"
        # ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        node_time = [
            spline_ocp_model.ref_points[i].t for i in range(n_stage + 1)]

        ocp.solver_options.shooting_nodes = np.array(node_time)

        # compile acados ocp
        json_file = os.path.join("./" + model.name + "_acados_ocp.json")
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)


def CalcKappa(dx, ddx, dy, ddy):
    # Calculate the numerator and denominator for kappa
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2) ** (3 / 2)
    if denominator == 0:
        raise ValueError("Denominator is zero, curvature is undefined.")
    # Calculate curvature
    kappa = numerator / denominator
    return kappa


def solve_problem(
    spline_ocp_model: SplineOcpModel, acados_solver: AcadosOcpSolver, tf, N
):
    bound = 0.5
    init_bound = 0.0
    init_dd_bound = 1e10

    for i in range(0, N + 1):
        point = spline_ocp_model.ref_points[i]
        if i == N:
            yref = np.array([point.x, point.dx, 0.0, point.y, point.dy, 0.0])
        else:
            yref = np.array(
                [point.x, point.dx, 0.0, point.y, point.dy, 0.0, 0.0, 0.0])

        acados_solver.set(i, "yref", yref)
        if i == 0:
            if X0_InitConstraint:
                lbx = np.array([point.x-init_bound, point.dx-init_bound, 0.0-init_dd_bound,
                            point.y-init_bound, point.dy-init_bound, 0.0-init_dd_bound])
                acados_solver.set(i, "lbx", lbx)
                ubx = np.array([point.x+init_bound, point.dx+init_bound, 0.0+init_dd_bound,
                            point.y+init_bound, point.dy+init_bound, 0.0+init_dd_bound])
                acados_solver.set(i, "ubx", ubx)
            else:
                lbx = np.array([point.x-init_bound, point.dx-init_bound,
                            point.y-init_bound, point.dy-init_bound])
                acados_solver.set(i, "lbx", lbx)
                ubx = np.array([point.x+init_bound, point.dx+init_bound,
                            point.y+init_bound, point.dy+init_bound])
                acados_solver.set(i, "ubx", ubx)
        else:
            lbx = np.array([point.x - bound, point.y - bound])
            acados_solver.set(i, "lbx", lbx)
            ubx = np.array([point.x + bound, point.y + bound])
            acados_solver.set(i, "ubx", ubx)

    # point = spline_ocp_model.ref_points[-1]
    # yref_e = np.array([point.x, point.dx, 0.0, point.y, point.dy, 0.0])
    # acados_solver.set(N, "yref", yref_e)
    # lbx = np.array([point.x - bound, point.y - bound])
    # acados_solver.set(N, "lbx", lbx)
    # ubx = np.array([point.x + bound, point.y + bound])
    # acados_solver.set(N, "ubx", ubx)
    start_time = time.perf_counter()
    status = acados_solver.solve()
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) * 1000
    print(f"Elapsed time: {elapsed_time:.2f} milliseconds status: {status}")
    x = [acados_solver.get(i, "x") for i in range(N + 1)]
    u = [acados_solver.get(i, "u") for i in range(N)]
    for ii in range(N + 1):
        print(" x:{:.6f} dx:{:.6f} ddx:{:.6f} y:{:.6f} dy:{:.6f} ddy:{:.6f}".format(
            x[ii][X_XIDX],
            x[ii][DX_XIDX],
            x[ii][DDX_XIDX],
            x[ii][Y_XIDX],
            x[ii][DY_XIDX],
            x[ii][DDY_XIDX]
        ))

        if ii < N:
            print(" dddx:{:.6f} dddy:{:.6f}".format(
                u[ii][DDDX_UIDX], u[ii][DDDY_UIDX]
            ))

    opt_anchor_points: List[Point] = [
        Point(
            x[i][X_XIDX],
            x[i][Y_XIDX],
            x[i][DX_XIDX],
            x[i][DY_XIDX],
            CalcKappa(x[i][DX_XIDX], x[i][DDX_XIDX],
                      x[i][DY_XIDX], x[i][DDY_XIDX]),
            spline_ocp_model.ref_points[i].t,
        )
        for i in range(N + 1)
    ]

    spline_2d_segs: List[Spline2dSeg] = [
        Spline2dSeg(
            x[i][X_XIDX],
            x[i][DX_XIDX],
            x[i][DDX_XIDX] / 2.0,
            u[i][DDDX_UIDX] / 6.0,
            x[i][Y_XIDX],
            x[i][DY_XIDX],
            x[i][DDY_XIDX] / 2.0,
            u[i][DDDY_UIDX] / 6.0,
        )
        for i in range(N)
    ]

    t_knots: List[float] = [
        spline_ocp_model.ref_points[i].t for i in range(N + 1)]
    t_span: List[float] = [t_knots[i] - t_knots[i - 1]
                           for i in range(1, N + 1)]
    spline_2d = Spline2d()
    spline_2d.segs = spline_2d_segs
    spline_2d.t_span = t_span
    spline_2d.t_knots = t_knots

    return spline_2d, opt_anchor_points


def draw_results(
    spline_2d: Spline2d, opt_anchor_points: List[Point], origin_points: List[Point]
):
    # Create a figure and axis
    plt.figure()
    ax = plt.gca()

    # Plot the optimized spline segments
    for seg, t_end in zip(spline_2d.segs, spline_2d.t_span):
        t_start = 0.0
        t_values = np.linspace(t_start, t_end, 100)
        x_values, y_values = seg.evaluate(t_values)
        ax.plot(
            x_values,
            y_values,
            "b-",
            label="Optimized Spline",
        )

    # Plot the optimized anchor points
    x_points = [point.x for point in opt_anchor_points]
    y_points = [point.y for point in opt_anchor_points]
    ax.scatter(x_points, y_points, c="r", marker="o",
               label="Optimized Anchor Points")
    orig_x_points = [point.x for point in origin_points]
    orig_y_points = [point.y for point in origin_points]
    ax.scatter(
        orig_x_points, orig_y_points, c="g", marker="o", label="Origin Anchor Points"
    )

    # Add labels and legend
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Optimized Spline and Anchor Points")
    ax.legend()
    ax.set_xlim(left=-10, right=110)  # 明确参数名
    ax.set_ylim(bottom=-100, top=100)
    # Show the plot
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    spline_ocp_model = SplineOcpModel()
    origin_points = spline_ocp_model.ref_points
    t_horizon = origin_points[-1].t
    n_stage = len(origin_points) - 1
    ocp = SplineOcpOpt(spline_ocp_model, t_horizon, n_stage)
    spline_2d, opt_anchor_points = solve_problem(
        spline_ocp_model, ocp.solver, t_horizon, n_stage
    )
    draw_results(spline_2d, opt_anchor_points, origin_points)
