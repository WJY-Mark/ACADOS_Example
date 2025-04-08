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
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from vec2d import *
from spline2d import *

import numpy as np
import scipy.linalg
from typing import List

LON_BOUND = 0.1
LAT_BOUND = 0.1
DERIVATIVE_BOUND = 1.0
LON_VEL_BOUND = 1.0
LAT_VEL_BOUND = 1.0
OBB_CNSTR = True
OBB_E_CNSTR = True
VEL_CNSTR = True
VEL_E_CNSTR = True

C_E_DIM = 0
if OBB_E_CNSTR:
    C_E_DIM +=2
if VEL_E_CNSTR:
    C_E_DIM +=2

C_DIM = 0
if OBB_CNSTR:
    C_DIM +=2
if VEL_CNSTR:
    C_DIM +=2

X0_SeconOrderDerivativeConstraint = True
X_FirstOrderDerivativeConstraint = False
Xe_FirstOrderDerivativeConstraint = False


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


def set_cost_weight(nx, nu):
    d0_weight = 0.1
    d1_weight = 10.0
    d2_weight = 0.1
    d3_weight = 0.1
    d4_weight = 0.1
    d5_weight = 0.1
    if POLY_ORDER == 5:
        Q = np.eye(nx)
        Q[X_XIDX, X_XIDX] = d0_weight
        Q[DX_XIDX, DX_XIDX] = d1_weight
        Q[D2X_XIDX, D2X_XIDX] = d2_weight
        Q[D3X_XIDX, D3X_XIDX] = d3_weight
        Q[D4X_XIDX, D4X_XIDX] = d4_weight
        Q[Y_XIDX, Y_XIDX] = d0_weight
        Q[DY_XIDX, DY_XIDX] = d1_weight
        Q[D2Y_XIDX, D2Y_XIDX] = d2_weight
        Q[D3Y_XIDX, D3Y_XIDX] = d3_weight
        Q[D4Y_XIDX, D4Y_XIDX] = d4_weight

        R = np.eye(nu)
        R[D5X_UIDX, D5X_UIDX] = d5_weight
        R[D5Y_UIDX, D5Y_UIDX] = d5_weight

        return Q, R
    elif POLY_ORDER == 4:
        Q = np.eye(nx)
        Q[X_XIDX, X_XIDX] = d0_weight
        Q[DX_XIDX, DX_XIDX] = d1_weight
        Q[D2X_XIDX, D2X_XIDX] = d2_weight
        Q[D3X_XIDX, D3X_XIDX] = d3_weight
        Q[Y_XIDX, Y_XIDX] = d0_weight
        Q[DY_XIDX, DY_XIDX] = d1_weight
        Q[D2Y_XIDX, D2Y_XIDX] = d2_weight
        Q[D3Y_XIDX, D3Y_XIDX] = d3_weight

        R = np.eye(nu)
        R[D4X_UIDX, D4X_UIDX] = d5_weight
        R[D4Y_UIDX, D4Y_UIDX] = d5_weight
        return Q, R
    elif POLY_ORDER == 3:
        Q = np.eye(nx)
        Q[X_XIDX, X_XIDX] = d0_weight
        Q[DX_XIDX, DX_XIDX] = d1_weight
        Q[D2X_XIDX, D2X_XIDX] = d2_weight
        Q[Y_XIDX, Y_XIDX] = d0_weight
        Q[DY_XIDX, DY_XIDX] = d1_weight
        Q[D2Y_XIDX, D2Y_XIDX] = d2_weight

        R = np.eye(nu)
        R[D3X_UIDX, D3X_UIDX] = d5_weight
        R[D3Y_UIDX, D3Y_UIDX] = d5_weight
        return Q, R


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
        n_params = len(model.p)
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)
        Q, R = set_cost_weight(ocp.dims.nx, ocp.dims.nu)

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
        ocp.cost.yref_0 = np.zeros((ocp.dims.ny, 1))

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.Vu = Vu
        ocp.cost.Vx = Vx
        ocp.cost.yref = np.zeros((ocp.dims.ny, 1))

        ocp.cost.Vx_e = np.eye(ocp.dims.nx)
        ocp.cost.W_e = Q
        ocp.cost.yref_e = np.zeros((ocp.dims.nx, 1))

        # set constraints

        if X0_SeconOrderDerivativeConstraint:
            ocp.constraints.idxbx_0 = np.array(
                [X_XIDX, DX_XIDX, D2X_XIDX, Y_XIDX, DY_XIDX, D2Y_XIDX])
            ocp.constraints.lbx_0 = np.array(
                [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10])
            ocp.constraints.ubx_0 = np.array(
                [1e10, 1e10, 1e10, 1e10, 1e10, 1e10])
        else:
            ocp.constraints.idxbx_0 = np.array(
                [X_XIDX, DX_XIDX, Y_XIDX, DY_XIDX])
            ocp.constraints.lbx_0 = np.array([-1e10, -1e10,  -1e10, -1e10,])
            ocp.constraints.ubx_0 = np.array([1e10, 1e10, 1e10, 1e10])

        if X_FirstOrderDerivativeConstraint:
            ocp.constraints.idxbx = np.array(
                [DX_XIDX, DY_XIDX])
            ocp.constraints.lbx = np.array([-1e10, -1e10,])
            ocp.constraints.ubx = np.array([1e10, 1e10])

        if Xe_FirstOrderDerivativeConstraint:
            ocp.constraints.idxbx_e = np.array(
                [DX_XIDX, DY_XIDX])
            ocp.constraints.lbx_e = np.array([-1e10, -1e10,])
            ocp.constraints.ubx_e = np.array([1e10, 1e10])

        if C_DIM > 0:
            ug = np.full(C_DIM, 1e10)
            lg = np.full(C_DIM, -1e10)
            ocp.constraints.C = np.zeros((C_DIM, ocp.dims.nx))
            ocp.constraints.D = np.zeros((C_DIM, ocp.dims.nu))
            ocp.constraints.lg = lg
            ocp.constraints.ug = ug

        if C_E_DIM > 0:
            ug_e = np.full(C_DIM, 1e10)
            lg_e = np.full(C_DIM, -1e10)
            ocp.constraints.C_e = np.zeros((C_E_DIM, ocp.dims.nx))
            ocp.constraints.D_e = np.zeros((C_E_DIM, ocp.dims.nu))
            ocp.constraints.lg_e = lg_e
            ocp.constraints.ug_e = ug_e

        # solver options
        node_time = [
            spline_ocp_model.ref_spline_points[i].t for i in range(n_stage + 1)]

        ocp.solver_options.tf = t_horizon
        ocp.solver_options.shooting_nodes = np.array(node_time)
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        # ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        # ocp.solver_options.sim_method_num_stages = 1
        # ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.print_level = 0
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.nlp_solver_max_iter = 2
        # compile acados ocp
        json_file = os.path.join("./" + model.name + "_acados_ocp.json")
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)


def print_result(x, u, t, N):
    if POLY_ORDER == 3:
        for ii in range(N + 1):
            print("t:{:.6f} x:{:.6f} dx:{:.6f} d2x:{:.6f} y:{:.6f} dy:{:.6f} d2y:{:.6f}".format(
                t[ii],
                x[ii][X_XIDX],
                x[ii][DX_XIDX],
                x[ii][D2X_XIDX],
                x[ii][Y_XIDX],
                x[ii][DY_XIDX],
                x[ii][D2Y_XIDX]
            ))
            if ii < N:
                print(" d3x:{:.6f} d3y:{:.6f}".format(
                    u[ii][D3X_UIDX], u[ii][D3Y_UIDX]))
    if POLY_ORDER == 4:
        for ii in range(N + 1):
            print("t:{:.6f} x:{:.6f} dx:{:.6f} d2x:{:.6f} d3x:{:.6f} y:{:.6f} dy:{:.6f} d2y:{:.6f} d3y:{:.6f}".format(
                t[ii],
                x[ii][X_XIDX],
                x[ii][DX_XIDX],
                x[ii][D2X_XIDX],
                x[ii][D3X_XIDX],
                x[ii][Y_XIDX],
                x[ii][DY_XIDX],
                x[ii][D2Y_XIDX],
                x[ii][D3Y_XIDX]
            ))
            if ii < N:
                print(" d4x:{:.6f} d4y:{:.6f}".format(
                    u[ii][D4X_UIDX], u[ii][D4Y_UIDX]))
    if POLY_ORDER == 5:
        for ii in range(N + 1):
            print("t:{:.6f} x:{:.6f} d2x:{:.6f} d3x:{:.6f} d4x:{:.6f} y:{:.6f} dy:{:.6f} d2y:{:.6f} d3y:{:.6f} d4y:{:.6f}".format(
                t[ii],
                x[ii][X_XIDX],
                x[ii][D2X_XIDX],
                x[ii][D3X_XIDX],
                x[ii][D4X_XIDX],
                x[ii][Y_XIDX],
                x[ii][DY_XIDX],
                x[ii][D2Y_XIDX],
                x[ii][D3Y_XIDX],
                x[ii][D4Y_XIDX]
            ))
            if ii < N:
                print(" d5x:{:.6f} d5y:{:.6f}".format(
                    u[ii][D5X_UIDX], u[ii][D5Y_UIDX]))


def get_result(x, u, t, N):
    opt_dyn_points: List[DynamicKnotPoint] = []
    for i in range(N + 1):
        vel, theta = get_vel_theta_from_dx_dy(x[i][DX_XIDX],
                                              x[i][DY_XIDX])
        kappa = get_kappa(x[i][DX_XIDX],
                          x[i][DY_XIDX], x[i][D2X_XIDX], x[i][D2Y_XIDX])
        vec = Vec2d.from_angle(theta)
        at, an = get_at_an(x[i][D2X_XIDX], x[i][D2Y_XIDX], vec)
        dyn_point = DynamicKnotPoint(
            x=x[i][X_XIDX],
            y=x[i][Y_XIDX],
            theta=theta,
            kappa=kappa,
            v=vel, a_lat=at, a_lon=an, t=t[i],
        )
        opt_dyn_points.append(dyn_point)
    opt_spline_points = []
    for i in range(N+1):
        if i == N:
            opt_spline_points.append(SplineKnotPoint(x[i], [0.0, 0.0], t[i]))
        else:
            opt_spline_points.append(SplineKnotPoint(x[i], u[i], t[i]))

    spline_2d_segs = []
    for i in range(N):
        if POLY_ORDER == 5:
            seg = Spline2dSeg.spline_seg_from_derivatives(
                x=x[i][X_XIDX], dx=x[i][DX_XIDX], d2x=x[i][D2X_XIDX], d3x=x[i][D3X_XIDX], d4x=x[i][D4X_XIDX], d5x=u[i][D5X_UIDX],
                y=x[i][Y_XIDX], dy=x[i][DY_XIDX], d2y=x[i][D2Y_XIDX], d3y=x[i][D3Y_XIDX], d4y=x[i][D4Y_XIDX], d5y=u[i][D5Y_UIDX])
        elif POLY_ORDER == 4:
            seg = Spline2dSeg.spline_seg_from_derivatives(
                x=x[i][X_XIDX], dx=x[i][DX_XIDX], d2x=x[i][D2X_XIDX], d3x=x[i][D3X_XIDX], d4x=u[i][D4X_UIDX], d5x=0.0,
                y=x[i][Y_XIDX], dy=x[i][DY_XIDX], d2y=x[i][D2Y_XIDX], d3y=x[i][D3Y_XIDX], d4y=u[i][D4Y_UIDX], d5y=0.0)
        elif POLY_ORDER == 3:
            seg = Spline2dSeg.spline_seg_from_derivatives(
                x=x[i][X_XIDX], dx=x[i][DX_XIDX], d2x=x[i][D2X_XIDX], d3x=u[i][D3X_UIDX], d4x=0.0, d5x=0.0,
                y=x[i][Y_XIDX], dy=x[i][DY_XIDX], d2y=x[i][D2Y_XIDX], d3y=u[i][D3Y_UIDX], d4y=0.0, d5y=0.0,)

        spline_2d_segs.append(seg)

    return opt_dyn_points, opt_spline_points, spline_2d_segs


def solve_problem(
    spline_ocp_model: SplineOcpModel, acados_solver: AcadosOcpSolver, tf, N
):
    init_bound = 0.0
    init_dd_bound = 0.0

    for i in range(0, N + 1):
        spline_point: SplineKnotPoint = spline_ocp_model.ref_spline_points[i]
        state = spline_point.state
        x_pos = state[X_XIDX]
        y_pos = state[Y_XIDX]
        dx = state[DX_XIDX]
        dy = state[DY_XIDX]
        d2x = state[D2X_XIDX]
        d2y = state[D2Y_XIDX]
        dyn_point: DynamicKnotPoint = spline_ocp_model.ref_dyn_points[i]
        if i == 0:
            if X0_SeconOrderDerivativeConstraint:
                lbx = np.array([x_pos-init_bound, dx-init_bound, d2x-init_dd_bound,
                                y_pos-init_bound, dy-init_bound, d2y-init_dd_bound])
                acados_solver.set(i, "lbx", lbx)
                ubx = np.array([x_pos+init_bound, dx+init_bound, d2x+init_dd_bound,
                                y_pos+init_bound, dy+init_bound, d2y+init_dd_bound])
                acados_solver.set(i, "ubx", ubx)
            else:
                lbx = np.array([x_pos-init_bound, dx-init_bound,
                                y_pos-init_bound, dy-init_bound])
                acados_solver.set(i, "lbx", lbx)
                ubx = np.array([x_pos+init_bound, dx+init_bound,
                                y_pos+init_bound, dy+init_bound])
                acados_solver.set(i, "ubx", ubx)
        elif i == N:
            if Xe_FirstOrderDerivativeConstraint:
                lbx = np.array([dx-DERIVATIVE_BOUND, dy-DERIVATIVE_BOUND])
                ubx = np.array([dx+DERIVATIVE_BOUND, dy+DERIVATIVE_BOUND])
                acados_solver.set(i, "ubx", ubx)
                acados_solver.set(i, "lbx", lbx)
            if C_E_DIM>0:
                dim = 0
                theta = dyn_point.theta
                C = np.zeros((C_E_DIM, spline_ocp_model.nx))
                lg = np.full(C_E_DIM, -1e10)
                ug = np.full(C_E_DIM, 1e10)
                if OBB_E_CNSTR:
                    C[dim, X_XIDX] = math.cos(theta)
                    C[dim, Y_XIDX] = math.sin(theta)
                    C[dim+1, X_XIDX] = -math.sin(theta)
                    C[dim+1, Y_XIDX] = math.cos(theta)
                    ug[dim] = math.cos(theta)*dyn_point.x + math.sin(theta)*dyn_point.y + LON_BOUND
                    ug[dim+1] = -math.sin(theta)*dyn_point.x + math.cos(theta)*dyn_point.y + LAT_BOUND
                    lg[dim] = math.cos(theta)*dyn_point.x + math.sin(theta)*dyn_point.y - LON_BOUND
                    lg[dim+1] = -math.sin(theta)*dyn_point.x + math.cos(theta)*dyn_point.y - LAT_BOUND
                    dim += 2
                if VEL_E_CNSTR:
                    C[dim, DX_XIDX] = math.cos(theta)
                    C[dim, DY_XIDX] = math.sin(theta)
                    C[dim+1, DX_XIDX] = -math.sin(theta)
                    C[dim+1, DY_XIDX] = math.cos(theta)
                    ug[dim] = dyn_point.v + LON_VEL_BOUND
                    ug[dim+1] = LAT_VEL_BOUND
                    lg[dim] = dyn_point.v - LON_VEL_BOUND
                    lg[dim+1] = -LAT_VEL_BOUND
                    dim+=2
                acados_solver.constraints_set(i, "C", C, 'new')
                acados_solver.constraints_set(i, "ug", ug, 'new')
                acados_solver.constraints_set(i, "lg", lg, 'new')
        else:
            if X_FirstOrderDerivativeConstraint:
                lbx = np.array([dx-DERIVATIVE_BOUND, dy-DERIVATIVE_BOUND])
                ubx = np.array([dx+DERIVATIVE_BOUND, dy+DERIVATIVE_BOUND])
                acados_solver.set(i, "ubx", ubx)
                acados_solver.set(i, "lbx", lbx)

            if C_DIM>0:
                dim = 0
                theta = dyn_point.theta
                C = np.zeros((C_DIM, spline_ocp_model.nx))
                lg = np.full(C_DIM, -1e10)
                ug = np.full(C_DIM, 1e10)
                if OBB_CNSTR:
                    C[dim, X_XIDX] = math.cos(theta)
                    C[dim, Y_XIDX] = math.sin(theta)
                    C[dim+1, X_XIDX] = -math.sin(theta)
                    C[dim+1, Y_XIDX] = math.cos(theta)
                    ug[dim] = math.cos(theta)*dyn_point.x + math.sin(theta)*dyn_point.y + LON_BOUND
                    ug[dim+1] = -math.sin(theta)*dyn_point.x + math.cos(theta)*dyn_point.y + LAT_BOUND
                    lg[dim] = math.cos(theta)*dyn_point.x + math.sin(theta)*dyn_point.y - LON_BOUND
                    lg[dim+1] = -math.sin(theta)*dyn_point.x + math.cos(theta)*dyn_point.y - LAT_BOUND
                    dim += 2
                if VEL_CNSTR:
                    C[dim, DX_XIDX] = math.cos(theta)
                    C[dim, DY_XIDX] = math.sin(theta)
                    C[dim+1, DX_XIDX] = -math.sin(theta)
                    C[dim+1, DY_XIDX] = math.cos(theta)
                    ug[dim] = dyn_point.v + LON_VEL_BOUND
                    ug[dim+1] = LAT_VEL_BOUND
                    lg[dim] = dyn_point.v - LON_VEL_BOUND
                    lg[dim+1] = -LAT_VEL_BOUND
                    dim+=2
                acados_solver.constraints_set(i, "C", C, 'new')
                acados_solver.constraints_set(i, "ug", ug, 'new')
                acados_solver.constraints_set(i, "lg", lg, 'new')

    for i in range(N+1):
        spline_point: SplineKnotPoint = spline_ocp_model.ref_spline_points[i]
        state = spline_point.state
        control = spline_point.control
        state_and_control = np.concatenate((state, control))
        if i == N:
            yref = np.array(state)
        else:
            yref = np.array(state_and_control)

        acados_solver.set(i, "yref", yref)

    start_time = time.perf_counter()
    status = acados_solver.solve()
    nlp_iter = acados_solver.get_stats("nlp_iter")
    sqp_iter = acados_solver.get_stats("sqp_iter")
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) * 1000
    print(f"Elapsed time: {elapsed_time:.2f} ms status: {status} nlp_iter: {nlp_iter} sqp_iter: {sqp_iter}")
    x = [acados_solver.get(i, "x") for i in range(N + 1)]
    u = [acados_solver.get(i, "u") for i in range(N)]

    t_knots: List[float] = [
        spline_ocp_model.ref_spline_points[i].t for i in range(N + 1)]
    t_span: List[float] = [t_knots[i] - t_knots[i - 1]
                           for i in range(1, N + 1)]

    opt_dyn_points, opt_spline_points, spline_2d_segs = get_result(
        x, u, t_knots, N)
    print_result(x, u, t_knots, N)
    spline_2d = Spline2d()
    spline_2d.segs = spline_2d_segs
    spline_2d.t_span = t_span
    spline_2d.t_knots = t_knots

    return spline_2d, opt_dyn_points, opt_spline_points


def draw_rotated_box(ax, center_x, center_y, length, width, angle_rad,
                     edgecolor="orange", facecolor="none", linewidth=1):
    """
    在指定位置绘制旋转矩形
    Args:
        ax: matplotlib的Axes对象
        center_x, center_y: 矩形中心坐标
        length: 矩形总长度
        width: 矩形总宽度
        angle_rad: 旋转角度
        edgecolor: 边框颜色
        facecolor: 填充颜色
        linewidth: 边框线宽
    """
    # 创建矩形（初始位置为未旋转时的左下角）
    box = Rectangle(
        xy=(center_x - length/2, center_y - width/2),
        width=length,
        height=width,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )

    t = Affine2D().rotate_around(center_x, center_y, angle_rad) + ax.transData
    box.set_transform(t)
    ax.add_patch(box)


def draw_results(
    spline_2d: Spline2d, opt_dyn_points: List[DynamicKnotPoint],
    opt_spline_points: List[SplineKnotPoint], origin_dyn_points: List[DynamicKnotPoint],
    origin_spline_points: List[SplineKnotPoint]
):
    plt.figure()
    ax = plt.gca()

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

    opt_x = [point.x for point in opt_dyn_points]
    opt_y = [point.y for point in opt_dyn_points]
    ax.scatter(opt_x, opt_y, c="r", marker="o",
               label="Optimized Anchor Points")

    orig_x = [point.x for point in origin_dyn_points]
    orig_y = [point.y for point in origin_dyn_points]
    orig_theta = [point.theta for point in origin_dyn_points]
    ax.scatter(orig_x, orig_y, c="g", marker="o", label="Origin Anchor Points")

    for x, y, theta in zip(orig_x, orig_y, orig_theta):
        draw_rotated_box(
            ax=ax,
            center_x=x,
            center_y=y,
            length=2 * LON_BOUND,
            width=2 * LAT_BOUND,
            angle_rad=theta,
            edgecolor="orange"
        )

    # Add labels and legend
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Optimized Spline and Anchor Points")
    ax.legend()
    ax.set_xlim(left=-10, right=110)
    ax.set_ylim(bottom=-100, top=100)
    # Show the plot
    plt.grid(True)
    plt.show()
    pass


if __name__ == "__main__":

    spline_ocp_model = SplineOcpModel()
    origin_spline_points = spline_ocp_model.ref_spline_points
    origin_dyn_points = spline_ocp_model.ref_dyn_points
    t_horizon = origin_dyn_points[-1].t
    n_stage = len(origin_dyn_points) - 1
    ocp = SplineOcpOpt(spline_ocp_model, t_horizon, n_stage)
    spline_2d, opt_dyn_points, opt_spline_points = solve_problem(
        spline_ocp_model, ocp.solver, t_horizon, n_stage
    )
    draw_results(spline_2d, opt_dyn_points, opt_spline_points,
                 origin_dyn_points, origin_spline_points)
