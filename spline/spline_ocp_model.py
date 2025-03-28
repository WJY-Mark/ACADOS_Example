import numpy as np
import casadi as ca
from acados_template import AcadosModel
from typing import List


class Point:
    def __init__(self, _x, _y, _dx, _dy, _k, _t):
        self.x = _x
        self.y = _y
        self.dx = _dx
        self.dy = _dy
        self.kappa = _k
        self.t = _t


X_XIDX = 0
DX_XIDX = 1
DDX_XIDX = 2
Y_XIDX = 3
DY_XIDX = 4
DDY_XIDX = 5

DDDX_UIDX = 0
DDDY_UIDX = 1


class SplineOcpModel(object):
    def __init__(
        self,
    ):
        model = AcadosModel()  # ca.types.SimpleNamespace()
        x = ca.SX.sym("x")
        dx = ca.SX.sym("dx")
        ddx = ca.SX.sym("ddx")
        dddx = ca.SX.sym("dddx")

        y = ca.SX.sym("y")
        dy = ca.SX.sym("dy")
        ddy = ca.SX.sym("ddy")
        dddy = ca.SX.sym("dddy")

        controls = ca.vertcat(dddx, dddy)
        states = ca.vertcat(x, dx, ddx, y, dy, ddy)

        rhs = [dx, ddx, dddx, dy, ddy, dddy]

        # function
        f = ca.Function(
            "f", [states, controls], [ca.vcat(rhs)], [
                "state", "control_input"], ["rhs"]
        )
        # f_expl = ca.vcat(rhs)
        # acados model
        x_dot = ca.SX.sym("x_dot", len(rhs))
        f_impl = x_dot - f(states, controls)

        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = []
        model.name = "spline_ocp"

        self.model = model
        self.ref_points: List[Point] = [
            Point(0.0, 0.0, 10.0, 0.0, 0.0, 0.0),
            Point(20.0, 1.0, 20.0, 0.0, 0.0, 1.0),
            Point(40.0, 1.0, 20.0, 0.0, 0.0, 2.0),
            Point(60.0, 0.5, 20.0, 0.0, 0.0, 3.0),
            Point(80.0, 1.5, 20.0, 0.0, 0.0, 4.0),
            Point(100.0, 1.0, 20.0, 0.0, 0.0, 5.0),
            # Point(120.0, 1.0, 20.0, 0.0, 0.0, 6.0),
            # Point(140.0, 1.0, 20.0, 0.0, 0.0, 7.0),
            # Point(160.0, 1.0, 20.0, 0.0, 0.0, 8.0),
            # Point(180.0, 1.0, 20.0, 0.0, 0.0, 9.0),
            # Point(200.0, 1.0, 20.0, 0.0, 0.0, 10.0),
            # Point(220.0, 1.0, 20.0, 0.0, 0.0, 11.0),
            # Point(240.0, 1.0, 20.0, 0.0, 0.0, 12.0),
            # Point(260.0, 1.0, 20.0, 0.0, 0.0, 13.0),
            # Point(280.0, 1.0, 20.0, 0.0, 0.0, 14.0),
            # Point(300.0, 1.0, 20.0, 0.0, 0.0, 15.0),
        ]
        # self.ref_points.append(point0)
        # self.ref_points.append(point1)
        # self.ref_points.append(point2)
        # self.ref_points.append(point3)
        # self.ref_points.append(point4)
        # self.ref_points.append(point5)

        # self.constraint = constraint
