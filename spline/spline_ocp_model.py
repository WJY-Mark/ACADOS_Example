import numpy as np
import casadi as ca
from acados_template import AcadosModel
from typing import List
import math
from vec2d import *

POLY_ORDER = 4
if POLY_ORDER == 5:
    X_XIDX = 0
    DX_XIDX = 1
    D2X_XIDX = 2
    D3X_XIDX = 3
    D4X_XIDX = 4
    Y_XIDX = 5
    DY_XIDX = 6
    D2Y_XIDX = 7
    D3Y_XIDX = 8
    D4Y_XIDX = 9

    D5X_UIDX = 0
    D5Y_UIDX = 1
elif POLY_ORDER == 4:
    X_XIDX = 0
    DX_XIDX = 1
    D2X_XIDX = 2
    D3X_XIDX = 3
    Y_XIDX = 4
    DY_XIDX = 5
    D2Y_XIDX = 6
    D3Y_XIDX = 7
    D4X_UIDX = 0
    D4Y_UIDX = 1
elif POLY_ORDER == 3:
    X_XIDX = 0
    DX_XIDX = 1
    D2X_XIDX = 2
    Y_XIDX = 3
    DY_XIDX = 4
    D2Y_XIDX = 5
    D3X_UIDX = 0
    D3Y_UIDX = 1


def get_dx_dy_from_vel_theta(vel, theta):
    """Convert velocity and angle to x,y components"""
    dx = vel * math.cos(theta)
    dy = vel * math.sin(theta)
    return dx, dy


def get_vel_theta_from_dx_dy(dx, dy):
    """Convert x,y components to velocity and angle"""
    vel = math.hypot(dx, dy)  # More accurate than sqrt(dx*dx + dy*dy)
    theta = math.atan2(dy, dx)
    return vel, theta


def get_d2x_d2y(at, an, tan_unit_vec):
    """Convert tangential/normal acceleration to x,y components"""
    # tan_unit_vec should be a tuple/list like (x_component, y_component)
    d2x = at * tan_unit_vec[0] - an * tan_unit_vec[1]
    d2y = at * tan_unit_vec[1] + an * tan_unit_vec[0]
    return d2x, d2y


def get_at_an(d2x, d2y, tan_unit_vec: Vec2d):
    """Convert x,y acceleration to tangential/normal components"""
    """ this dose no include third order derivatives, because we assume v = vx*i + vy*j, and vy == 0.
    Otherwise, di/dt = j, dj/dt = -i, and dv/dt = (dvx/dt-vy*w)*i + (dvy/dt+vx*w)*j """
    at = tan_unit_vec.x * d2x + tan_unit_vec.y * d2y
    an = -tan_unit_vec.y * d2x + tan_unit_vec.x * d2y
    return at, an  # Note: an is positive for left turns, negative for right


def get_kappa(dx, dy, d2x, d2y):
    """Calculate curvature from velocity and acceleration components"""
    speed_sq = dx*dx + dy*dy
    if speed_sq < 1e-10:  # Avoid division by zero for stationary objects
        return 0.0
    return (d2y*dx - d2x*dy) / (speed_sq ** 1.5)


class DynamicKnotPoint:
    def __init__(self, x, y, theta, kappa, v, a_lon, a_lat, t):
        self.x = x  # x position
        self.y = y  # y position
        self.theta = theta  # orientation angle in rad
        self.kappa = kappa  # curvature
        self.v = v  # velocity m/s
        self.a_lon = a_lon  # longitudinal acceleration m/s^2
        self.a_lat = a_lat  # lateral acceleration m/s^2
        self.t = t  # time stamp in s


class SplineKnotPoint:
    def __init__(self, state, control, t):
        self.state = state
        self.control = control
        self.t = t


def dynamic_to_spline_knot(dynamic_point):
    """Convert DynamicKnotPoint to SplineKnotPoint
    Args:
        dynamic_point: DynamicKnotPoint instance
    Returns:
        SplineKnotPoint instance with:
        - x, dx, d2x, y, dy, d2y from dynamic_point
        - Higher derivatives (d3x, d4x, d5x, d3y, d4y, d5y) set to 0.0
        - t copied directly
    """
    # Calculate dx and dy from velocity and angle
    dx, dy = get_dx_dy_from_vel_theta(dynamic_point.v, dynamic_point.theta)

    # Calculate d2x and d2y from longitudinal and lateral acceleration
    # Unit tangent vector (direction of motion)
    tan_vec = (math.cos(dynamic_point.theta), math.sin(dynamic_point.theta))
    d2x, d2y = get_d2x_d2y(dynamic_point.a_lon, dynamic_point.a_lat, tan_vec)

    # Create state array using the defined indices
    state = np.zeros(POLY_ORDER * 2)
    state[X_XIDX] = dynamic_point.x
    state[DX_XIDX] = dx
    state[D2X_XIDX] = d2x
    state[Y_XIDX] = dynamic_point.y
    state[DY_XIDX] = dy
    state[D2Y_XIDX] = d2y
    # d3x, d4x, d3y, d4y remain 0.0

    # Create control array (all zeros)
    control = np.zeros(2)

    return SplineKnotPoint(
        state=state,
        control=control,
        t=dynamic_point.t
    )


def spline_to_dynamic_knot(spline_point):
    """Convert SplineKnotPoint to DynamicKnotPoint
    Args:
        spline_point: SplineKnotPoint instance
    Returns:
        DynamicKnotPoint instance with:
        - x, y, t copied directly
        - theta calculated from dx and dy
        - v calculated from dx and dy
        - kappa calculated from derivatives
        - a_lon and a_lat calculated from d2x, d2y and theta
    """
    # Extract values using the defined indices
    dx = spline_point.state[DX_XIDX]
    dy = spline_point.state[DY_XIDX]
    d2x = spline_point.state[D2X_XIDX]
    d2y = spline_point.state[D2Y_XIDX]

    # Calculate velocity and angle
    vel, theta = get_vel_theta_from_dx_dy(dx, dy)

    # Calculate curvature
    kappa = get_kappa(dx, dy, d2x, d2y)

    # Calculate longitudinal and lateral acceleration
    if vel > 1e-6:  # Only calculate if moving (avoid division by zero)
        tan_vec = (dx/vel, dy/vel)  # Unit tangent
        a_lon, a_lat = get_at_an(d2x, d2y, tan_vec)
    else:
        a_lon, a_lat = 0.0, 0.0

    return DynamicKnotPoint(
        x=spline_point.state[X_XIDX],
        y=spline_point.state[Y_XIDX],
        theta=theta,
        kappa=kappa,
        v=vel,
        a_lon=a_lon,
        a_lat=a_lat,
        t=spline_point.t
    )


def make_spline_model():
    if POLY_ORDER == 5:
        x = ca.SX.sym("x")
        dx = ca.SX.sym("dx")
        d2x = ca.SX.sym("d2x")
        d3x = ca.SX.sym("d3x")
        d4x = ca.SX.sym("d4x")
        d5x = ca.SX.sym("d5x")

        y = ca.SX.sym("y")
        dy = ca.SX.sym("dy")
        d2y = ca.SX.sym("d2y")
        d3y = ca.SX.sym("d3y")
        d4y = ca.SX.sym("d4y")
        d5y = ca.SX.sym("d5y")

        controls = ca.vertcat(d5x, d5y)
        states = ca.vertcat(x, dx, d2x, d3x, d4x, y, dy, d2y, d3y, d4y)

        rhs = [dx, d2x, d3x, d4x, d5x, dy, d2y, d3y, d4y, d5y]
        return states, controls, rhs
    elif POLY_ORDER == 4:
        x = ca.SX.sym("x")
        dx = ca.SX.sym("dx")
        d2x = ca.SX.sym("d2x")
        d3x = ca.SX.sym("d3x")
        d4x = ca.SX.sym("d4x")
        y = ca.SX.sym("y")
        dy = ca.SX.sym("dy")
        d2y = ca.SX.sym("d2y")
        d3y = ca.SX.sym("d3y")
        d4y = ca.SX.sym("d4y")

        controls = ca.vertcat(d4x, d4y)
        states = ca.vertcat(x, dx, d2x, d3x, y, dy, d2y, d3y)

        rhs = [dx, d2x, d3x, d4x, dy, d2y, d3y, d4y]
        return states, controls, rhs
    elif POLY_ORDER == 3:
        x = ca.SX.sym("x")
        dx = ca.SX.sym("dx")
        d2x = ca.SX.sym("d2x")
        d3x = ca.SX.sym("d3x")
        y = ca.SX.sym("y")
        dy = ca.SX.sym("dy")
        d2y = ca.SX.sym("d2y")
        d3y = ca.SX.sym("d3y")
        controls = ca.vertcat(d3x, d3y)
        states = ca.vertcat(x, dx, d2x, y, dy, d2y)
        rhs = [dx, d2x, d3x, dy, d2y, d3y]
        return states, controls, rhs


def generate_dynamic_points():
    # return generate_defined_points()
    return generate_circular_motion()


def generate_defined_points():
    list = [
        DynamicKnotPoint(x=0.0, y=0.0, theta=0.5, kappa=0.01,
                         v=15.0, a_lon=5.0, a_lat=2.25, t=0.0),
        DynamicKnotPoint(x=20.0, y=0.0, theta=0.0, kappa=0.0,
                         v=20.0, a_lon=0.0, a_lat=0.0, t=1.0),
        DynamicKnotPoint(x=40.0, y=0.0, theta=0.0, kappa=0.0,
                         v=20.0, a_lon=0.0, a_lat=0.0, t=2.0),
        DynamicKnotPoint(x=60.0, y=0.0, theta=0.0, kappa=0.0,
                         v=20.0, a_lon=0.0, a_lat=0.0, t=3.0),
        DynamicKnotPoint(x=80.0, y=0.0, theta=0.0, kappa=0.0,
                         v=20.0, a_lon=0.0, a_lat=0.0, t=4.0),
        DynamicKnotPoint(x=100.0, y=0.0, theta=0.0, kappa=0.0,
                         v=20.0, a_lon=0.0, a_lat=0.0, t=5.0),
    ]
    return list


def generate_circular_motion():
    points = []
    radius = 10.0            # 圆周半径 (m)
    omega = 0.2             # 角速度 (rad/s)
    v = omega * radius      # 线速度 (m/s)
    a_lon = 0.0             # 纵向加速度 (匀速运动为0)
    a_lat = v**2 / radius   # 向心加速度 (m/s²)

    for t in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        theta = omega * t                   # 角度随时间增加
        x = radius * math.sin(theta)        # x = r*sin(θ)
        y = radius * (1 - math.cos(theta))  # y = r*(1-cos(θ))
        kappa = 1 / radius                  # 曲率 = 1/r

        point = DynamicKnotPoint(
            x=x, y=y, theta=theta, kappa=kappa,
            v=v, a_lon=a_lon, a_lat=a_lat, t=t
        )
        points.append(point)

    return points


class SplineOcpModel(object):
    def __init__(
        self,
    ):
        model = AcadosModel()  # ca.types.SimpleNamespace()
        states, controls, rhs = make_spline_model()

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

        self.nx = states.shape[0]
        self.nu = controls.shape[0]
        self.model = model
        self.ref_dyn_points: List[DynamicKnotPoint] = generate_dynamic_points()

        self.ref_spline_points: List[SplineKnotPoint] = [dynamic_to_spline_knot(self.ref_dyn_points[i])
                                                         for i in range(len(self.ref_dyn_points))]
