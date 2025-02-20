from casadi import SX, vertcat, sin, cos, MX, tan, types
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import numpy as np
import scipy.linalg
import os
import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def export_simple_frenet_model():
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()
    model_name = "improved_frenet"
    l = SX.sym("l")
    delta_theta = SX.sym("delta_theta")

    # control
    dk = SX.sym("dk")  # this is not the real curvature change rate, difference between kb and ks
    u = vertcat(dk)

    # state, lateral, theta deviation, kappa
    k = SX.sym("k")  # curvature as control input, this is differentte
    x = vertcat(l, delta_theta, k)

    # state
    l_dot = SX.sym("l_dot")
    delta_theta_dot = SX.sym("delta_theta_dot")
    k_dot = SX.sym("k_dot")
    xdot = vertcat(l_dot, delta_theta_dot, k_dot)

    kb = SX.sym("kb")  # curvature as control input
    yref = SX.sym("yref")  # countour reference, lateral distance reference

    p = vertcat(kb, yref)

    factor = 1 - l * kb

    f_expl = vertcat(factor * tan(delta_theta),
                     factor * k / cos(delta_theta) - kb,
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
    z = vertcat([])
    model.z = z

    return model


def to_acados_model(model):
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    return model_ac

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
    logger.debug('default coefficient for objectives: \n %s', ocp.cost.W.__str__())
    logger.debug('init parameters: %s', ocp.model.p.__str__())

    logger.debug('[Solver general settings.............]')
    logger.debug('Stages numbers: %d', ocp.dims.N)
    logger.debug('horizon of the problem: %d', ocp.solver_options.tf)
    logger.debug('step size:%f', ocp.solver_options.tf/ocp.dims.N)


    logger.debug('qp solver type:%s', ocp.solver_options.qp_solver)
    logger.debug('nlp solver type:%s', ocp.solver_options.nlp_solver_type)
    logger.debug('hessian_approx method:%s', ocp.solver_options.hessian_approx)
    logger.debug('integrator_type:%s', ocp.solver_options.integrator_type)
    logger.debug('sim_method_num_stages:%d', ocp.solver_options.sim_method_num_stages)
    logger.debug('sim_method_num_steps:%d', ocp.solver_options.sim_method_num_steps)
    logger.debug('print_level:%d', ocp.solver_options.print_level)
    logger.debug('tollerance:%f', ocp.solver_options.tol)

    logger.debug('init state:%s', ocp.constraints.x0.__str__())

def set_acados_model(stage_n, sf, l_coeff, dtheta_coeff, k_coeff, dk_coeff):

    # default state boundaries
    l_lower = -10
    l_upper = 10
    dtheta_lower = -1.5
    dtheta_upper = 1.5
    k_lower = -0.5
    k_upper = 0.5

    dk_lower = -0.5
    dk_upper = 0.5

    ocp = AcadosOcp()
    model = export_simple_frenet_model()
    model_ac = to_acados_model(model)
    ocp.model = model_ac

    # set the dimension of the problem
    ocp.dims.N = stage_n
    ocp.dims.nx = ocp.model.x.size()[0]
    ocp.dims.nu = ocp.model.u.size()[0]
    ocp.dims.ny = ocp.dims.nx + ocp.dims.nu
    ocp.dims.ny_e = 0  # not set for now
    ocp.dims.nbx = ocp.dims.nx  # number of state bounds
    ocp.dims.nbu = ocp.dims.nu  # number of control bounds
    ocp.dims.np = ocp.model.p.size()[0] # parameter size

    ocp.dims.nh = 0     # number of nonlinear constraints
    ocp.dims.nsh = 0    # number of soft nonlinear constraints
    ocp.dims.ns = 0     # total number of slacks
    ocp.dims.nsg = 1     # total number of slacks


    # cost functions and weights setting
    ocp.cost.cost_type = "LINEAR_LS"  #  cost type, default format
    Q = np.eye(ocp.dims.nx)
    Q[0, 0] = l_coeff
    Q[1, 1] = dtheta_coeff
    Q[2, 2] = k_coeff
    R = np.eye(ocp.dims.nu)
    R[0, 0] = dk_coeff
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    # set the cost function form
    Vx = np.zeros((ocp.dims.ny, ocp.dims.nx))
    Vx[:ocp.dims.nx, :ocp.dims.nx] = np.eye(ocp.dims.nx)
    Vu = np.zeros((ocp.dims.ny, ocp.dims.nu))
    Vu[ocp.dims.ny-1, 0] = 1
    ocp.cost.Vu = Vu
    ocp.cost.Vx = Vx
    ocp.cost.yref = np.array([0, 0, 0, 0])  # l, dtheta, k, dk
    # for slack variables
    # L2 penenaty
    ocp.cost.Zl = np.array([0])
    ocp.cost.Zu = np.array([0])
    # L1 penenaty
    ocp.cost.zl = np.array([100])
    ocp.cost.zu = np.array([100])

    # path constraints, C , D matrix and corresponding lower and upper bounds
    # lg <= C*X + D*U <= ug
    C = np.zeros((ocp.dims.ny, ocp.dims.nx))
    C[:ocp.dims.nx, :ocp.dims.nx] = np.eye(ocp.dims.nx)
    D = np.zeros((ocp.dims.ny, ocp.dims.nu))
    D[ocp.dims.ny - 1, 0] = 1  # in reality, these variables change according to the referenceline
    ocp.constraints.D = D
    ocp.constraints.C = C
    ocp.constraints.lg = np.array([l_lower, dtheta_lower, k_lower, dk_lower])
    ocp.constraints.ug = np.array([l_upper, dtheta_upper, k_upper, dk_upper])
    ocp.constraints.lsg = np.array([1.0])

    # state constraints
    ocp.constraints.lbx = np.array([l_lower, dtheta_lower, k_lower])
    ocp.constraints.ubx = np.array([l_upper, dtheta_upper, k_upper])
    ocp.constraints.idxbx = np.array([0, 1, 2])
    ocp.constraints.usg = np.array([1.0])
    ocp.constraints.idxsg = np.array([0])

    # upper bounds
    ocp.constraints.lbu = np.array([dk_lower])
    ocp.constraints.ubu = np.array([dk_upper])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = np.array([0, 0, 0])

    # ocp parameters default
    ocp.parameter_values = np.array([0, 0])

    # solver settings
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 1
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.print_level = 0
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.tf = sf

    output_debug_info_ocp(ocp)

    acados_solver = AcadosOcpSolver(ocp)

    return acados_solver

solver = set_acados_model(200, 100.0, 10, 0, 1000.0, 100.0)
