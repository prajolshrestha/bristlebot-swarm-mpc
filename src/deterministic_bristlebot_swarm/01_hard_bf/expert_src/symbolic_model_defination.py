"""Builds the CasADi optimal control problem: dynamics, costs and safety constraints."""

import math
import numpy as np
from casadi import *
from typing import Dict


def ellipse_robot_tracking_mpc(constraints: Dict[str, float]):
    """
    Build the CasADi symbolic model for trajectory-coupled DMPC
    with Reynolds Rules (alignment + cohesion as soft costs).

    Parameters
    ----------
    constraints : dict
        Must contain all keys from config/solver_ellipse_mpc.yaml under
        'model_bounds', 'collision', and optionally 'reynolds':
            px_min, px_max, py_min, py_max,
            th_min, th_max,
            v_min,  v_max,
            om_min, om_max,
            max_neighbours  (int  K)
            d_safe          (float, metres)
            q_cohesion      (float, default 5.0)
            q_align         (float, default 10.0)

    Returns
    -------
    model      : SimpleNamespace   CasADi / Acados model fields
    constraint : SimpleNamespace   (empty - constraints live in model)

    Parameter layout per stage k  (np = 12 + 5*K + 3*M)
    ────────────────────────────────────────────────────
      [0..9]             tracking + cost weights
      [10]               q_cohesion  (Reynolds)
      [11]               q_align     (Reynolds)
      [12 + 5*j + 0]     nb_j_px     position x
      [12 + 5*j + 1]     nb_j_py     position y
      [12 + 5*j + 2]     nb_j_vx     velocity x    (for alignment)
      [12 + 5*j + 3]     nb_j_vy     velocity y    (for alignment)
      [12 + 5*j + 4]     nb_j_active 1.0=active, 0.0=padded (on/off mask)
      obstacle blocks of [px, py, active] follow the neighbour blocks
    """
    model      = types.SimpleNamespace()
    constraint = types.SimpleNamespace()
    model_name = "ellipse_robot_tracking_mpc_tcoupled_reynolds"

    MAX_NEIGHBOURS = int(constraints["max_neighbours"])
    D_SAFE         = float(constraints["d_safe"])

    MAX_OBS        = int(constraints.get("max_obstacles", 5))
    D_SAFE_OBS     = float(constraints.get("d_safe_obs", 0.10))

    model.px_min = float(constraints["px_min"])
    model.px_max = float(constraints["px_max"])
    model.py_min = float(constraints["py_min"])
    model.py_max = float(constraints["py_max"])
    model.th_min = float(constraints["th_min"])
    model.th_max = float(constraints["th_max"])
    model.v_min  = float(constraints["v_min"])
    model.v_max  = float(constraints["v_max"])
    model.om_min = float(constraints["om_min"])
    model.om_max = float(constraints["om_max"])

    px = SX.sym("px")
    py = SX.sym("py")
    th = SX.sym("th")
    x  = vertcat(px, py, th)

    v_cmd  = SX.sym("v_cmd")
    om_cmd = SX.sym("om_cmd")
    u      = vertcat(v_cmd, om_cmd)

    px_dot = SX.sym("px_dot")
    py_dot = SX.sym("py_dot")
    th_dot = SX.sym("th_dot")
    xdot   = vertcat(px_dot, py_dot, th_dot)

    z = vertcat([])

    px_ref    = SX.sym("px_ref")
    py_ref    = SX.sym("py_ref")
    th_ref    = SX.sym("th_ref")
    v_ref     = SX.sym("v_ref")
    om_ref    = SX.sym("om_ref")
    q_pos     = SX.sym("q_pos")
    q_theta   = SX.sym("q_theta")
    q_vel     = SX.sym("q_vel")
    q_om      = SX.sym("q_om")
    r_u       = SX.sym("r_u")

    q_cohesion = SX.sym("q_cohesion")
    q_align    = SX.sym("q_align")

    nb_px     = [SX.sym(f"nb{j}_px")     for j in range(MAX_NEIGHBOURS)]
    nb_py     = [SX.sym(f"nb{j}_py")     for j in range(MAX_NEIGHBOURS)]
    nb_vx     = [SX.sym(f"nb{j}_vx")     for j in range(MAX_NEIGHBOURS)]
    nb_vy     = [SX.sym(f"nb{j}_vy")     for j in range(MAX_NEIGHBOURS)]
    nb_active = [SX.sym(f"nb{j}_active") for j in range(MAX_NEIGHBOURS)]

    obs_px     = [SX.sym(f"obs{m}_px")     for m in range(MAX_OBS)]
    obs_py     = [SX.sym(f"obs{m}_py")     for m in range(MAX_OBS)]
    obs_active = [SX.sym(f"obs{m}_active") for m in range(MAX_OBS)]

    nb_params = vertcat(*[
        vertcat(nb_px[j], nb_py[j], nb_vx[j], nb_vy[j], nb_active[j])
        for j in range(MAX_NEIGHBOURS)
    ])
    obs_params = vertcat(*[
        vertcat(obs_px[m], obs_py[m], obs_active[m])
        for m in range(MAX_OBS)
    ])
    p = vertcat(
        px_ref, py_ref, th_ref, v_ref, om_ref,
        q_pos, q_theta, q_vel, q_om, r_u,
        q_cohesion, q_align,
        nb_params,
        obs_params,
    )

    f_expl = vertcat(
        v_cmd * cos(th),
        v_cmd * sin(th),
        om_cmd,
    )

    e_px  = px    - px_ref
    e_py  = py    - py_ref
    e_th  = atan2(sin(th - th_ref), cos(th - th_ref))
    e_v   = v_cmd - v_ref
    e_om  = om_cmd - om_ref

    cost_tracking = (
        e_px   * e_px   * q_pos
        + e_py   * e_py   * q_pos
        + e_th   * e_th   * q_theta
        + e_v    * e_v    * q_vel
        + e_om   * e_om   * q_om
        + v_cmd  * v_cmd  * r_u
        + om_cmd * om_cmd * r_u
    )

    eps = 1e-4

    sum_active  = fmax(sum1(vertcat(*nb_active)), eps)
    centroid_x  = sum1(vertcat(*[nb_active[j] * nb_px[j] for j in range(MAX_NEIGHBOURS)])) / sum_active
    centroid_y  = sum1(vertcat(*[nb_active[j] * nb_py[j] for j in range(MAX_NEIGHBOURS)])) / sum_active

    dx_coh = px - centroid_x
    dy_coh = py - centroid_y

    any_active  = fmin(sum_active / (eps + 1.0), 1.0)
    cost_cohesion = q_cohesion * any_active * (dx_coh * dx_coh + dy_coh * dy_coh)

    sum_vx  = sum1(vertcat(*[nb_active[j] * nb_vx[j] for j in range(MAX_NEIGHBOURS)]))
    sum_vy  = sum1(vertcat(*[nb_active[j] * nb_vy[j] for j in range(MAX_NEIGHBOURS)]))
    th_mean = atan2(sum_vy + eps, sum_vx + eps)

    e_align       = atan2(sin(th - th_mean), cos(th - th_mean))
    cost_alignment = q_align * any_active * (e_align * e_align)

    model.cost_expr_ext_cost = cost_tracking + cost_cohesion + cost_alignment

    model.cost_expr_ext_cost_e = (
        e_px * e_px * q_pos
        + e_py * e_py * q_pos
        + e_th * e_th * q_theta
    )

    h_list = []
    for j in range(MAX_NEIGHBOURS):
        dx = px - nb_px[j]
        dy = py - nb_py[j]
        h_list.append(dx * dx + dy * dy - D_SAFE ** 2)

    for m in range(MAX_OBS):
        dx = px - obs_px[m]
        dy = py - obs_py[m]
        h_list.append(dx * dx + dy * dy - D_SAFE_OBS ** 2)

    model.con_h_expr   = vertcat(*h_list)
    model.con_h_expr_e = vertcat(*h_list)

    model.MAX_NEIGHBOURS = MAX_NEIGHBOURS
    model.MAX_OBS        = MAX_OBS
    model.D_SAFE         = D_SAFE
    model.D_SAFE_OBS     = D_SAFE_OBS
    model.NB_STRIDE      = 5
    model.OBS_STRIDE     = 3
    model.NP_BASE        = 12

    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x           = x
    model.xdot        = xdot
    model.u           = u
    model.z           = z
    model.p           = p
    model.name        = model_name

    params = types.SimpleNamespace()
    params.px_ref      = px_ref
    params.py_ref      = py_ref
    params.th_ref      = th_ref
    params.v_ref       = v_ref
    params.om_ref      = om_ref
    params.q_pos       = q_pos
    params.q_theta     = q_theta
    params.q_vel       = q_vel
    params.q_om        = q_om
    params.r_u         = r_u
    params.q_cohesion  = q_cohesion
    params.q_align     = q_align

    NB_STRIDE = 5
    OBS_STRIDE = 3
    NP_BASE   = 12
    params.IDX_Q_COHESION  = 10
    params.IDX_Q_ALIGN     = 11
    params.IDX_NB_PX       = lambda j: NP_BASE + NB_STRIDE * j + 0
    params.IDX_NB_PY       = lambda j: NP_BASE + NB_STRIDE * j + 1
    params.IDX_NB_VX       = lambda j: NP_BASE + NB_STRIDE * j + 2
    params.IDX_NB_VY       = lambda j: NP_BASE + NB_STRIDE * j + 3
    params.IDX_NB_ACTIVE   = lambda j: NP_BASE + NB_STRIDE * j + 4
    
    OBS_START = NP_BASE + NB_STRIDE * MAX_NEIGHBOURS
    params.IDX_OBS_START   = OBS_START
    params.IDX_OBS_PX      = lambda m: OBS_START + OBS_STRIDE * m + 0
    params.IDX_OBS_PY      = lambda m: OBS_START + OBS_STRIDE * m + 1
    params.IDX_OBS_ACTIVE  = lambda m: OBS_START + OBS_STRIDE * m + 2
    params.NP_TOTAL        = OBS_START + OBS_STRIDE * MAX_OBS

    model.params = params

    return model, constraint