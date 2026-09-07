"""The same optimal control problem, with the centre-of-rotation offset in the dynamics."""

import types
import numpy as np
from casadi import *
from typing import Dict


def ellipse_robot_tracking_mpc(constraints: Dict[str, float]):
    """
    Distributed Predictive Flocking with Control Barrier Functions (CBF-MPC).

    Safety formulation (hybrid: barrier + discrete-time CBF)
    -------------------------------------------------------
    Barrier per neighbour j / obstacle m:
        h_j(x) = ||p - p_j||² - D_safe²          (> 0 = safe)

    At each stage k = 0..N-1 two slacked rows are enforced:
        h_j(x_k)                          ≥ 0    (barrier anchor)
        h_j(x̂_{k+1}) - (1-α)·h_j(x_k)    ≥ 0    (discrete CBF, one-step Euler)

    x̂_{k+1} is the one-step Euler prediction of the robot's position (neighbour
    positions are advanced with their reported velocity). α is the CBF decay rate:
    larger α relaxes the look-ahead, smaller α tightens it.

    Both rows are softened via acados con_h_expr / nsh slack variables for
    guaranteed feasibility. The slack penalties (Zl, zl) are set in
    generate_solver_via_acados.py and can be updated at runtime via
    solver.cost_set(k, 'Zl', ...).

    TERMINAL STAGE: con_h_expr_e enforces only the simple barrier h_j(x_N) ≥ 0
    (no discrete CBF, since there is no control input at the terminal stage), so
    the predicted terminal position stays at least D_safe from all neighbours.

    Parameter layout per stage k  (np = 12 + 5*K + 3*M + 1)
    ──────────────────────────────────────────────────────
      [0..9]             tracking + cost weights
      [10]               q_cohesion  (Reynolds)
      [11]               q_align     (Reynolds)
      [12 + 5*j + 0]     nb_j_px
      [12 + 5*j + 1]     nb_j_py
      [12 + 5*j + 2]     nb_j_vx    (alignment + CBF neighbour advance)
      [12 + 5*j + 3]     nb_j_vy    (alignment + CBF neighbour advance)
      [12 + 5*j + 4]     nb_j_active  1.0=active, 0.0=padded
      [12 + 5*K + 3*m + 0]  obs_m_px
      [12 + 5*K + 3*m + 1]  obs_m_py
      [12 + 5*K + 3*m + 2]  obs_m_active
      [12 + 5*K + 3*M]      avoid_on   1.0=avoidance on, 0.0=all rows vanish

    NOTE: cbf_alpha is compiled into the solver (CBF_ALPHA, CBF_ALPHA_OBS).
    NP_BASE = 12 unchanged from the slack version.

    Constraint layout  (con_h_expr  nh = 2·(K + M))
    ────────────────────────────────────────────────
      Stage (k=0..N-1):
        Row 0..K-1:         h_j(x_k)                        ≥ 0  (BF anchor, robots)
        Row K..2K-1:        h_j(x̂_{k+1}) - (1-α)·h_j(x_k)  ≥ 0  (Dt-CBF, robots)
        Row 2K..2K+M-1:     h_m(x_k)                        ≥ 0  (BF anchor, obstacles)
        Row 2K+M..2(K+M)-1: h_m(x̂_{k+1}) - (1-α_obs)·h_m(x_k) ≥ 0  (Dt-CBF, obstacles)
      Terminal (k=N, nh_e = K + M):  h_j(x_N) ≥ 0  and  h_m(x_N) ≥ 0
    """
    model      = types.SimpleNamespace()
    constraint = types.SimpleNamespace()
    model_name = "ellipse_robot_tracking_mpc_cbf_reynolds_obs_off"

    MAX_NEIGHBOURS = int(constraints["max_neighbours"])
    D_SAFE         = float(constraints["d_safe"])
    TS             = float(constraints.get("Ts", 0.1))
    CBF_ALPHA      = float(constraints.get("cbf_alpha",
                            constraints.get("alpha_cbf", 0.5)))
    W_SOFT_DEFAULT = float(constraints.get("w_soft",        1e4))
    W_SOFT_LIN     = float(constraints.get("w_soft_lin",    0.0))

    MAX_OBS        = int(constraints.get("max_obstacles",   5))
    D_SAFE_OBS     = float(constraints.get("d_safe_obs",    0.10))
    CBF_ALPHA_OBS  = float(constraints.get("cbf_alpha_obs",
                            constraints.get("alpha_cbf_obs", CBF_ALPHA)))
    W_SOFT_OBS_DEF = float(constraints.get("w_soft_obs",    1e4))
    W_SOFT_OBS_LIN = float(constraints.get("w_soft_obs_lin",0.0))

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

    px_ref  = SX.sym("px_ref")
    py_ref  = SX.sym("py_ref")
    th_ref  = SX.sym("th_ref")
    v_ref   = SX.sym("v_ref")
    om_ref  = SX.sym("om_ref")
    q_pos   = SX.sym("q_pos")
    q_theta = SX.sym("q_theta")
    q_vel   = SX.sym("q_vel")
    q_om    = SX.sym("q_om")
    r_u     = SX.sym("r_u")

    q_cohesion = SX.sym("q_cohesion")
    q_align    = SX.sym("q_align")
    avoid_on   = SX.sym("avoid_on")

    nb_px     = [SX.sym(f"nb{j}_px")     for j in range(MAX_NEIGHBOURS)]
    nb_py     = [SX.sym(f"nb{j}_py")     for j in range(MAX_NEIGHBOURS)]
    nb_vx     = [SX.sym(f"nb{j}_vx")     for j in range(MAX_NEIGHBOURS)]
    nb_vy     = [SX.sym(f"nb{j}_vy")     for j in range(MAX_NEIGHBOURS)]
    nb_active = [SX.sym(f"nb{j}_active") for j in range(MAX_NEIGHBOURS)]

    obs_px     = [SX.sym(f"obs{m}_px")     for m in range(MAX_OBS)]
    obs_py     = [SX.sym(f"obs{m}_py")     for m in range(MAX_OBS)]
    obs_active = [SX.sym(f"obs{m}_active") for m in range(MAX_OBS)]

    nb_params  = vertcat(*[
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
        avoid_on,
    )

    d_cw  = 0.0145
    d_ccw = 0.0161
    perp_x = -sin(th)
    perp_y =  cos(th)
    off_x = if_else(om_cmd > 0, d_ccw * perp_x, -d_cw * perp_x)
    off_y = if_else(om_cmd > 0, d_ccw * perp_y, -d_cw * perp_y)

    f_expl = vertcat(
        v_cmd * cos(th) + om_cmd * off_y,
        v_cmd * sin(th) - om_cmd * off_x,
        om_cmd,
    )

    px_next = px + TS * (v_cmd * cos(th) + om_cmd * off_y)
    py_next = py + TS * (v_cmd * sin(th) - om_cmd * off_x)

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

    eps        = 1e-4
    sum_active = fmax(sum1(vertcat(*nb_active)), eps)
    centroid_x = sum1(vertcat(*[nb_active[j] * nb_px[j] for j in range(MAX_NEIGHBOURS)])) / sum_active
    centroid_y = sum1(vertcat(*[nb_active[j] * nb_py[j] for j in range(MAX_NEIGHBOURS)])) / sum_active

    dx_coh        = px - centroid_x
    dy_coh        = py - centroid_y
    any_active    = fmin(sum_active, 1.0)
    cost_cohesion = q_cohesion * any_active * (dx_coh * dx_coh + dy_coh * dy_coh)

    sum_vx  = sum1(vertcat(*[nb_active[j] * nb_vx[j] for j in range(MAX_NEIGHBOURS)]))
    sum_vy  = sum1(vertcat(*[nb_active[j] * nb_vy[j] for j in range(MAX_NEIGHBOURS)]))
    th_mean = atan2(sum_vy + eps, sum_vx + eps)

    e_align        = atan2(sin(th - th_mean), cos(th - th_mean))
    cost_alignment = q_align * any_active * (e_align * e_align)

    model.cost_expr_ext_cost = cost_tracking + cost_cohesion + cost_alignment

    tracking_cost_e = (
        e_px * e_px * q_pos
        + e_py * e_py * q_pos
        + e_th * e_th * q_theta
    )
    model.cost_expr_ext_cost_e = tracking_cost_e

    D_SAFE_SQ = D_SAFE     ** 2
    D_OBS_SQ  = D_SAFE_OBS ** 2
    BIG       = SX(100.0)

    nb_avoid  = [nb_active[j] * avoid_on for j in range(MAX_NEIGHBOURS)]
    obs_avoid = [obs_active[m] * avoid_on for m in range(MAX_OBS)]

    h_stage_list = []

    for j in range(MAX_NEIGHBOURS):
        dx_j = px - nb_px[j]
        dy_j = py - nb_py[j]
        h_j  = dx_j * dx_j + dy_j * dy_j - D_SAFE_SQ
        h_j_blended = nb_avoid[j] * h_j + (1.0 - nb_avoid[j]) * BIG
        h_stage_list.append(h_j_blended)

    for j in range(MAX_NEIGHBOURS):
        dx_j = px - nb_px[j]
        dy_j = py - nb_py[j]
        h_j  = dx_j * dx_j + dy_j * dy_j - D_SAFE_SQ

        nb_px_n = nb_px[j] + TS * nb_vx[j]
        nb_py_n = nb_py[j] + TS * nb_vy[j]
        dx_n    = px_next - nb_px_n
        dy_n    = py_next - nb_py_n
        h_j_n   = dx_n * dx_n + dy_n * dy_n - D_SAFE_SQ

        cbf_j         = h_j_n - (1.0 - CBF_ALPHA) * h_j
        cbf_j_blended = nb_avoid[j] * cbf_j + (1.0 - nb_avoid[j]) * BIG
        h_stage_list.append(cbf_j_blended)

    for m in range(MAX_OBS):
        dx_m = px - obs_px[m]
        dy_m = py - obs_py[m]
        h_m  = dx_m * dx_m + dy_m * dy_m - D_OBS_SQ
        h_m_blended = obs_avoid[m] * h_m + (1.0 - obs_avoid[m]) * BIG
        h_stage_list.append(h_m_blended)

    for m in range(MAX_OBS):
        dx_m = px - obs_px[m]
        dy_m = py - obs_py[m]
        h_m  = dx_m * dx_m + dy_m * dy_m - D_OBS_SQ

        dx_n = px_next - obs_px[m]
        dy_n = py_next - obs_py[m]
        h_m_n = dx_n * dx_n + dy_n * dy_n - D_OBS_SQ

        cbf_m         = h_m_n - (1.0 - CBF_ALPHA_OBS) * h_m
        cbf_m_blended = obs_avoid[m] * cbf_m + (1.0 - obs_avoid[m]) * BIG
        h_stage_list.append(cbf_m_blended)

    nh_total   = 2 * (MAX_NEIGHBOURS + MAX_OBS)
    con_h_expr = vertcat(*h_stage_list)

    h_terminal_list = []

    for j in range(MAX_NEIGHBOURS):
        dx_j  = px - nb_px[j]
        dy_j  = py - nb_py[j]
        h_j_e = dx_j * dx_j + dy_j * dy_j - D_SAFE_SQ
        h_j_e_blended = nb_avoid[j] * h_j_e + (1.0 - nb_avoid[j]) * BIG
        h_terminal_list.append(h_j_e_blended)

    for m in range(MAX_OBS):
        dx_m  = px - obs_px[m]
        dy_m  = py - obs_py[m]
        h_m_e = dx_m * dx_m + dy_m * dy_m - D_OBS_SQ
        h_m_e_blended = obs_avoid[m] * h_m_e + (1.0 - obs_avoid[m]) * BIG
        h_terminal_list.append(h_m_e_blended)

    con_h_expr_e = vertcat(*h_terminal_list)

    lb_h   = np.zeros(nh_total)
    lb_h_e = np.zeros(nh_total)

    model.con_h_expr   = con_h_expr
    model.con_h_expr_e = con_h_expr_e

    constraint.con_h_expr   = con_h_expr
    constraint.con_h_expr_e = con_h_expr_e
    constraint.lb_h         = lb_h
    constraint.lb_h_e       = lb_h_e
    constraint.nh_total     = nh_total
    constraint.nh_robots    = 2 * MAX_NEIGHBOURS
    constraint.nh_obs       = 2 * MAX_OBS

    NB_STRIDE  = 5
    OBS_STRIDE = 3
    NP_BASE    = 12

    model.MAX_NEIGHBOURS  = MAX_NEIGHBOURS
    model.MAX_OBS         = MAX_OBS
    model.D_SAFE          = D_SAFE
    model.D_SAFE_OBS      = D_SAFE_OBS
    model.CBF_ALPHA       = CBF_ALPHA
    model.CBF_ALPHA_OBS   = CBF_ALPHA_OBS
    model.TS              = TS
    model.W_SOFT_DEFAULT  = W_SOFT_DEFAULT
    model.W_SOFT_LIN      = W_SOFT_LIN
    model.W_SOFT_OBS_DEF  = W_SOFT_OBS_DEF
    model.W_SOFT_OBS_LIN  = W_SOFT_OBS_LIN
    model.NB_STRIDE       = NB_STRIDE
    model.OBS_STRIDE      = OBS_STRIDE
    model.NP_BASE         = NP_BASE

    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x           = x
    model.xdot        = xdot
    model.u           = u
    model.z           = z
    model.p           = p
    model.name        = model_name

    params = types.SimpleNamespace()
    params.IDX_Q_COHESION = 10
    params.IDX_Q_ALIGN    = 11
    params.IDX_NB_START   = NP_BASE
    params.IDX_NB_PX      = lambda j: NP_BASE + NB_STRIDE * j + 0
    params.IDX_NB_PY      = lambda j: NP_BASE + NB_STRIDE * j + 1
    params.IDX_NB_VX      = lambda j: NP_BASE + NB_STRIDE * j + 2
    params.IDX_NB_VY      = lambda j: NP_BASE + NB_STRIDE * j + 3
    params.IDX_NB_ACTIVE  = lambda j: NP_BASE + NB_STRIDE * j + 4
    OBS_START = NP_BASE + NB_STRIDE * MAX_NEIGHBOURS
    params.IDX_OBS_START  = OBS_START
    params.IDX_OBS_PX     = lambda m: OBS_START + OBS_STRIDE * m + 0
    params.IDX_OBS_PY     = lambda m: OBS_START + OBS_STRIDE * m + 1
    params.IDX_OBS_ACTIVE = lambda m: OBS_START + OBS_STRIDE * m + 2
    params.IDX_AVOID_ON   = OBS_START + OBS_STRIDE * MAX_OBS
    params.NP_TOTAL       = OBS_START + OBS_STRIDE * MAX_OBS + 1
    params.IDX_H_ROBOT    = lambda j: j
    params.IDX_H_OBS      = lambda m: MAX_NEIGHBOURS + m
    params.NH_TOTAL       = nh_total
    params.NH_ROBOTS      = MAX_NEIGHBOURS
    params.NH_OBS         = MAX_OBS

    model.params = params

    return model, constraint