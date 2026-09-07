"""Builds the CasADi optimal control problem: dynamics, costs and safety constraints."""

import types
import numpy as np
from casadi import *
from typing import Dict


def ellipse_robot_tracking_mpc(constraints: Dict[str, float]):
    """
    Distributed Predictive Flocking with a DISCRETE-TIME Higher-Order CBF (Dt-HOCBF-MPC).

    Why a DISCRETE HOCBF (vs variant 09's continuous one)
    -----------------------------------------------------
    The robot is a kinematic unicycle: x=[px,py,th], u=[v,om]. The collision
    barrier h_j = ||p - p_j||^2 - D_safe^2 has RELATIVE DEGREE 2: position
    responds to v at relative degree 1, but to the turn-rate om only at relative
    degree 2. A one-step discrete CBF (variants 02/06/07) leaves om with ZERO
    gradient on the barrier (om only changes th_next, position two steps later),
    so the robot can only slow/stop — it cannot STEER. Variant 09 fixes this with
    the CONTINUOUS-time HOCBF chain (Lie derivatives, v*om cross-term). This
    variant instead builds the chain from ONE-STEP DIFFERENCES — the discrete
    analogue (cf. discrete exponential CBFs, Zeng-style) — so the certificate is
    stated on exactly the sampled states the discrete-time MPC controls.

    Dt-HOCBF chain (one-step differences, gamma1/gamma2 in (0, 1])
    ────────────────────────────────────────────────────────────────
        psi1(x) = h(x_hat1) - (1 - gamma1) * h(x)
        psi2    = psi1(x_hat1) - (1 - gamma2) * psi1(x)
                = h(x_hat2) - (2-g1-g2)*h(x_hat1) + (1-g1)*(1-g2)*h(x_k)  >= 0

    with the TWO-STEP explicit-Euler prediction under constant (v, om) over
    both substeps (u_{k+1} := u_k — acados stage constraints couple only
    (x_k, u_k), so the next stage's input is not available in the row):

        th_hat1 = th + Ts*om
        p_hat1  = p + Ts*v*[cos th,      sin th     ]
        p_hat2  = p_hat1 + Ts*v*[cos th_hat1, sin th_hat1]

    om enters h(x_hat2) through th_hat1:
        d h(x_hat2) / d om  =  2*Ts^2*v*( -rx2*sin th_hat1 + ry2*cos th_hat1 )
    — the discrete analogue of variant 09's 2*v*om*(-rx sin th + ry cos th)
    term, scaled by Ts^2. NOTE the per-row om gradient is O(Ts^2 * v) (weak);
    gamma and slack tuning matter, and the N-stage horizon supplies the rest of
    the steering authority.

    Invariance chain (discrete): psi2 >= 0 keeps psi1 >= 0 (given psi1_0 >= 0,
    gamma2 in (0,1]); psi1 >= 0 gives h_{k+1} >= (1-gamma1)*h_k, keeping h >= 0.
    Per-stage initial-condition requirements are covered by the slacked BF
    anchor rows below. Degenerate ends: gamma -> 1 collapses psi2 to the pure
    two-step lookahead barrier h(x_hat2) >= 0; gamma -> 0 approaches the raw
    second difference (~ Ts^2 * h_ddot, no decay — too permissive).

    ASSUMPTIONS: neighbour locally constant-velocity over both substeps
    (nb_hat1 = nb + Ts*nb_v, nb_hat2 = nb + 2*Ts*nb_v; obstacle velocity = 0);
    ego (v, om) constant over both substeps (mirrors variant 09's v_dot = 0).

    Hybrid layout (kept IDENTICAL to variants 07/09 so the generate script and
    controller slack-vector slicing work unchanged):
      Row 0..K-1:        h_j(x_k)  >= 0                 (slacked BF anchor, robots)
      Row K..2K-1:       psi2_j >= 0                    (slacked Dt-HOCBF, robots)
      Row 2K..2K+M-1:    h_o(x_k) >= 0                  (slacked BF anchor, obs)
      Row 2K+M..2K+2M-1: psi2_o >= 0                    (slacked Dt-HOCBF, obs)
    Terminal (k=N):  h_j(x_N) >= 0  and  h_o(x_N) >= 0  (simple barriers, no u).

    Parameter layout per stage k  (np = 12 + 5*K + 3*M + 1)  — IDENTICAL to variant 07
    ──────────────────────────────────────────────────────
      [0..9]             tracking + cost weights
      [10]               q_cohesion  (Reynolds)
      [11]               q_align     (Reynolds)
      [12 + 5*j + 0..4]  nb_j: px, py, vx, vy, active
      [12 + 5*K + 3*m + 0..2]  obs_m: px, py, active
      [12 + 5*K + 3*M]         avoid_on  1.0=avoidance on, 0.0=all rows vanish

    Dt-HOCBF decay rates (hocbf_gamma1/2 robots, hocbf_gamma1/2_obs obstacles,
    each in (0,1], default 0.5) are compiled into the solver from YAML — no
    extra runtime parameters, NP_BASE=12. Changing gamma requires a recompile.
    """
    model      = types.SimpleNamespace()
    constraint = types.SimpleNamespace()
    model_name = "ellipse_robot_tracking_mpc_dt_hocbf"

    MAX_NEIGHBOURS = int(constraints["max_neighbours"])
    D_SAFE         = float(constraints["d_safe"])
    TS             = float(constraints.get("Ts", 0.1))
    CBF_ALPHA      = float(constraints.get("cbf_alpha",
                            constraints.get("alpha_cbf", 0.5)))
    W_SOFT_DEFAULT = float(constraints.get("w_soft",        1e4))
    W_SOFT_LIN     = float(constraints.get("w_soft_lin",    0.0))

    HOCBF_GAMMA1   = float(constraints.get("hocbf_gamma1",  0.5))
    HOCBF_GAMMA2   = float(constraints.get("hocbf_gamma2",  0.5))

    MAX_OBS        = int(constraints.get("max_obstacles",   5))
    D_SAFE_OBS     = float(constraints.get("d_safe_obs",    0.10))
    CBF_ALPHA_OBS  = float(constraints.get("cbf_alpha_obs",
                            constraints.get("alpha_cbf_obs", CBF_ALPHA)))
    W_SOFT_OBS_DEF = float(constraints.get("w_soft_obs",    1e4))
    W_SOFT_OBS_LIN = float(constraints.get("w_soft_obs_lin",0.0))

    HOCBF_GAMMA1_OBS = float(constraints.get("hocbf_gamma1_obs", HOCBF_GAMMA1))
    HOCBF_GAMMA2_OBS = float(constraints.get("hocbf_gamma2_obs", HOCBF_GAMMA2))

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

    f_expl = vertcat(
        v_cmd * cos(th),
        v_cmd * sin(th),
        om_cmd,
    )

    th_next  = th + TS * om_cmd
    px_next  = px + TS * v_cmd * cos(th)
    py_next  = py + TS * v_cmd * sin(th)
    px_next2 = px_next + TS * v_cmd * cos(th_next)
    py_next2 = py_next + TS * v_cmd * sin(th_next)

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

    def _dt_hocbf_psi2(pjx, pjy, vjx, vjy, d_sq, gamma1, gamma2):
        """Discrete-time HOCBF psi2 >= 0 from one-step differences.

        pjx, pjy : neighbour/obstacle position at stage k (parameters)
        vjx, vjy : neighbour velocity (constant-velocity model); 0 for obstacles
        d_sq     : D_safe^2
        gamma1/2 : discrete decay rates in (0, 1]

        psi1(x) = h(x_hat1) - (1-gamma1)*h(x)
        psi2    = psi1(x_hat1) - (1-gamma2)*psi1(x)
                = h2 - (2-g1-g2)*h1 + (1-g1)*(1-g2)*h0
        where h0/h1/h2 evaluate the barrier at (x_k, x_hat1, x_hat2) with the
        neighbour advanced by its reported velocity (1 and 2 steps). The ego
        two-step prediction uses constant (v, om); om enters h2 through th_next.
        """
        dx0 = px - pjx
        dy0 = py - pjy
        h0  = dx0 * dx0 + dy0 * dy0 - d_sq
        dx1 = px_next - (pjx + TS * vjx)
        dy1 = py_next - (pjy + TS * vjy)
        h1  = dx1 * dx1 + dy1 * dy1 - d_sq
        dx2 = px_next2 - (pjx + 2.0 * TS * vjx)
        dy2 = py_next2 - (pjy + 2.0 * TS * vjy)
        h2  = dx2 * dx2 + dy2 * dy2 - d_sq
        psi2 = (h2
                - (2.0 - gamma1 - gamma2) * h1
                + (1.0 - gamma1) * (1.0 - gamma2) * h0)
        return psi2

    h_stage_list = []

    for j in range(MAX_NEIGHBOURS):
        dx_j = px - nb_px[j]
        dy_j = py - nb_py[j]
        h_j  = dx_j * dx_j + dy_j * dy_j - D_SAFE_SQ
        h_j_blended = nb_avoid[j] * h_j + (1.0 - nb_avoid[j]) * BIG
        h_stage_list.append(h_j_blended)

    for j in range(MAX_NEIGHBOURS):
        psi2_j = _dt_hocbf_psi2(nb_px[j], nb_py[j], nb_vx[j], nb_vy[j],
                                D_SAFE_SQ, HOCBF_GAMMA1, HOCBF_GAMMA2)
        psi2_j_blended = nb_avoid[j] * psi2_j + (1.0 - nb_avoid[j]) * BIG
        h_stage_list.append(psi2_j_blended)

    for m in range(MAX_OBS):
        dx_m = px - obs_px[m]
        dy_m = py - obs_py[m]
        h_m  = dx_m * dx_m + dy_m * dy_m - D_OBS_SQ
        h_m_blended = obs_avoid[m] * h_m + (1.0 - obs_avoid[m]) * BIG
        h_stage_list.append(h_m_blended)

    for m in range(MAX_OBS):
        psi2_m = _dt_hocbf_psi2(obs_px[m], obs_py[m], SX(0.0), SX(0.0),
                                D_OBS_SQ, HOCBF_GAMMA1_OBS, HOCBF_GAMMA2_OBS)
        psi2_m_blended = obs_avoid[m] * psi2_m + (1.0 - obs_avoid[m]) * BIG
        h_stage_list.append(psi2_m_blended)

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
    model.HOCBF_GAMMA1     = HOCBF_GAMMA1
    model.HOCBF_GAMMA2     = HOCBF_GAMMA2
    model.HOCBF_GAMMA1_OBS = HOCBF_GAMMA1_OBS
    model.HOCBF_GAMMA2_OBS = HOCBF_GAMMA2_OBS
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