"""Compiles the optimal control problem into an acados solver."""


import argparse
import json
import os

import numpy as np
import yaml

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from symbolic_model_defination import ellipse_robot_tracking_mpc


NB_STRIDE  = 5
OBS_STRIDE = 3
NP_BASE    = 12



def build_stage_param_vector(
    px_ref: float,
    py_ref: float,
    th_ref: float,
    v_ref:  float,
    om_ref: float,
    q_pos:  float,
    q_theta: float,
    q_vel:  float,
    q_om:   float,
    r_u:    float,
    q_cohesion: float,
    q_align:    float,
    neighbour_states_at_stage_k: np.ndarray,
    obstacle_states_at_stage_k:  np.ndarray,
    max_neighbours: int,
    max_obstacles:  int,
    padding_position: float = 10.0,
) -> np.ndarray:
    """
    Build the parameter vector p_stage_k ∈ R^(12 + 5*K + 3*M) for stage k.

    Parameters
    ----------
    q_cohesion, q_align : float
        Reynolds weight scalars.
    neighbour_states_at_stage_k : ndarray, shape (n_nb, 4)
        Predicted [px, py, vx, vy] of each active neighbour at MPC stage k.
    obstacle_states_at_stage_k : ndarray, shape (n_obs, 2)
        Predicted [px, py] of each nearby obstacle at MPC stage k.
    padding_position : float
        Value used for inactive position slots (far outside arena).

    Returns
    -------
    p : ndarray, shape (12 + 5*max_neighbours + 3*max_obstacles,)

    Note
    ----
    w_soft and w_soft_obs are NO LONGER part of the parameter vector.
    They are encoded as slack costs (Zl/zl) compiled into the acados solver
    and can be updated at runtime via solver.cost_set(stage, 'Zl', ...).
    """
    np_total = NP_BASE + NB_STRIDE * max_neighbours + OBS_STRIDE * max_obstacles
    p = np.zeros(np_total)

    p[0] = px_ref
    p[1] = py_ref
    p[2] = th_ref
    p[3] = v_ref
    p[4] = om_ref
    p[5] = q_pos
    p[6] = q_theta
    p[7] = q_vel
    p[8] = q_om
    p[9] = r_u

    p[10] = q_cohesion
    p[11] = q_align

    n_active_nb = min(len(neighbour_states_at_stage_k), max_neighbours)
    for j in range(max_neighbours):
        base = NP_BASE + NB_STRIDE * j
        if j < n_active_nb:
            nb = neighbour_states_at_stage_k[j]
            p[base + 0] = nb[0]
            p[base + 1] = nb[1]
            p[base + 2] = nb[2]
            p[base + 3] = nb[3]
            p[base + 4] = 1.0
        else:
            p[base + 0] = padding_position
            p[base + 1] = padding_position
            p[base + 2] = 0.0
            p[base + 3] = 0.0
            p[base + 4] = 0.0

    OBS_START = NP_BASE + NB_STRIDE * max_neighbours
    n_active_obs = min(len(obstacle_states_at_stage_k), max_obstacles)
    for m in range(max_obstacles):
        base = OBS_START + OBS_STRIDE * m
        if m < n_active_obs:
            obs = obstacle_states_at_stage_k[m]
            p[base + 0] = obs[0]
            p[base + 1] = obs[1]
            p[base + 2] = 1.0
        else:
            p[base + 0] = padding_position
            p[base + 1] = padding_position
            p[base + 2] = 0.0

    return p



def build_acados_solver(
    N:                   int,
    Ts:                  float,
    acados_source_path:  str,
    model_factory,
    solver_opts_override: dict = None,
):

    if solver_opts_override is None:
        solver_opts_override = {}

    ocp = AcadosOcp()

    model, constraint = model_factory()

    nh      = model.MAX_NEIGHBOURS
    MAX_OBS = model.MAX_OBS
    nx      = model.x.size()[0]
    nu      = model.u.size()[0]
    np_dim  = model.p.size()[0]

    nh_total = constraint.nh_total

    model_ac = AcadosModel()
    model_ac.f_impl_expr  = model.f_impl_expr
    model_ac.f_expl_expr  = model.f_expl_expr
    model_ac.x            = model.x
    model_ac.xdot         = model.xdot
    model_ac.u            = model.u
    model_ac.z            = model.z
    model_ac.p            = model.p
    model_ac.name         = model.name

    model_ac.con_h_expr   = constraint.con_h_expr
    model_ac.con_h_expr_e = constraint.con_h_expr

    ocp.model = model_ac

    ocp.dims.nx     = nx
    ocp.dims.np     = np_dim
    ocp.dims.ny     = 0
    ocp.dims.ny_e   = 0
    ocp.dims.nu     = nu

    ocp.dims.nbx    = 2
    ocp.dims.nbx_e  = 2
    ocp.dims.nbx_0  = nx
    ocp.dims.nbxe_0 = nx

    ocp.dims.nbu    = nu

    ocp.dims.nh     = nh_total
    ocp.dims.nh_e   = nh_total
    ocp.dims.nsh    = nh_total
    ocp.dims.nsh_e  = nh_total

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model.cost_expr_ext_cost   = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    w_soft     = model.W_SOFT_DEFAULT
    w_soft_lin = model.W_SOFT_LIN
    w_obs      = model.W_SOFT_OBS_DEF
    w_obs_lin  = model.W_SOFT_OBS_LIN

    Zl_stage = np.zeros(nh_total)
    Zl_stage[:nh]        = w_soft
    Zl_stage[nh:]        = w_obs

    Zu_stage = np.zeros(nh_total)

    zl_stage = np.zeros(nh_total)
    zl_stage[:nh]        = w_soft_lin
    zl_stage[nh:]        = w_obs_lin

    zu_stage = np.zeros(nh_total)

    ocp.cost.Zl   = Zl_stage
    ocp.cost.Zu   = Zu_stage
    ocp.cost.zl   = zl_stage
    ocp.cost.zu   = zu_stage
    ocp.cost.Zl_e = Zl_stage.copy()
    ocp.cost.Zu_e = Zu_stage.copy()
    ocp.cost.zl_e = zl_stage.copy()
    ocp.cost.zu_e = zu_stage.copy()

    ocp.constraints.idxsh   = np.arange(nh_total)
    ocp.constraints.idxsh_e = np.arange(nh_total)

    ocp.constraints.lbx    = np.array([model.px_min, model.py_min])
    ocp.constraints.ubx    = np.array([model.px_max, model.py_max])
    ocp.constraints.idxbx  = np.array([0, 1])

    ocp.constraints.lbx_e  = np.array([model.px_min, model.py_min])
    ocp.constraints.ubx_e  = np.array([model.px_max, model.py_max])
    ocp.constraints.idxbx_e = np.array([0, 1])

    ocp.constraints.x0       = np.zeros(nx)
    ocp.constraints.idxbxe_0 = np.arange(nx)

    ocp.constraints.lbu   = np.array([model.v_min,  model.om_min])
    ocp.constraints.ubu   = np.array([model.v_max,  model.om_max])
    ocp.constraints.idxbu = np.arange(nu)

    lh = constraint.lb_h.copy()
    uh = 1e9 * np.ones(nh_total)

    ocp.constraints.lh   = lh
    ocp.constraints.uh   = uh
    ocp.constraints.lh_e = lh.copy()
    ocp.constraints.uh_e = uh.copy()

    ocp.constraints.lsh   = np.zeros(nh_total)
    ocp.constraints.ush   = np.zeros(nh_total)
    ocp.constraints.lsh_e = np.zeros(nh_total)
    ocp.constraints.ush_e = np.zeros(nh_total)

    p0 = np.zeros(np_dim)
    p0[5]  = 1.0
    p0[6]  = 1.0
    p0[7]  = 1.0
    p0[8]  = 1.0
    p0[9]  = 0.01
    p0[10] = 5.0
    p0[11] = 10.0

    for j in range(nh):
        base = NP_BASE + NB_STRIDE * j
        p0[base + 0] = 10.0
        p0[base + 1] = 10.0
        p0[base + 2] = 0.0
        p0[base + 3] = 0.0
        p0[base + 4] = 0.0

    OBS_START = NP_BASE + NB_STRIDE * nh
    for m in range(MAX_OBS):
        base = OBS_START + OBS_STRIDE * m
        p0[base + 0] = 10.0
        p0[base + 1] = 10.0
        p0[base + 2] = 0.0

    p0[NP_BASE + NB_STRIDE * nh + OBS_STRIDE * MAX_OBS] = 1.0
    ocp.parameter_values = p0

    ocp.solver_options.N_horizon              = N
    ocp.solver_options.Tsim                   = Ts
    ocp.solver_options.tf                     = Ts * N

    ocp.solver_options.qp_solver             = solver_opts_override.get(
        "qp_solver", "PARTIAL_CONDENSING_HPIPM"
    )
    ocp.solver_options.nlp_solver_type       = "SQP_RTI"
    ocp.solver_options.hessian_approx        = solver_opts_override.get(
        "hessian_approx", "GAUSS_NEWTON"
    )
    ocp.solver_options.levenberg_marquardt   = float(solver_opts_override.get(
        "levenberg_marquardt", 0.1
    ))
    ocp.solver_options.integrator_type       = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 3
    ocp.solver_options.print_level           = 0
    ocp.solver_options.nlp_solver_max_iter   = int(solver_opts_override.get(
        "nlp_solver_max_iter", 200
    ))
    ocp.solver_options.qp_solver_iter_max    = int(solver_opts_override.get(
        "qp_solver_max_iter", 50
    ))
    ocp.solver_options.tol                   = float(solver_opts_override.get(
        "tol", 1e-4
    ))

    ocp.code_export_directory = "lib"
    json_file = "lib/acados_ellipse_tracking_mpc_solver_config.json"

    print("─" * 60)
    print("Generating Acados C-code (slack + Reynolds + Obs Avoidance) …")
    AcadosOcpSolver.generate(ocp, json_file=json_file)

    print("Compiling shared library …")
    AcadosOcpSolver(ocp, json_file=json_file, generate=False, build=True)
    print("Compilation done.")

    with open(json_file, "r") as f:
        jdata = json.load(f)

    OBS_START = NP_BASE + NB_STRIDE * nh

    jdata["trajectory_coupled_dmpc"] = {
        "max_neighbours":     nh,
        "max_obstacles":      MAX_OBS,
        "d_safe":             model.D_SAFE,
        "d_safe_obs":         model.D_SAFE_OBS,
        "w_soft":             model.W_SOFT_DEFAULT,
        "w_soft_lin":         model.W_SOFT_LIN,
        "w_soft_obs":         model.W_SOFT_OBS_DEF,
        "w_soft_obs_lin":     model.W_SOFT_OBS_LIN,
        "np_total":           np_dim,
        "np_base":            NP_BASE,
        "nb_stride":          NB_STRIDE,
        "obs_stride":         OBS_STRIDE,
        "idx_q_cohesion":     10,
        "idx_q_align":        11,
        "idx_nb_start":       NP_BASE,
        "idx_obs_start":      OBS_START,
        "idx_avoid_on":       NP_BASE + NB_STRIDE * nh + OBS_STRIDE * MAX_OBS,
        "padding_position":   10.0,
        "horizon_N":          N,
        "sampling_time_Ts":   Ts,
        "hessian_approx":     ocp.solver_options.hessian_approx,
        "constraint_type":    "slack_variables",
        "nh_total":           nh_total,
        "nh_robots":          nh,
        "nh_obs":             MAX_OBS,
        "reynolds_rules": {
            "alignment":  "soft cost in stage cost - match neighbour mean heading",
            "cohesion":   "soft cost in stage cost - steer toward neighbour centroid",
            "separation": "slack variable on con_h_expr - dist²(robot,nb) >= D_SAFE²",
        },
        "obstacle_avoidance": {
            "type":        "slack variable on con_h_expr - dist²(robot,obs) >= D_OBS²",
            "max_obs":     MAX_OBS,
            "obs_stride":  OBS_STRIDE,
            "d_safe_obs":  model.D_SAFE_OBS,
            "w_soft_obs":  model.W_SOFT_OBS_DEF,
            "obs_layout":  "[px, py, active] per obstacle slot",
        },
        "description": (
            "Slack-variable TC-DMPC with Reynolds Rules and Obstacle Avoidance. "
            "Avoidance via con_h_expr >= 0 softened with Zl/zl slack penalties. "
            "Stage-varying parameters: set solver.set(k, 'p', p_k) for k=0..N. "
            "Neighbour block: p_k[NP_BASE + 5*j .. +4] = [px_j, py_j, vx_j, vy_j, active_j]. "
            "Obstacle block: p_k[OBS_START + 3*m .. +2] = [px_m, py_m, active_m]. "
            "Update slack weights at runtime: solver.cost_set(k, 'Zl', Zl_vec)."
        ),
    }

    with open(json_file, "w") as f:
        json.dump(jdata, f, indent=2)

    print(f"Metadata written to {json_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Acados solver for TC-DMPC (slack + Reynolds + Obs Avoid)"
    )
    parser.add_argument(
        "--config", type=str, default="solver_ellipse_mpc.yaml",
        help="Config YAML filename (looked up in config/ subdir)",
    )
    parser.add_argument(
        "--acados_source", type=str,
        default=os.environ.get("ACADOS_SOURCE_DIR", ""),
        help="Path to acados source directory",
    )
    args = parser.parse_args()

    controller_path = os.path.dirname(os.path.abspath(__file__))
    config_path     = os.path.join(controller_path, "config", args.config)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    N  = int(cfg["solver_creation"]["N"])
    Ts = float(cfg["solver_creation"]["Ts"])
    solver_opts = cfg.get("solver_options", {})

    col    = cfg.get("collision", {})
    max_nb = int(col.get("max_neighbours", 5))
    d_safe = float(col.get("d_safe", 0.07))
    w_soft = float(col.get("w_soft", 1e6))
    w_soft_lin = float(col.get("w_soft_lin", 0.0))

    rew   = cfg.get("reynolds", {})
    q_coh = float(rew.get("q_cohesion", 5.0))
    q_al  = float(rew.get("q_align",   10.0))

    obs_cfg        = cfg.get("obstacle_avoidance", {})
    max_obs        = int(obs_cfg.get("max_obstacles", 5))
    d_safe_obs     = float(obs_cfg.get("d_safe_obs", 0.08))
    w_soft_obs     = float(obs_cfg.get("w_soft_obs", 1e6))
    w_soft_obs_lin = float(obs_cfg.get("w_soft_obs_lin", 0.0))

    def model_factory():
        constraints = {
            **cfg["model_bounds"],
            **col,
            "w_soft_lin":    w_soft_lin,
            "max_obstacles": max_obs,
            "d_safe_obs":    d_safe_obs,
            "w_soft_obs":    w_soft_obs,
            "w_soft_obs_lin": w_soft_obs_lin,
        }
        return ellipse_robot_tracking_mpc(constraints)

    build_acados_solver(
        N=N,
        Ts=Ts,
        acados_source_path=args.acados_source,
        model_factory=model_factory,
        solver_opts_override=solver_opts,
    )

    np_total = NP_BASE + NB_STRIDE * max_nb + OBS_STRIDE * max_obs
    nh_total = max_nb + max_obs
    print()
    print("═" * 65)
    print("  Distributed Predictive Flocking (SLACK + Reynolds + Obs Avoidance)")
    print("═" * 65)
    print(f"  Horizon N            : {N}")
    print(f"  Sampling time Ts     : {Ts} s")
    print(f"  Look-ahead time      : {N * Ts:.1f} s")
    print(f"  MAX_NEIGHBOURS K     : {max_nb}")
    print(f"  MAX_OBSTACLES  M     : {max_obs}")
    print(f"  D_SAFE (robots)      : {d_safe * 100:.1f} cm  (slack constraint h >= 0)")
    print(f"  D_SAFE_OBS           : {d_safe_obs * 100:.1f} cm  (robot r + obs r)")
    print(f"  Zl (robots)          : {w_soft:.0e}  (quadratic slack weight)")
    print(f"  zl (robots)          : {w_soft_lin:.0e}  (linear/L1 slack weight)")
    print(f"  Zl (obstacles)       : {w_soft_obs:.0e}  (quadratic slack weight)")
    print(f"  zl (obstacles)       : {w_soft_obs_lin:.0e}  (linear/L1 slack weight)")
    print(f"  nh_total             : {nh_total}  ({max_nb} robot + {max_obs} obs constraints)")
    print(f"  np per stage         : {np_total}  "
          f"({NP_BASE} base + {NB_STRIDE}×{max_nb} nb + {OBS_STRIDE}×{max_obs} obs)")
    print(f"  Hessian              : {solver_opts.get('hessian_approx', 'GAUSS_NEWTON')}")
    print(f"  Constraint type      : SLACK VARIABLES (con_h_expr + Zl/zl)")
    print(f"  Reynolds q_cohesion  : {q_coh}")
    print(f"  Reynolds q_align     : {q_al}")
    print("═" * 65)