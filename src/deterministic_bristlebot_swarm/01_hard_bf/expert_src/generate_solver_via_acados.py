"""Compiles the optimal control problem into an acados solver."""

import argparse
import json
import os

import numpy as np
import yaml

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from symbolic_model_defination import ellipse_robot_tracking_mpc


NB_STRIDE = 5
NP_BASE   = 12



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
    max_neighbours: int,
    obstacle_manager = None,
    robot_pos = None,
    stage: int = 0,
    dt: float = 0.1,
    max_obstacles: int = 5,
    padding_position: float = 10.0,
) -> np.ndarray:
    """
    Build the parameter vector p_stage_k ∈ R^(12 + 5*K + 3*M) for stage k.

    Parameters
    ----------
    neighbour_states_at_stage_k : ndarray, shape (n_active, 4)
        Predicted [px, py, vx, vy] of each active neighbour at MPC stage k.
        Rows beyond n_active are padded (active=0.0) so Reynolds and
        collision constraints are trivially satisfied for inactive slots.
    q_cohesion, q_align : float
        Reynolds weight scalars (can be set to 0 to disable Reynolds rules).
    padding_position : float
        Value used for inactive position slots.

    Returns
    -------
    p : ndarray, shape (12 + 5*max_neighbours + 3*max_obstacles,)
    """
    np_total = NP_BASE + NB_STRIDE * max_neighbours + 3 * max_obstacles
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

    n_active = min(len(neighbour_states_at_stage_k), max_neighbours)
    for j in range(max_neighbours):
        base = NP_BASE + NB_STRIDE * j
        if j < n_active:
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

    obs_start = NP_BASE + NB_STRIDE * max_neighbours
    if obstacle_manager is not None and robot_pos is not None:
        obs_block = obstacle_manager.build_obs_param_block(
            robot_pos=robot_pos,
            stage=stage,
            dt=dt,
            padding_pos=padding_position,
        )
        p[obs_start: obs_start + len(obs_block)] = obs_block
    else:
        for m in range(max_obstacles):
            base = obs_start + 3 * m
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
    """
    Generate C-code and compile the Acados solver for trajectory-coupled DMPC
    with Reynolds Rules collective behavior (alignment + cohesion soft costs).

    Parameters
    ----------
    N               : MPC horizon (number of shooting intervals)
    Ts              : sampling time [s]
    acados_source_path : path to acados source (for ACADOS_SOURCE_DIR)
    model_factory   : callable () -> (model, constraint)
    solver_opts_override : dict of optional overrides from YAML [solver_options]
    """
    if solver_opts_override is None:
        solver_opts_override = {}

    ocp = AcadosOcp()

    model, constraint = model_factory()

    nh      = model.MAX_NEIGHBOURS + model.MAX_OBS
    nx      = model.x.size()[0]
    nu      = model.u.size()[0]
    np_dim  = model.p.size()[0]

    model_ac = AcadosModel()
    model_ac.f_impl_expr  = model.f_impl_expr
    model_ac.f_expl_expr  = model.f_expl_expr
    model_ac.x            = model.x
    model_ac.xdot         = model.xdot
    model_ac.u            = model.u
    model_ac.z            = model.z
    model_ac.p            = model.p
    model_ac.name         = model.name

    model_ac.con_h_expr   = model.con_h_expr
    model_ac.con_h_expr_e = model.con_h_expr_e

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

    ocp.dims.nh     = nh
    ocp.dims.nh_e   = nh

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model.cost_expr_ext_cost   = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

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

    ocp.constraints.lh   = np.zeros(nh)
    ocp.constraints.uh   = np.full(nh, 1e9)
    ocp.constraints.lh_e = np.zeros(nh)
    ocp.constraints.uh_e = np.full(nh, 1e9)

    p0 = np.zeros(np_dim)
    p0[5]  = 1.0
    p0[6]  = 1.0
    p0[7]  = 1.0
    p0[8]  = 1.0
    p0[9]  = 0.01
    p0[10] = 5.0
    p0[11] = 10.0
    for j in range(model.MAX_NEIGHBOURS):
        base = NP_BASE + NB_STRIDE * j
        p0[base + 0] = 10.0
        p0[base + 1] = 10.0
        p0[base + 2] = 0.0
        p0[base + 3] = 0.0
        p0[base + 4] = 0.0
    obs_start = NP_BASE + NB_STRIDE * model.MAX_NEIGHBOURS
    for m in range(model.MAX_OBS):
        base = obs_start + 3 * m
        p0[base + 0] = 10.0
        p0[base + 1] = 10.0
        p0[base + 2] = 0.0
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
    ocp.solver_options.tol                   = float(solver_opts_override.get(
        "tol", 1e-4
    ))
    ocp.solver_options.qp_solver_iter_max    = int(solver_opts_override.get(
        "qp_solver_max_iter", 10
    ))

    ocp.code_export_directory = "lib"
    json_file = "lib/acados_ellipse_tracking_mpc_solver_config.json"

    print("─" * 60)
    print("Generating Acados C-code (Reynolds Rules enabled) …")
    AcadosOcpSolver.generate(ocp, json_file=json_file)

    print("Compiling shared library …")
    AcadosOcpSolver(ocp, json_file=json_file, generate=False, build=True)
    print("Compilation done.")

    with open(json_file, "r") as f:
        jdata = json.load(f)

    obs_start = NP_BASE + NB_STRIDE * model.MAX_NEIGHBOURS
    jdata["trajectory_coupled_dmpc"] = {
        "max_neighbours":     model.MAX_NEIGHBOURS,
        "max_obstacles":      model.MAX_OBS,
        "d_safe":             model.D_SAFE,
        "d_safe_obs":         model.D_SAFE_OBS,
        "np_total":           np_dim,
        "np_base":            NP_BASE,
        "nb_stride":          NB_STRIDE,
        "obs_stride":         3,
        "idx_tracking_end":   10,
        "idx_q_cohesion":     10,
        "idx_q_align":        11,
        "idx_nb_start":       NP_BASE,
        "idx_obs_start":      obs_start,
        "padding_position":   10.0,
        "horizon_N":          N,
        "sampling_time_Ts":   Ts,
        "hessian_approx":     ocp.solver_options.hessian_approx,
        "reynolds_rules": {
            "alignment":  "soft cost in stage cost - match neighbour mean heading",
            "cohesion":   "soft cost in stage cost - steer toward neighbour centroid",
            "separation": "handled by hard collision constraint h_j >= 0 (unchanged)",
            "obstacle_avoidance": "handled by hard collision constraint h_obs_m >= 0 (NEW)",
        },
        "description": (
            "Stage-varying parameters: set solver.set(k,'p', p_k) "
            "for k=0..N where p_k block [NP_BASE + NB_STRIDE*j .. +4] = "
            "[px_j, py_j, vx_j, vy_j, active_j] of neighbour j at stage k, and "
            "[idx_obs_start + 3*m .. +2] = [px_m, py_m, active_m] of obstacle m at stage k."
        ),
    }

    with open(json_file, "w") as f:
        json.dump(jdata, f, indent=2)

    print(f"Metadata written to {json_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Acados solver for trajectory-coupled DMPC with Reynolds Rules"
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

    col     = cfg.get("collision", {})
    max_nb  = int(col.get("max_neighbours", 5))
    d_safe  = float(col.get("d_safe", 0.07))

    obs_av  = cfg.get("obstacle_avoidance", {})
    max_obs = int(obs_av.get("max_obstacles", 5))
    d_safe_obs = float(obs_av.get("d_safe_obs", 0.10))

    rew     = cfg.get("reynolds", {})
    q_coh   = float(rew.get("q_cohesion", 5.0))
    q_al    = float(rew.get("q_align",   10.0))

    def model_factory():
        constraints = {**cfg["model_bounds"], **col, **obs_av}
        return ellipse_robot_tracking_mpc(constraints)

    build_acados_solver(
        N=N,
        Ts=Ts,
        acados_source_path=args.acados_source,
        model_factory=model_factory,
        solver_opts_override=solver_opts,
    )

    np_total = NP_BASE + NB_STRIDE * max_nb + 3 * max_obs
    print()
    print("═" * 60)
    print("  Distributed Predictive Flocking + Reynolds Rules - solver generated")
    print("═" * 60)
    print(f"  Horizon N          : {N}")
    print(f"  Sampling time Ts   : {Ts} s")
    print(f"  Look-ahead time    : {N * Ts:.1f} s")
    print(f"  MAX_NEIGHBOURS K   : {max_nb}")
    print(f"  MAX_OBSTACLES M    : {max_obs}")
    print(f"  D_SAFE             : {d_safe * 100:.0f} cm  (hard constraint)")
    print(f"  D_SAFE_OBS         : {d_safe_obs * 100:.0f} cm  (hard constraint)")
    print(f"  np per stage       : {np_total}  (12 base + {NB_STRIDE}×{max_nb} neighbours + 3×{max_obs} obstacles)")
    print(f"  Hessian            : {solver_opts.get('hessian_approx', 'GAUSS_NEWTON')}")
    print(f"  nh (path)          : {max_nb + max_obs}")
    print(f"  nh_e (terminal)    : {max_nb + max_obs}")
    print(f"  Reynolds q_cohesion: {q_coh}  (soft - steers toward centroid)")
    print(f"  Reynolds q_align   : {q_al}   (soft - aligns heading)")
    print(f"  Separation         : HARD constraint (unchanged)")
    print(f"  Obstacle Avoidance : HARD constraint (NEW)")
    print("═" * 60)