"""Compiles the solver on native Windows, using the MinGW toolchain."""

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


def build_acados_solver(
    N:                   int,
    Ts:                  float,
    acados_source_path:  str,
    model_factory,
    solver_opts_override: dict = None,
):
    """
    Generate C-code and compile the Acados solver for CBF-MPC.

    Stage constraints: h_j(x̂_{k+1}) - (1-α)·h_j(x_k) ≥ 0  (discrete CBF)
    Terminal constraints: h_j(x_N) ≥ 0  (simple barrier)
    All softened with nsh / nsh_e slack variables.
    """
    if solver_opts_override is None:
        solver_opts_override = {}

    ocp = AcadosOcp()

    model, constraint = model_factory()

    K       = model.MAX_NEIGHBOURS
    M       = model.MAX_OBS
    nx      = model.x.size()[0]
    nu      = model.u.size()[0]
    np_dim  = model.p.size()[0]
    nh      = constraint.nh_total
    nh_e    = K + M

    model_ac              = AcadosModel()
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

    ocp.dims.nh   = nh
    ocp.dims.nh_e = nh_e
    ocp.dims.nsh  = nh
    ocp.dims.nsh_e = nh_e

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model.cost_expr_ext_cost   = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    Zl_rob = model.W_SOFT_DEFAULT
    zl_rob = model.W_SOFT_LIN
    Zl_obs = model.W_SOFT_OBS_DEF
    zl_obs = model.W_SOFT_OBS_LIN

    Zl_vec = np.zeros(nh)
    Zl_vec[:K]              = 8.0 * Zl_rob
    Zl_vec[K:2 * K]         = Zl_rob
    Zl_vec[2 * K:2 * K + M] = 8.0 * Zl_obs
    Zl_vec[2 * K + M:]      = Zl_obs

    zl_vec = np.zeros(nh)
    zl_vec[:K]              = 8.0 * zl_rob
    zl_vec[K:2 * K]         = zl_rob
    zl_vec[2 * K:2 * K + M] = 8.0 * zl_obs
    zl_vec[2 * K + M:]      = zl_obs

    ocp.cost.Zl = Zl_vec
    ocp.cost.Zu = np.zeros(nh)
    ocp.cost.zl = zl_vec
    ocp.cost.zu = np.zeros(nh)

    Zl_e = np.zeros(nh_e)
    Zl_e[:K] = 8.0 * Zl_rob
    Zl_e[K:] = 8.0 * Zl_obs
    zl_e = np.zeros(nh_e)
    zl_e[:K] = 8.0 * zl_rob
    zl_e[K:] = 8.0 * zl_obs
    ocp.cost.Zl_e = Zl_e
    ocp.cost.Zu_e = np.zeros(nh_e)
    ocp.cost.zl_e = zl_e
    ocp.cost.zu_e = np.zeros(nh_e)

    ocp.constraints.lh   =  0.0 * np.ones(nh)
    ocp.constraints.uh   =  1e9 * np.ones(nh)

    ocp.constraints.lh_e =  0.0 * np.ones(nh_e)
    ocp.constraints.uh_e =  1e9 * np.ones(nh_e)

    ocp.constraints.idxsh   = np.arange(nh)
    ocp.constraints.idxsh_e = np.arange(nh_e)

    ocp.constraints.lsh   = np.zeros(nh)
    ocp.constraints.ush   = 1e9 * np.ones(nh)
    ocp.constraints.lsh_e = np.zeros(nh_e)
    ocp.constraints.ush_e = 1e9 * np.ones(nh_e)

    ocp.constraints.lbx   = np.array([model.px_min, model.py_min])
    ocp.constraints.ubx   = np.array([model.px_max, model.py_max])
    ocp.constraints.idxbx = np.array([0, 1])

    ocp.constraints.lbx_e   = np.array([model.px_min, model.py_min])
    ocp.constraints.ubx_e   = np.array([model.px_max, model.py_max])
    ocp.constraints.idxbx_e = np.array([0, 1])

    ocp.constraints.x0       = np.zeros(nx)
    ocp.constraints.idxbxe_0 = np.arange(nx)

    ocp.constraints.lbu   = np.array([model.v_min,  model.om_min])
    ocp.constraints.ubu   = np.array([model.v_max,  model.om_max])
    ocp.constraints.idxbu = np.arange(nu)

    p0     = np.zeros(np_dim)
    p0[5]  = 1.0
    p0[6]  = 1.0
    p0[7]  = 1.0
    p0[8]  = 1.0
    p0[9]  = 0.01
    p0[10] = 5.0
    p0[11] = 10.0
    for j in range(K):
        base = NP_BASE + NB_STRIDE * j
        p0[base + 0] = 10.0
        p0[base + 1] = 10.0
        p0[base + 2] = 0.0
        p0[base + 3] = 0.0
        p0[base + 4] = 0.0
    obs_base = NP_BASE + NB_STRIDE * K
    for m in range(M):
        base = obs_base + OBS_STRIDE * m
        p0[base + 0] = 10.0
        p0[base + 1] = 10.0
        p0[base + 2] = 0.0
    p0[NP_BASE + NB_STRIDE * K + OBS_STRIDE * M] = 1.0
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
        "nlp_solver_max_iter", 1
    ))
    ocp.solver_options.tol                   = float(solver_opts_override.get(
        "tol", 1e-4
    ))
    ocp.solver_options.qp_solver_iter_max    = int(solver_opts_override.get(
        "qp_solver_max_iter", 10
    ))

    ocp.code_export_directory = "lib"
    json_file = "lib/acados_ellipse_tracking_mpc_solver_config.json"

    print("-" * 60)
    cmake_builder = None
    if os.name == "nt":
        from acados_template.builders import ocp_get_default_cmake_builder

        cmake_builder = ocp_get_default_cmake_builder()
        cmake_builder.generator = "MinGW Makefiles"
        cmake_builder.additional_cmake_options = "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

    print("Generating Acados C-code (CBF-MPC) ...")
    AcadosOcpSolver.generate(ocp, json_file=json_file, cmake_builder=cmake_builder)

    print("Compiling shared library ...")
    AcadosOcpSolver(
        ocp, json_file=json_file, generate=False, build=True, cmake_builder=cmake_builder
    )
    print("Compilation done.")

    with open(json_file, "r") as f:
        jdata = json.load(f)

    jdata["trajectory_coupled_dmpc"] = {
        "max_neighbours":     K,
        "max_obstacles":      M,
        "d_safe":             model.D_SAFE,
        "d_safe_obs":         model.D_SAFE_OBS,
        "cbf_alpha":          model.CBF_ALPHA,
        "cbf_alpha_obs":      model.CBF_ALPHA_OBS,
        "Ts":                 model.TS,
        "nh_total":           nh,
        "nh_robots":          2 * K,
        "nh_obs":             2 * M,
        "np_total":           np_dim,
        "np_base":            NP_BASE,
        "nb_stride":          NB_STRIDE,
        "obs_stride":         OBS_STRIDE,
        "idx_nb_start":       NP_BASE,
        "idx_obs_start":      NP_BASE + NB_STRIDE * K,
        "idx_avoid_on":       NP_BASE + NB_STRIDE * K + OBS_STRIDE * M,
        "padding_position":   10.0,
        "horizon_N":          N,
        "sampling_time_Ts":   Ts,
        "hessian_approx":     ocp.solver_options.hessian_approx,
        "constraint_type":    "hybrid: slacked BF + slacked Dt-CBF at stages",
        "slack_Zl_rob":       float(Zl_rob),
        "slack_zl_rob":       float(zl_rob),
        "slack_Zl_obs":       float(Zl_obs),
        "slack_zl_obs":       float(zl_obs),
        "description": (
            "Hybrid slacked BF+Dt-CBF: stage h_j>=0 AND h_j(x_{k+1})>=(1-alpha)h_j(x_k), "
            "terminal slacked h_j>=0. Distance slack penalized 8× vs Dt-CBF slack."
        ),
    }

    with open(json_file, "w") as f:
        json.dump(jdata, f, indent=2)

    print(f"Metadata written to {json_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Acados solver for CBF-MPC with Reynolds flocking"
    )
    parser.add_argument(
        "--config", type=str, default="solver_ellipse_mpc.yaml",
        help="Config YAML filename (in config/ subdirectory)",
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
    obs    = cfg.get("obstacle_avoidance", {})
    rew    = cfg.get("reynolds", {})

    max_nb      = int(col.get("max_neighbours", 5))
    d_safe      = float(col.get("d_safe",       0.07))
    cbf_alpha   = float(col.get("cbf_alpha", col.get("alpha_cbf", 0.5)))
    w_soft      = float(col.get("w_soft",       1e4))
    w_soft_lin  = float(col.get("w_soft_lin",   0.0))

    max_obs         = int(obs.get("max_obstacles",   5))
    d_safe_obs      = float(obs.get("d_safe_obs",   0.10))
    cbf_alpha_obs   = float(obs.get("cbf_alpha_obs",
                        obs.get("alpha_cbf_obs", cbf_alpha)))
    w_soft_obs      = float(obs.get("w_soft_obs",   1e4))
    w_soft_obs_lin  = float(obs.get("w_soft_obs_lin",0.0))

    q_coh  = float(rew.get("q_cohesion", 5.0))
    q_al   = float(rew.get("q_align",   10.0))

    def model_factory():
        constraints = {
            **cfg["model_bounds"],
            **col,
            **obs,
            "Ts": Ts,
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
    print()
    print("═" * 65)
    print("  Distributed Predictive Flocking (CBF-MPC + Reynolds + Obs Avoidance)")
    print("═" * 65)
    print(f"  Horizon N            : {N}")
    print(f"  Sampling time Ts     : {Ts} s")
    print(f"  Look-ahead time      : {N * Ts:.1f} s")
    print(f"  MAX_NEIGHBOURS K     : {max_nb}")
    print(f"  MAX_OBSTACLES  M     : {max_obs}")
    print(f"  D_SAFE (robots)      : {d_safe * 100:.1f} cm")
    print(f"  D_SAFE_OBS           : {d_safe_obs * 100:.1f} cm")
    print(f"  cbf_alpha (robots)   : {cbf_alpha}  (discrete CBF decay)")
    print(f"  cbf_alpha (obstacles): {cbf_alpha_obs}")
    print(f"  Zl (robots)          : {w_soft:.0e}  (quadratic slack)")
    print(f"  zl (robots)          : {w_soft_lin:.0e}  (linear slack)")
    print(f"  Zl (obstacles)       : {w_soft_obs:.0e}")
    print(f"  zl (obstacles)       : {w_soft_obs_lin:.0e}")
    print(f"  nh_total             : {2 * (max_nb + max_obs)}  (dist+CBF per robot/obs)")
    print(f"  np per stage         : {np_total}  ({NP_BASE} base + {NB_STRIDE}×{max_nb} nb + {OBS_STRIDE}×{max_obs} obs)")
    print(f"  Hessian              : {solver_opts.get('hessian_approx', 'GAUSS_NEWTON')}")
    print(f"  Constraint type      : hybrid slacked BF + slacked Dt-CBF at stages")
    print(f"  Reynolds q_cohesion  : {q_coh}")
    print(f"  Reynolds q_align     : {q_al}")
    print("═" * 65)