"""Simulates the swarm passing through a doorway."""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(_ROOT)

import sim_engine.headless_sim_engine as H
from sim_engine.swarm_dynamics import SwarmBehavior

BEHAVIORS = {
    "phototaxis":  SwarmBehavior.PHOTOTAXIS,
    "orbital":     SwarmBehavior.PHOTOTAXIS_ORBITAL,
    "poc":         SwarmBehavior.PHOTOTAXIS_ORBITAL_CONTRACTING,
    "no_reynolds": SwarmBehavior.PHOTOTAXIS_NO_REYNOLDS,
    "no_light":    SwarmBehavior.NO_LIGHT,
    "no_stimulus": SwarmBehavior.NO_LIGHT,
}
ENVIRONMENTS = {"free": 0, "sparse": 2, "cluttered": 12}

DOORS = {
    "bottom": dict(pos=[0.0, -0.36], theta=np.pi / 2, goal=[0.0, -0.15],
                   wall="bottom", axis=1, corridor=-0.22),
    "corner": dict(pos=[-0.36, -0.28], theta=0.0, goal=[-0.15, -0.28],
                   wall="left", axis=0, corridor=-0.22),
    "corner_low": dict(pos=[-0.36, -0.38], theta=0.0, goal=[-0.15, -0.38],
                       wall="left", axis=0, corridor=-0.22),
}
DOOR_HALF = 0.055
DOOR_CLEAR = 0.075
ENTRY_STEPS = 20

OBSTACLE_SEEDS = {
    ("bottom", 0): 0, ("bottom", 2): 1, ("bottom", 12): 197,
    ("corner", 0): 0, ("corner", 2): 1, ("corner", 12): 197,
    ("corner_low", 0): 0, ("corner_low", 2): 1, ("corner_low", 12): 197,
}
HEATMAP_RES = 20


class DoorSpawnSim(H.HeadlessSim):
    """HeadlessSim with incremental door spawning up to nmax robots."""

    def __init__(self, behavior_mode, nmax=137, spawn_interval=1.0,
                 door="bottom", force_spawn=False, **kw):
        super().__init__(n_robots=nmax, behavior_mode=behavior_mode,
                         enable_collision=True, **kw)
        self.nmax = nmax
        self.spawn_interval = float(spawn_interval)
        self.force_spawn = bool(force_spawn)
        self.door_name = door
        d = DOORS[door]
        self.door_pos = np.array(d["pos"], dtype=float)
        self.door_theta = float(d["theta"])
        self.door_goal = np.array(d["goal"], dtype=float)
        self.door_wall = d["wall"]
        self.door_axis = int(d["axis"])
        self.door_corridor = float(d["corridor"])
        self.active_count = 0
        self.last_spawn = -1e9
        self.spawn_step = np.full(nmax, -1, dtype=np.int64)
        self._step_idx = 0
        for i, rb in enumerate(self.robots):
            rb.pos = np.array([0.0, -1.0 - 0.30 * i])
            rb.theta = np.pi / 2
            rb.velocity = np.zeros(2)
            rb.ang_velocity = 0.0
            self.histories[i] = [rb.pos.copy()]
            self._predicted_trajs[i] = None
        self._rebuild_collision_handler()

    def _rebuild_collision_handler(self):
        if self.enable_collision:
            self.collision = H.CollisionHandler(
                self.robots[:self.active_count],
                steer_decay=H.COLLISION_STEER_DECAY,
                flash_frames=H.COLLISION_FLASH_FRAMES)

    def _try_spawn(self):
        na = self.active_count
        if na >= self.nmax:
            return
        if (self.current_time - self.last_spawn) < self.spawn_interval - 1e-9:
            return
        if not self.force_spawn:
            for j in range(na):
                if np.linalg.norm(self.robots[j].pos - self.door_pos) < DOOR_CLEAR:
                    return
        rb = self.robots[na]
        rb.pos = self.door_pos.copy()
        rb.theta = self.door_theta
        rb.velocity = np.zeros(2)
        rb.ang_velocity = 0.0
        self.histories[na] = [rb.pos.copy()]
        self._predicted_trajs[na] = None
        self.spawn_step[na] = self._step_idx
        self.active_count = na + 1
        self.last_spawn = self.current_time
        self._rebuild_collision_handler()

    def _goal_from_behavior_with_collision(self, robot_idx):
        age = self._step_idx - int(self.spawn_step[robot_idx])
        if (0 <= age < ENTRY_STEPS
                and self.robots[robot_idx].pos[self.door_axis] < self.door_corridor):
            v = H.V_TARGET
            th = self.door_theta
            return (self.door_goal.copy(), th,
                    np.array([v * np.cos(th), v * np.sin(th)]))
        return super()._goal_from_behavior_with_collision(robot_idx)

    def step(self):
        from scipy.spatial import KDTree
        import expert_src.controller as ctrl_module

        self._try_spawn()
        na = self.active_count

        if self.field is not None and self.behavior_mode != SwarmBehavior.ORBIT:
            self.field.step(H.DT)
        if self.obstacle_manager is not None:
            self.obstacle_manager.step(H.DT)

        self.kdtree = KDTree(np.array([r.pos for r in self.robots]))

        all_nb_trajs = [self._get_neighbour_trajs(i) for i in range(na)]

        original_max_cons = ctrl_module.MAX_CONSENSUS_ITER
        ctrl_module.MAX_CONSENSUS_ITER = self.max_consensus

        controls = []
        for i in range(na):
            robot, ctrl = self.robots[i], self.controllers[i]
            goal_pos, ref_theta, ref_vel = self._goal_from_behavior_with_collision(i)
            u = ctrl.compute_control(
                robot, goal_pos, ref_theta=ref_theta, ref_vel=ref_vel,
                ref_omega=0.0, neighbour_trajs=all_nb_trajs[i],
                obstacle_manager=self.obstacle_manager, robot_name=f"R{i}")
            controls.append(u)
            self.solver_statuses[i] = ctrl.last_solver_status
            if ctrl.last_solver_status != 0:
                self.failure_counts[i] += 1

        ctrl_module.MAX_CONSENSUS_ITER = original_max_cons

        for i in range(na):
            self._predicted_trajs[i] = self.controllers[i].get_predicted_trajectory()
        for i in range(na):
            self.controllers[i].apply_control(self.robots[i], controls[i])
            self.histories[i].append(self.robots[i].pos.copy())

        self._resolve_collisions()

        if na >= 2:
            positions = np.array([r.pos for r in self.robots[:na]])
            diffs = positions[:, None, :] - positions[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists, np.inf)
            self.min_dist_log.append(float(np.min(dists)))
        else:
            self.min_dist_log.append(np.inf)

        self.current_time += H.DT
        self.time_history.append(self.current_time)
        self._step_idx += 1


def apply_v_min(sim, v_min):
    """Raise the solver's lower speed bound at runtime (no recompile).

    Bound VALUES are runtime data in acados; only the OCP structure is baked.
    Also raises the controller's own clip/fallback floor.
    """
    for ctrl in sim.controllers:
        ctrl.min_linear_vel = float(v_min)
        om_min = float(getattr(ctrl, "min_angular_vel", -3.463))
        lbu = np.array([float(v_min), om_min])
        for k in range(ctrl.N):
            ctrl.solver.constraints_set(k, "lbu", lbu)


def run_case(behavior_key, env_key, nmax, spawn_interval, duration, out_path,
             v_min=None, door="bottom", force_spawn=False):
    count = ENVIRONMENTS[env_key]
    H.ENABLE_OBSTACLES = count > 0
    H.OBSTACLE_DEFS = H.build_obstacle_defs(count, seed=OBSTACLE_SEEDS[(door, count)])

    sim = DoorSpawnSim(BEHAVIORS[behavior_key], nmax=nmax,
                       spawn_interval=spawn_interval, door=door,
                       force_spawn=force_spawn, enable_obstacles=count > 0)
    if v_min is not None and v_min > 0.0:
        apply_v_min(sim, v_min)

    n_steps = int(round(duration / H.DT))
    pos = np.zeros((n_steps, nmax, 2), dtype=np.float32)
    theta = np.zeros((n_steps, nmax), dtype=np.float32)
    active = np.zeros(n_steps, dtype=np.int32)
    n_infeas = np.zeros(n_steps, dtype=np.int32)
    colliding = np.zeros((n_steps, nmax), dtype=bool)
    total_coll = np.zeros(n_steps, dtype=np.int32)
    min_dist = np.zeros(n_steps, dtype=np.float32)
    heat = None
    src_pos = np.full((n_steps, 2), np.nan, dtype=np.float32)

    if sim.field is not None:
        heat = np.zeros((n_steps, HEATMAP_RES, HEATMAP_RES), dtype=np.float16)
        _, extent = sim.field.heatmap(HEATMAP_RES)
        src_amp = float(getattr(sim.field.sources[0], "amplitude", 1.0))
    else:
        extent = (-H.WALL_HALF, H.WALL_HALF, -H.WALL_HALF, H.WALL_HALF)
        src_amp = 0.0

    t0 = time.time()
    for s in range(n_steps):
        sim.step()
        na = sim.active_count
        pos[s] = [r.pos for r in sim.robots]
        theta[s] = [r.theta for r in sim.robots]
        active[s] = na
        n_infeas[s] = sum(1 for i in range(na) if sim.solver_statuses[i] != 0)
        if sim.collision is not None:
            for i in sim.collision.active_set():
                colliding[s, i] = True
        total_coll[s] = sim.total_collisions
        min_dist[s] = sim.min_dist_log[-1]
        if sim.field is not None:
            grid, _ = sim.field.heatmap(HEATMAP_RES)
            heat[s] = grid.astype(np.float16)
            src_pos[s] = sim.field.brightest_source_pos()
        if s % 100 == 0:
            print(f"  [{behavior_key}/{env_key}] step {s}/{n_steps} "
                  f"active={na} ({time.time()-t0:.0f}s)", flush=True)

    obs = []
    if sim.obstacle_manager is not None:
        obs = [[o.pos[0], o.pos[1], o.radius]
               for o in sim.obstacle_manager.active_obstacles]

    np.savez_compressed(
        out_path, pos=pos, theta=theta, active=active, n_infeas=n_infeas,
        colliding=colliding, total_coll=total_coll, min_dist=min_dist,
        heat=(heat if heat is not None else np.zeros((0,))),
        heat_extent=np.array(extent, dtype=np.float32), src_pos=src_pos,
        src_amp=src_amp, obstacles=np.array(obs, dtype=np.float32),
        spawn_step=sim.spawn_step, dt=H.DT, nmax=nmax,
        spawn_interval=spawn_interval, behavior=behavior_key, env=env_key,
        v_min=(v_min if v_min is not None else 0.0),
        door_wall=sim.door_wall, door_pos=sim.door_pos.astype(np.float32),
        door_half=np.float32(DOOR_HALF))
    print(f"  [{behavior_key}/{env_key}] saved {out_path} "
          f"({time.time()-t0:.0f}s, final N={sim.active_count})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", required=True, choices=list(BEHAVIORS))
    ap.add_argument("--env", action="append", choices=list(ENVIRONMENTS))
    ap.add_argument("--nmax", type=int, default=137)
    ap.add_argument("--spawn-interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=480.0,
                    help="total simulated seconds (spawning continues while "
                         "the doorway is clear and N < nmax)")
    ap.add_argument("--outdir", default=os.path.join(_HERE, "door_runs"))
    ap.add_argument("--v-min", type=float, default=None,
                    help="raise the solver's lower speed bound [m/s] at "
                         "runtime (default: keep the compiled bound, 0.0)")
    ap.add_argument("--door", choices=list(DOORS), default="bottom",
                    help="entry door location (default: bottom-center)")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the output npz already exists")
    ap.add_argument("--force-spawn", action="store_true",
                    help="admit a robot on every interval even when the doorway "
                         "is occupied, instead of waiting for it to clear")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    envs = args.env or list(ENVIRONMENTS)
    for env in envs:
        out = os.path.join(args.outdir, f"{args.behavior}_{env}.npz")
        if os.path.exists(out) and not args.force:
            print(f"  [{args.behavior}/{env}] exists, skipping ({out})",
                  flush=True)
            continue
        run_case(args.behavior, env, args.nmax, args.spawn_interval,
                 args.duration, out, v_min=args.v_min, door=args.door,
                 force_spawn=args.force_spawn)
    print(f"[{args.behavior}] all environments done")


if __name__ == "__main__":
    main()
