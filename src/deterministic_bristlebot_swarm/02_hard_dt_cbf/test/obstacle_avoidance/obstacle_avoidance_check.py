"""Drives the swarm past obstacles to check the avoidance constraints."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle, Rectangle
from matplotlib.lines import Line2D
_ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SIM_DIR    = os.path.join(_ROOT_DIR, "sim_engine")
_EXPERT_DIR = os.path.join(_ROOT_DIR, "expert_src")
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _EXPERT_DIR)

from sim_engine.obstacle_model import ObstacleManager, StaticObstacle, MovingObstacle
from expert_src.Ellipse2dObj import EllipticalBot2D
from expert_src.controller import CasadiAcadosEllipticalController, D_SAFE_OBS

DT          = 0.1
N_HORIZON   = 15
MAX_STEPS   = 100
ARENA_HALF  = 0.40
OBS_RADIUS  = 0.03
MPC_PRED_STAGES = 15
SAVE_GIF    = False
GIF_NAME    = "obstacle_avoidance_check_dsafe_7cm_dsafeobs_10cm_n_15.gif"

GOAL_TOL    = 0.02


def run_obstacle_avoidance_animation():

    robot_a = EllipticalBot2D(
        r=np.array([-0.25, 0.25]), theta_rad=0.0,
        vmag=0.0, w=0.0, a=0.03, b=0.015,
    )
    robot_b = EllipticalBot2D(
        r=np.array([-0.25, -0.25]), theta_rad=0.0,
        vmag=0.0, w=0.0, a=0.03, b=0.015,
    )

    goal_a = np.array([ 0.25,  0.25])
    goal_b = np.array([ 0.25, -0.25])

    ctrl_a = CasadiAcadosEllipticalController(dt=DT, N=N_HORIZON)
    ctrl_b = CasadiAcadosEllipticalController(dt=DT, N=N_HORIZON)
    

    manager = ObstacleManager(dt=DT, arena_half=ARENA_HALF, max_obs=5, obs_range=0.60)

    moving_obs = MovingObstacle(
        pos=[ 0.25, -0.25],
        vel=[-0.05, 0.00],
        radius=OBS_RADIUS,
        arena_half=ARENA_HALF,
        bounce=True,
    )
    static_obs = StaticObstacle(
        pos=[0.00, 0.25],
        radius=OBS_RADIUS,
    )
    manager.add(moving_obs)
    manager.add(static_obs)

    history_a = [robot_a.pos.copy()]
    history_b = [robot_b.pos.copy()]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5, color='gray')

    arena_rect = Rectangle(
        (-ARENA_HALF, -ARENA_HALF), 2 * ARENA_HALF, 2 * ARENA_HALF,
        linewidth=1.2, edgecolor='black', facecolor='none', linestyle='--',
    )
    ax.add_patch(arena_rect)

    ax.plot(*goal_a, marker='x', ms=12, mew=2.5, color='#00d4ff', zorder=10)
    ax.plot(*goal_b, marker='x', ms=12, mew=2.5, color='#f9a825', zorder=10)
    ax.text(goal_a[0] + 0.02, goal_a[1] - 0.025, 'Goal A',
            color='#00d4ff', fontsize=7)
    ax.text(goal_b[0] + 0.02, goal_b[1] - 0.025, 'Goal B',
            color='#f9a825', fontsize=7)

    trail_a, = ax.plot([], [], '-', lw=1.6, color='#00d4ff', alpha=0.55)
    trail_b, = ax.plot([], [], '-', lw=1.6, color='#f9a825', alpha=0.55)

    pred_a, = ax.plot([], [], ':', lw=1.2, color='#00d4ff', alpha=0.45,
                      label='NMPC pred A')
    pred_b, = ax.plot([], [], ':', lw=1.2, color='#f9a825', alpha=0.45,
                      label='NMPC pred B')

    patch_a = Ellipse(robot_a.pos, 0.06, 0.03, angle=0,
                      facecolor='#00d4ff', alpha=0.85, zorder=6, label='Robot A')
    patch_b = Ellipse(robot_b.pos, 0.06, 0.03, angle=0,
                      facecolor='#f9a825', alpha=0.85, zorder=6, label='Robot B')
    ax.add_patch(patch_a)
    ax.add_patch(patch_b)

    patch_mov = Circle(moving_obs.pos.copy(), OBS_RADIUS,
                       color='#ff4444', alpha=0.80, zorder=5, label='Moving Obs')
    ring_mov  = Circle(moving_obs.pos.copy(), OBS_RADIUS + D_SAFE_OBS / 2,
                       fill=False, edgecolor='#ff4444', linestyle='--', lw=1.2,
                       alpha=0.50, zorder=4)
    ax.add_patch(patch_mov)
    ax.add_patch(ring_mov)

    ghost_dots = [
        ax.plot([], [], 'o', ms=4, color='#ff8888', alpha=0.0)[0]
        for _ in range(MPC_PRED_STAGES + 1)
    ]
    ghost_line, = ax.plot([], [], '-', lw=0.8, color='#ff4444', alpha=0.30)

    patch_sta = Circle(static_obs.pos.copy(), OBS_RADIUS,
                       color='#cc44ff', alpha=0.80, zorder=5, label='Static Obs')
    ring_sta  = Circle(static_obs.pos.copy(), OBS_RADIUS + D_SAFE_OBS / 2,
                       fill=False, edgecolor='#cc44ff', linestyle='--', lw=1.2,
                       alpha=0.50, zorder=4)
    ax.add_patch(patch_sta)
    ax.add_patch(ring_sta)

    title = ax.set_title(
        "TC-DMPC Obstacle Avoidance  |  Step: 000",
        color='black', fontsize=11, pad=8,
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00d4ff',
               ms=8, label='Robot A'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f9a825',
               ms=8, label='Robot B'),
        Line2D([0], [0], linestyle=':', color='#00d4ff',
               lw=1.5, label='NMPC pred A'),
        Line2D([0], [0], linestyle=':', color='#f9a825',
               lw=1.5, label='NMPC pred B'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444',
               ms=8, label='Moving Obstacle'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#cc44ff',
               ms=8, label='Static Obstacle'),
        Line2D([0], [0], linestyle='--', color='#ff4444',
               lw=1.2, alpha=0.7, label=f'Safety ring (D_safe_obs)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=7,
              facecolor='white', labelcolor='black', framealpha=0.85)
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')


    def init():
        trail_a.set_data([], [])
        trail_b.set_data([], [])
        pred_a.set_data([], [])
        pred_b.set_data([], [])
        ghost_line.set_data([], [])
        for dot in ghost_dots:
            dot.set_data([], [])
        return (trail_a, trail_b, pred_a, pred_b,
                patch_a, patch_b, patch_mov, ring_mov,
                ghost_line, title)

    def animate(i):
        done_a = np.linalg.norm(robot_a.pos - goal_a) < GOAL_TOL
        done_b = np.linalg.norm(robot_b.pos - goal_b) < GOAL_TOL

        if done_a and done_b:
            title.set_text(f"Both Goals Reached!  Step: {i:03d}")
            return (trail_a, trail_b, pred_a, pred_b,
                    patch_a, patch_b, patch_mov, ring_mov,
                    *ghost_dots, ghost_line, title)

        manager.step(DT)

        if not done_a:
            traj_b_for_a = (
                np.tile(np.concatenate([robot_b.pos, robot_b.velocity]), (N_HORIZON + 1, 1))
            )
            u_a = ctrl_a.compute_control(
                robot_a,
                goal=goal_a,
                neighbour_trajs=[traj_b_for_a],
                obstacle_manager=manager,
                robot_name="RobotA",
            )
            ctrl_a.apply_control(robot_a, u_a)
            history_a.append(robot_a.pos.copy())

        if not done_b:
            traj_a_for_b = (
                np.tile(np.concatenate([robot_a.pos, robot_a.velocity]), (N_HORIZON + 1, 1))
            )
            u_b = ctrl_b.compute_control(
                robot_b,
                goal=goal_b,
                neighbour_trajs=[traj_a_for_b],
                obstacle_manager=manager,
                robot_name="RobotB",
            )
            ctrl_b.apply_control(robot_b, u_b)
            history_b.append(robot_b.pos.copy())

        patch_a.set_center(robot_a.pos)
        patch_a.angle = np.rad2deg(robot_a.theta)
        patch_a.set_facecolor('black' if ctrl_a.last_solver_status != 0 else '#00d4ff')

        patch_b.set_center(robot_b.pos)
        patch_b.angle = np.rad2deg(robot_b.theta)
        patch_b.set_facecolor('black' if ctrl_b.last_solver_status != 0 else '#f9a825')

        ha = np.array(history_a)
        hb = np.array(history_b)
        trail_a.set_data(ha[:, 0], ha[:, 1])
        trail_b.set_data(hb[:, 0], hb[:, 1])

        traj_pred_a = ctrl_a.get_predicted_trajectory()
        if traj_pred_a is not None:
            pred_a.set_data(traj_pred_a[:, 0], traj_pred_a[:, 1])

        traj_pred_b = ctrl_b.get_predicted_trajectory()
        if traj_pred_b is not None:
            pred_b.set_data(traj_pred_b[:, 0], traj_pred_b[:, 1])

        patch_mov.set_center(moving_obs.pos)
        ring_mov.set_center(moving_obs.pos)

        preds = [manager.predict_pos_at_stage(moving_obs, k, DT)
                 for k in range(MPC_PRED_STAGES + 1)]
        gx = [p[0] for p in preds]
        gy = [p[1] for p in preds]
        ghost_line.set_data(gx, gy)
        for k, dot in enumerate(ghost_dots):
            dot.set_data([gx[k]], [gy[k]])
            dot.set_alpha(max(0.05, 0.60 - k * (0.55 / MPC_PRED_STAGES)))

        dist_a_mov = (np.linalg.norm(robot_a.pos - moving_obs.pos)
                      - OBS_RADIUS - 0.015)
        dist_b_sta = (np.linalg.norm(robot_b.pos - static_obs.pos)
                      - OBS_RADIUS - 0.015)
        title.set_text(
            f"Step: {i:03d}  |  "
            f"A↔MovObs: {dist_a_mov*100:.1f} cm  |  "
            f"B↔StatObs: {dist_b_sta*100:.1f} cm  |  "
            f"Solver A: {ctrl_a.last_solver_status}  B: {ctrl_b.last_solver_status}"
        )

        return (trail_a, trail_b, pred_a, pred_b,
                patch_a, patch_b, patch_mov, ring_mov,
                *ghost_dots, ghost_line, title)

    ani = animation.FuncAnimation(
        fig, animate, frames=MAX_STEPS,
        init_func=init, interval=50, blit=False, repeat=False,
    )

    plt.tight_layout()

    if SAVE_GIF:
        print(f"Saving GIF to {GIF_NAME} …")
        ani.save(GIF_NAME, writer='pillow', fps=20)
        print("Done.")

    plt.show()


if __name__ == "__main__":
    run_obstacle_avoidance_animation()
