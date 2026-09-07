"""Exercises inter-robot and obstacle avoidance together."""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle, Rectangle
from matplotlib.lines import Line2D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sim_engine.obstacle_model import ObstacleManager, StaticObstacle, MovingObstacle
from expert_src.Ellipse2dObj import EllipticalBot2D
from expert_src.controller import CasadiAcadosEllipticalController, D_SAFE, D_SAFE_OBS

def run_combined_animation():
    robot_a = EllipticalBot2D(
        r=np.array([-0.35, 0.0]),
        theta_rad=0.0,
        vmag=0.0, w=0.0,
        a=0.03, b=0.015
    )
    goal_a = np.array([0.35, 0.0])
    
    robot_c = EllipticalBot2D(
        r=np.array([0.0, 0.35]),
        theta_rad=-np.pi/2,
        vmag=0.0, w=0.0,
        a=0.03, b=0.015
    )
    goal_c = np.array([0.0, -0.35])

    robot_b = EllipticalBot2D(
        r=np.array([0.2, -0.2]),
        theta_rad=0.0,
        vmag=0.0, w=0.0,
        a=0.03, b=0.015
    )

    DT = 0.1
    N_HORIZON = 15
    ctrl_a = CasadiAcadosEllipticalController(dt=DT, N=N_HORIZON)
    ctrl_c = CasadiAcadosEllipticalController(dt=DT, N=N_HORIZON)
    
    ARENA_HALF = 0.45
    OBS_RADIUS = 0.03
    MPC_PRED_STAGES = 15

    manager = ObstacleManager(dt=DT, arena_half=ARENA_HALF, max_obs=5, obs_range=0.60)

    moving_obs = MovingObstacle(
        pos=[0.25, 0.25],
        vel=[-0.05, -0.05],
        radius=OBS_RADIUS,
        arena_half=ARENA_HALF,
        bounce=True,
    )
    static_obs = StaticObstacle(
        pos=[-0.15, 0.15],
        radius=OBS_RADIUS,
    )
    manager.add(moving_obs)
    manager.add(static_obs)

    max_steps = 200
    SAVE_GIF  = False
    GIF_NAME  = "test_demo/collision_and_obstacle_avoidance_dsafeobs_10cm_n_15_obsradius_3cm.gif"
    
    history_a = [robot_a.pos.copy()]
    history_c = [robot_c.pos.copy()]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    
    arena_rect = Rectangle(
        (-ARENA_HALF, -ARENA_HALF), 2 * ARENA_HALF, 2 * ARENA_HALF,
        linewidth=1.2, edgecolor='black', facecolor='none', linestyle='--',
    )
    ax.add_patch(arena_rect)

    path_line_a, = ax.plot([], [], '-', lw=1.5, color='#00d4ff', label='Robot A Path', alpha=0.6)
    path_line_c, = ax.plot([], [], '-', lw=1.5, color='#ff00ff', label='Robot C Path', alpha=0.6)
    
    goal_marker_a, = ax.plot(goal_a[0], goal_a[1], 'x', color='#00d4ff', markersize=10, mew=2)
    goal_marker_c, = ax.plot(goal_c[0], goal_c[1], 'x', color='#ff00ff', markersize=10, mew=2)
    
    patch_b = Ellipse(robot_b.pos, 0.06, 0.03, facecolor='#f9a825', alpha=0.8, label='Robot B')
    ax.add_patch(patch_b)
    safety_circle_b = Circle(robot_b.pos, D_SAFE/2, fill=False, color='#f9a825', linestyle='--', alpha=0.5)
    ax.add_patch(safety_circle_b)
    
    patch_a = Ellipse(robot_a.pos, 0.06, 0.03, angle=0, facecolor='#00d4ff', alpha=0.8)
    ax.add_patch(patch_a)

    patch_c = Ellipse(robot_c.pos, 0.06, 0.03, angle=-90, facecolor='#ff00ff', alpha=0.8)
    ax.add_patch(patch_c)
    
    pred_line_a, = ax.plot([], [], ':', color='#00d4ff', alpha=0.4)
    pred_line_c, = ax.plot([], [], ':', color='#ff00ff', alpha=0.4)

    patch_mov = Circle(moving_obs.pos.copy(), OBS_RADIUS, color='#ff4444', alpha=0.80, zorder=5)
    ring_mov  = Circle(moving_obs.pos.copy(), OBS_RADIUS + D_SAFE_OBS / 2,
                       fill=False, edgecolor='#ff4444', linestyle='--', lw=1.2, alpha=0.50, zorder=4)
    ax.add_patch(patch_mov)
    ax.add_patch(ring_mov)

    ghost_dots = [
        ax.plot([], [], 'o', ms=4, color='#ff8888', alpha=0.0)[0]
        for _ in range(MPC_PRED_STAGES + 1)
    ]
    ghost_line, = ax.plot([], [], '-', lw=0.8, color='#ff4444', alpha=0.30)

    patch_sta = Circle(static_obs.pos.copy(), OBS_RADIUS, color='#cc44ff', alpha=0.80, zorder=5)
    ring_sta  = Circle(static_obs.pos.copy(), OBS_RADIUS + D_SAFE_OBS / 2,
                       fill=False, edgecolor='#cc44ff', linestyle='--', lw=1.2, alpha=0.50, zorder=4)
    ax.add_patch(patch_sta)
    ax.add_patch(ring_sta)

    title = ax.set_title("TC-DMPC Collision & Obstacle Avoidance", color='black', fontsize=11, pad=8)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00d4ff', ms=8, label='Robot A'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff00ff', ms=8, label='Robot C'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f9a825', ms=8, label='Robot B (Static)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', ms=8, label='Moving Obstacle'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#cc44ff', ms=8, label='Static Obstacle'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, facecolor='white', framealpha=0.85)

    def init():
        path_line_a.set_data([], [])
        path_line_c.set_data([], [])
        pred_line_a.set_data([], [])
        pred_line_c.set_data([], [])
        ghost_line.set_data([], [])
        for dot in ghost_dots:
            dot.set_data([], [])
        return (path_line_a, path_line_c, pred_line_a, pred_line_c,
                patch_a, patch_c, patch_mov, ring_mov, *ghost_dots, ghost_line, title)

    def animate(i):
        goal_reached_a = np.linalg.norm(robot_a.pos - goal_a) < 0.02
        goal_reached_c = np.linalg.norm(robot_c.pos - goal_c) < 0.02

        if goal_reached_a and goal_reached_c:
            title.set_text(f"Both Goals Reached! Step: {i:03d}")
            return (path_line_a, path_line_c, pred_line_a, pred_line_c,
                    patch_a, patch_c, patch_mov, ring_mov, *ghost_dots, ghost_line, title)

        manager.step(DT)

        traj_b = np.tile(np.concatenate([robot_b.pos, robot_b.velocity]), (N_HORIZON + 1, 1))
        
        traj_a_neighbor = ctrl_a.get_predicted_trajectory()
        if traj_a_neighbor is None:
            traj_a_neighbor = np.tile(np.concatenate([robot_a.pos, robot_a.velocity]), (N_HORIZON + 1, 1))
            
        traj_c_neighbor = ctrl_c.get_predicted_trajectory()
        if traj_c_neighbor is None:
            traj_c_neighbor = np.tile(np.concatenate([robot_c.pos, robot_c.velocity]), (N_HORIZON + 1, 1))

        if not goal_reached_a:
            u_a = ctrl_a.compute_control(
                robot_a,
                goal=goal_a,
                neighbour_trajs=[traj_b, traj_c_neighbor],
                obstacle_manager=manager,
                robot_name="RobotA"
            )
            ctrl_a.apply_control(robot_a, u_a)
            history_a.append(robot_a.pos.copy())

        if not goal_reached_c:
            u_c = ctrl_c.compute_control(
                robot_c,
                goal=goal_c,
                neighbour_trajs=[traj_b, traj_a_neighbor],
                obstacle_manager=manager,
                robot_name="RobotC"
            )
            ctrl_c.apply_control(robot_c, u_c)
            history_c.append(robot_c.pos.copy())
        
        hist_a_np = np.array(history_a)
        path_line_a.set_data(hist_a_np[:, 0], hist_a_np[:, 1])
        patch_a.set_center(robot_a.pos)
        patch_a.set_facecolor('black' if ctrl_a.last_solver_status != 0 else '#00d4ff')
        patch_a.angle = np.rad2deg(robot_a.theta)
        
        pred_traj_a = ctrl_a.get_predicted_trajectory()
        if pred_traj_a is not None:
            pred_line_a.set_data(pred_traj_a[:, 0], pred_traj_a[:, 1])

        hist_c_np = np.array(history_c)
        path_line_c.set_data(hist_c_np[:, 0], hist_c_np[:, 1])
        patch_c.set_center(robot_c.pos)
        patch_c.set_facecolor('black' if ctrl_c.last_solver_status != 0 else '#ff00ff')
        patch_c.angle = np.rad2deg(robot_c.theta)
        
        pred_traj_c = ctrl_c.get_predicted_trajectory()
        if pred_traj_c is not None:
            pred_line_c.set_data(pred_traj_c[:, 0], pred_traj_c[:, 1])
            
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

        dist_ac = np.linalg.norm(robot_a.pos - robot_c.pos)
        
        title.set_text(f"Step: {i:03d} | AC Dist: {dist_ac*100:.1f} cm | Status A: {ctrl_a.last_solver_status} C: {ctrl_c.last_solver_status}")
        
        return (path_line_a, path_line_c, pred_line_a, pred_line_c,
                patch_a, patch_c, patch_mov, ring_mov, *ghost_dots, ghost_line, title)

    ani = animation.FuncAnimation(
        fig, animate, frames=max_steps, init_func=init,
        interval=50, blit=False, repeat=False
    )
    
    plt.tight_layout()
    
    if SAVE_GIF:
        print(f"Saving animation to {GIF_NAME} …")
        ani.save(GIF_NAME, writer="pillow", fps=int(1000 / 50))
        print("Save complete.")

    plt.show()

if __name__ == "__main__":
    run_combined_animation()
