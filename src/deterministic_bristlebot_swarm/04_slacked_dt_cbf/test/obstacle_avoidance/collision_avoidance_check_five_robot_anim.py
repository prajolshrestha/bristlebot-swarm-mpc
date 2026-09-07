"""Five robots converging, animated."""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from expert_src.Ellipse2dObj import EllipticalBot2D
from expert_src.controller import CasadiAcadosEllipticalController, D_SAFE

def run_collision_animation():
    robot_a = EllipticalBot2D(
        r=np.array([-0.35, 0.0]),
        theta_rad=0.0,
        vmag=0.0, w=0.0,
        a=0.03, b=0.015
    )
    
    robot_b = EllipticalBot2D(
        r=np.array([0.0, 0.0]),
        theta_rad=0.0,
        vmag=0.0, w=0.0,
        a=0.03, b=0.015
    )

    robot_c = EllipticalBot2D(
        r=np.array([0.0, 0.35]),
        theta_rad=-np.pi/2,
        vmag=0.0, w=0.0,
        a=0.03, b=0.015
    )

    robot_d = EllipticalBot2D(
        r=np.array([-0.1, 0.2]),
        theta_rad=0.0,
        vmag=0.0, w=0.0,
        a=0.03, b=0.015
    )

    robot_e = EllipticalBot2D(
        r=np.array([0.2, -0.1]),
        theta_rad=0.0,
        vmag=0.0, w=0.0,
        a=0.03, b=0.015
    )
    
    ctrl_a = CasadiAcadosEllipticalController(dt=0.1, N=25)
    ctrl_c = CasadiAcadosEllipticalController(dt=0.1, N=25)
    
    goal_a = np.array([0.35, 0.0])
    goal_c = np.array([0.0, -0.35])
    max_steps = 485
    dt = 0.1
    SAVE_GIF  = False
    GIF_NAME  = "soft_collision_avoidance_check_five_robot.gif"
    
    history_a = [robot_a.pos.copy()]
    history_c = [robot_c.pos.copy()]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    path_line, = ax.plot([], [], 'b-', lw=1.5, label='Robot A Path', alpha=0.6)
    path_line_c, = ax.plot([], [], 'm-', lw=1.5, label='Robot C Path', alpha=0.6)
    
    goal_marker, = ax.plot(goal_a[0], goal_a[1], 'rx', markersize=10, label='Goal A')
    goal_marker_c, = ax.plot(goal_c[0], goal_c[1], 'cx', markersize=10, label='Goal C')
    
    patch_b = Ellipse(robot_b.pos, 0.06, 0.03, facecolor='green', alpha=0.5, label='Robot B')
    ax.add_patch(patch_b)
    
    safety_circle = Circle(robot_b.pos, D_SAFE/2, fill=False, color='red', linestyle='--', alpha=0.5, label=f'D_SAFE ({D_SAFE*100:.0f}cm)')
    ax.add_patch(safety_circle)
    
    patch_d = Ellipse(robot_d.pos, 0.06, 0.03, facecolor='gray', alpha=0.5, label='Robot D')
    ax.add_patch(patch_d)
    
    safety_circle_d = Circle(robot_d.pos, D_SAFE/2, fill=False, color='gray', linestyle='--', alpha=0.3)
    ax.add_patch(safety_circle_d)
    
    patch_e = Ellipse(robot_e.pos, 0.06, 0.03, facecolor='orange', alpha=0.5, label='Robot E')
    ax.add_patch(patch_e)
    
    safety_circle_e = Circle(robot_e.pos, D_SAFE/2, fill=False, color='orange', linestyle='--', alpha=0.3)
    ax.add_patch(safety_circle_e)
    
    patch_a = Ellipse(robot_a.pos, 0.06, 0.03, angle=0, facecolor='blue', alpha=0.8, label='Robot A')
    ax.add_patch(patch_a)

    patch_c = Ellipse(robot_c.pos, 0.06, 0.03, angle=-90, facecolor='magenta', alpha=0.8, label='Robot C')
    ax.add_patch(patch_c)
    
    pred_line, = ax.plot([], [], 'b:', alpha=0.4, label='Pred A')
    pred_line_c, = ax.plot([], [], 'm:', alpha=0.4, label='Pred C')

    title = ax.set_title("TC-DMPC Collision Avoidance Check")
    
    def init():
        path_line.set_data([], [])
        path_line_c.set_data([], [])
        pred_line.set_data([], [])
        pred_line_c.set_data([], [])
        patch_a.set_center(robot_a.pos)
        patch_c.set_center(robot_c.pos)
        return path_line, path_line_c, pred_line, pred_line_c, patch_a, patch_c, title

    def animate(i):
        nonlocal robot_a, robot_c
        
        goal_reached_a = np.linalg.norm(robot_a.pos - goal_a) < 0.02
        goal_reached_c = np.linalg.norm(robot_c.pos - goal_c) < 0.02

        if goal_reached_a and goal_reached_c:
            title.set_text(f"Both Goals Reached! Step: {i}")
            return path_line, path_line_c, pred_line, pred_line_c, patch_a, patch_c, title

        traj_b = np.tile(robot_b.pos, (41, 1))
        traj_d = np.tile(robot_d.pos, (41, 1))
        traj_e = np.tile(robot_e.pos, (41, 1))
        
        traj_a_neighbor = ctrl_a.get_predicted_trajectory()
        if traj_a_neighbor is None:
            traj_a_neighbor = np.tile(robot_a.pos, (41, 1))
            
        traj_c_neighbor = ctrl_c.get_predicted_trajectory()
        if traj_c_neighbor is None:
            traj_c_neighbor = np.tile(robot_c.pos, (41, 1))

        if not goal_reached_a:
            u_a = ctrl_a.compute_control(
                robot_a,
                goal=goal_a,
                neighbour_trajs=[traj_b, traj_c_neighbor, traj_d, traj_e],
                robot_name="RobotA"
            )
            ctrl_a.apply_control(robot_a, u_a)

        if not goal_reached_c:
            u_c = ctrl_c.compute_control(
                robot_c,
                goal=goal_c,
                neighbour_trajs=[traj_b, traj_a_neighbor, traj_d, traj_e],
                robot_name="RobotC"
            )
            ctrl_c.apply_control(robot_c, u_c)
        
        history_a.append(robot_a.pos.copy())
        history_c.append(robot_c.pos.copy())
        hist_a_np = np.array(history_a)
        hist_c_np = np.array(history_c)
        
        path_line.set_data(hist_a_np[:, 0], hist_a_np[:, 1])
        patch_a.set_center(robot_a.pos)
        patch_a.set_facecolor('black' if ctrl_a.last_solver_status != 0 else 'blue')
        patch_a.angle = np.rad2deg(robot_a.theta)
        
        pred_traj_a = ctrl_a.get_predicted_trajectory()
        if pred_traj_a is not None:
            pred_line.set_data(pred_traj_a[:, 0], pred_traj_a[:, 1])

        path_line_c.set_data(hist_c_np[:, 0], hist_c_np[:, 1])
        patch_c.set_center(robot_c.pos)
        patch_c.set_facecolor('black' if ctrl_c.last_solver_status != 0 else 'magenta')
        patch_c.angle = np.rad2deg(robot_c.theta)
        
        pred_traj_c = ctrl_c.get_predicted_trajectory()
        if pred_traj_c is not None:
            pred_line_c.set_data(pred_traj_c[:, 0], pred_traj_c[:, 1])
            
        dist_ab = np.linalg.norm(robot_a.pos - robot_b.pos)
        dist_ac = np.linalg.norm(robot_a.pos - robot_c.pos)
        dist_cb = np.linalg.norm(robot_c.pos - robot_b.pos)
        
        title.set_text(f"Step: {i} | AC Dist: {dist_ac*100:.1f} cm | Status A: {ctrl_a.last_solver_status} C: {ctrl_c.last_solver_status}")
        
        return path_line, path_line_c, pred_line, pred_line_c, patch_a, patch_c, title

    ani = animation.FuncAnimation(
        fig, animate, frames=max_steps, init_func=init,
        interval=50, blit=True, repeat=False
    )
    
    plt.legend(loc='upper left', fontsize='small')
    
    if SAVE_GIF:
        print(f"Saving animation to {GIF_NAME} …")
        ani.save(GIF_NAME, writer="pillow", fps=int(1000 / 50))
        print("Save complete.")

    plt.show()

if __name__ == "__main__":
    run_collision_animation()
