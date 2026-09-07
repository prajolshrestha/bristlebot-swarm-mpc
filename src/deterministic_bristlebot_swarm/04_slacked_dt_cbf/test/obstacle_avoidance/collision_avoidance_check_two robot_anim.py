"""Two robots meeting head-on, animated."""
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
    
    ctrl_a = CasadiAcadosEllipticalController(dt=0.1, N=15)
    
    goal_a = np.array([0.35, 0.0])
    max_steps = 200
    dt = 0.1
    SAVE_GIF  = True
    GIF_NAME  = "soft_collision_avoidance_check_two_robot.gif"
    
    history_a = [robot_a.pos.copy()]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    path_line, = ax.plot([], [], 'b-', lw=1.5, label='Robot A Path', alpha=0.6)
    goal_marker, = ax.plot(goal_a[0], goal_a[1], 'rx', markersize=10, label='Goal A')
    
    patch_b = Ellipse(robot_b.pos, 0.06, 0.03, facecolor='green', alpha=0.5, label='Robot B')
    ax.add_patch(patch_b)
    
    safety_circle = Circle(robot_b.pos, D_SAFE/2, fill=False, color='red', linestyle='--', alpha=0.5, label=f'D_SAFE ({D_SAFE*100:.0f}cm)')
    ax.add_patch(safety_circle)
    
    patch_a = Ellipse(robot_a.pos, 0.06, 0.03, angle=0, facecolor='blue', alpha=0.8, label='Robot A')
    ax.add_patch(patch_a)
    
    pred_line, = ax.plot([], [], 'b:', alpha=0.4, label='Predicted Traj')

    title = ax.set_title("TC-DMPC Collision Avoidance Check")
    
    def init():
        path_line.set_data([], [])
        pred_line.set_data([], [])
        patch_a.set_center(robot_a.pos)
        return path_line, pred_line, patch_a, title

    def animate(i):
        nonlocal robot_a
        
        if np.linalg.norm(robot_a.pos - goal_a) < 0.02:
            title.set_text(f"Goal Reached! Step: {i}")
            return path_line, pred_line, patch_a, title

        traj_b = np.tile(robot_b.pos, (41, 1))
        u_a = ctrl_a.compute_control(
            robot_a,
            goal=goal_a,
            neighbour_trajs=[traj_b],
            robot_name="RobotA"
        )
        
        ctrl_a.apply_control(robot_a, u_a)
        
        history_a.append(robot_a.pos.copy())
        hist_np = np.array(history_a)
        
        path_line.set_data(hist_np[:, 0], hist_np[:, 1])
        patch_a.set_center(robot_a.pos)
        patch_a.set_facecolor('black' if ctrl_a.last_solver_status != 0 else 'blue')
        patch_a.angle = np.rad2deg(robot_a.theta)
        
        pred_traj = ctrl_a.get_predicted_trajectory()
        if pred_traj is not None:
            pred_line.set_data(pred_traj[:, 0], pred_traj[:, 1])
            
        dist = np.linalg.norm(robot_a.pos - robot_b.pos)
        title.set_text(f"Step: {i} | Dist: {dist*100:.1f} cm | Status: {ctrl_a.last_solver_status}")
        
        return path_line, pred_line, patch_a, title

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
