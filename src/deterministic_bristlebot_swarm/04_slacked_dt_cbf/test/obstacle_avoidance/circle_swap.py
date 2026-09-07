"""Robots swap across a circle, the classic head-on avoidance test."""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle
from matplotlib.lines import Line2D
import matplotlib as mpl
from scipy.spatial import KDTree

N_ROBOTS      = 6
CIRCLE_RADIUS = 0.30
SAVE_GIF      = False
GIF_NAME      = f"circle_swap_{N_ROBOTS}robots_R{int(CIRCLE_RADIUS*100)}cm.gif"
GIF_FPS       = 15

DT            = 0.1
N_HORIZON     = 15
MAX_STEPS     = 100
GOAL_TOL      = 0.08



def _robot_colors(n: int):
    """Return n visually distinct colours (hex strings)."""
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = []
    for h in hues:
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.85)
        colors.append("#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255)))
    return colors

plt.style.use('default')
mpl.rcParams.update({
    'axes.facecolor':  '#FFFFFF',
    'figure.facecolor':'#FFFFFF',
    'grid.color':      '#E5E7EB',
    'axes.edgecolor':  '#9CA3AF',
    'text.color':      '#111827',
    'axes.labelcolor': '#374151',
    'xtick.color':     '#4B5563',
    'ytick.color':     '#4B5563',
    'font.family':     'sans-serif',
    'font.sans-serif': ['Inter', 'Roboto', 'Arial'],
})

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from expert_src.Ellipse2dObj import EllipticalBot2D
from expert_src.controller import CasadiAcadosEllipticalController, D_SAFE


def run_circle_swap_animation(
    n_robots: int      = N_ROBOTS,
    circle_radius: float = CIRCLE_RADIUS,
):
    """Run the N-robot circle swap animation."""

    colors = _robot_colors(n_robots)

    angles = np.linspace(0, 2 * np.pi, n_robots, endpoint=False)
    starts = np.column_stack([
        circle_radius * np.cos(angles),
        circle_radius * np.sin(angles),
    ])
    goals  = np.column_stack([
        circle_radius * np.cos(angles + np.pi),
        circle_radius * np.sin(angles + np.pi),
    ])

    robots = []
    for i, (pos, angle) in enumerate(zip(starts, angles)):
        heading = np.arctan2(goals[i, 1] - pos[1], goals[i, 0] - pos[0])
        robots.append(EllipticalBot2D(
            r=pos.copy(),
            theta_rad=heading,
            vmag=0.0, w=0.0,
            a=0.03, b=0.015,
        ))

    ctrls = [
        CasadiAcadosEllipticalController(
            dt=DT, N=N_HORIZON,
        )
        for _ in range(n_robots)
    ]

    arena_half = max(circle_radius + 0.12, 0.45)
    fig_size   = max(8, min(14, 8 + n_robots * 0.15))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.set_xlim(-arena_half, arena_half)
    ax.set_ylim(-arena_half, arena_half)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5, color='#E5E7EB')
    ax.set_facecolor('#FFFFFF')

    arena_circle = Circle((0, 0), arena_half * 0.95,
                           fill=False, edgecolor='#9CA3AF',
                           linestyle='--', linewidth=1.5)
    ax.add_patch(arena_circle)

    ref_circle = Circle((0, 0), circle_radius,
                         fill=False, edgecolor='#6B7280',
                         linestyle=':', linewidth=1.2, alpha=0.7)
    ax.add_patch(ref_circle)

    for i in range(n_robots):
        ax.plot(goals[i, 0], goals[i, 1],
                marker='*', color=colors[i], markersize=10,
                linestyle='None', alpha=0.7, zorder=3)

    path_lines   = []
    pred_lines   = []
    body_patches = []
    for i, robot in enumerate(robots):
        c = colors[i]
        edge = c

        path_line, = ax.plot([], [], color=c, lw=1.5, alpha=0.6, zorder=4)
        pred_line, = ax.plot([], [], color=c, linestyle='--', alpha=0.3, zorder=3)
        body = Ellipse(robot.pos, robot.bot_width, robot.bot_height,
                       angle=np.rad2deg(robot.theta),
                       facecolor=c, alpha=0.90,
                       edgecolor='#1F2937', linewidth=0.8, zorder=6)
        ax.add_patch(body)

        path_lines.append(path_line)
        pred_lines.append(pred_line)
        body_patches.append(body)

    histories = [[r.pos.copy()] for r in robots]

    title = ax.set_title(
        f"DMPC Circle Swap – {n_robots} Robots | R = {circle_radius*100:.0f} cm",
        fontsize=13, fontweight='bold', color='#111827', pad=14,
    )

    legend_handles = [
        Line2D([0], [0], color=colors[i], linewidth=2,
               label=f'Robot {i}')
        for i in range(min(n_robots, 6))
    ]
    if n_robots > 6:
        legend_handles.append(
            Line2D([0], [0], color='gray', linewidth=2, label=f'+{n_robots-6} more…')
        )
    ax.legend(handles=legend_handles, loc='lower left',
              frameon=True, facecolor='#F9FAFB', edgecolor='#D1D5DB',
              labelcolor='#111827', fontsize='x-small', ncol=2)

    def init():
        for pl in path_lines:
            pl.set_data([], [])
        for pr in pred_lines:
            pr.set_data([], [])
        return (*path_lines, *pred_lines, *body_patches, title)

    def animate(frame_idx):
        all_done = all(
            np.linalg.norm(robots[i].pos - goals[i]) < GOAL_TOL
            for i in range(n_robots)
        )
        if all_done:
            title.set_text(f"All {n_robots} robots reached goals! (frame {frame_idx})")
            return (*path_lines, *pred_lines, *body_patches, title)

        kdtree = KDTree(np.array([r.pos for r in robots]))

        prev_preds = []
        for ci in ctrls:
            p = ci.get_predicted_trajectory()
            if p is None:
                p = np.zeros((N_HORIZON + 1, 4))
            prev_preds.append(p)

        all_nb_trajs = []
        K = ctrls[0]._max_nb
        COMM_RADIUS = 0.30
        
        for i in range(n_robots):
            pos_i = robots[i].pos
            dists, indices = kdtree.query(pos_i, k=K+1, distance_upper_bound=COMM_RADIUS)
            
            if isinstance(indices, (int, np.integer)):
                indices = [indices]
                dists = [dists]
                
            trajs = []
            for d, j in zip(dists, indices):
                if j == i or j >= n_robots or d > COMM_RADIUS:
                    continue
                trajs.append(prev_preds[j])
            all_nb_trajs.append(trajs)

        controls = []
        new_preds = [None] * n_robots
        for i in range(n_robots):
            goal_pos = goals[i]
            nb_trajs = all_nb_trajs[i]
            
            u = ctrls[i].compute_control(
                robots[i], goal=goal_pos,
                neighbour_trajs=nb_trajs,
                robot_name=f"R{i}",
            )
            controls.append(u)

        statuses = []
        for i in range(n_robots):
            ctrls[i].apply_control(robots[i], controls[i])
            new_preds[i] = ctrls[i].get_predicted_trajectory()
            
            histories[i].append(robots[i].pos.copy())
            hist = np.array(histories[i])
            path_lines[i].set_data(hist[:, 0], hist[:, 1])

            status = ctrls[i].last_solver_status
            statuses.append(status)
            body_patches[i].set_center(robots[i].pos)
            body_patches[i].set_angle(np.rad2deg(robots[i].theta))
            body_patches[i].set_facecolor('#000000' if status != 0 else colors[i])

            pt = new_preds[i]
            if pt is not None:
                pred_lines[i].set_data(pt[:, 0], pt[:, 1])

        n_ok  = statuses.count(0)
        min_dist = float('inf')
        for i in range(n_robots):
            for j in range(i + 1, n_robots):
                d = np.linalg.norm(robots[i].pos - robots[j].pos)
                if d < min_dist:
                    min_dist = d
        title.set_text(
            f"Frame {frame_idx:03d} | Robots: {n_robots} | R={circle_radius*100:.0f}cm | "
            f"Min Dist: {min_dist*100:.1f}cm | Solver OK: {n_ok}/{n_robots}"
        )

        return (*path_lines, *pred_lines, *body_patches, title)

    ani = animation.FuncAnimation(
        fig, animate, frames=MAX_STEPS,
        init_func=init, interval=int(1000 / GIF_FPS),
        blit=False, repeat=False,
    )

    if SAVE_GIF:
        out_path = os.path.join(os.path.dirname(__file__), GIF_NAME)
        print(f"Saving GIF → {out_path}")
        ani.save(out_path, writer="pillow", fps=GIF_FPS)
        print("Saved.")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_circle_swap_animation(
        n_robots=N_ROBOTS,
        circle_radius=CIRCLE_RADIUS,
    )
