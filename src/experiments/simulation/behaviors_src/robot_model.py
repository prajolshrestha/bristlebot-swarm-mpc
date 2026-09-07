import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# IEEE-friendly defaults: serif body, STIX math, vector-ready export
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times', 'serif'],
    'font.size': 10,
    'mathtext.fontset': 'stix',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Parameters
a = 0.3   # semi-major (horizontal)
b = 0.15  # semi-minor (vertical)
d_safe = 0.7  # center-to-center safety distance (7 cm in dm units)

t = np.linspace(0, 2 * np.pi, 400)

x = a * np.cos(t)
y = b * np.sin(t)

center_left = -d_safe / 2
center_right = d_safe / 2
r_safe = d_safe / 2

# Paper-scale styling (half-column schematic, ~3.5 in wide)
FIG_W, FIG_H = 3.5, 2.0
FONT_SIZE = 10
ROBOT_LW = 2.2       # bold robot footprint
SAFETY_LW = 1.0
ANNOT_LW = 0.9
EXT_LW = 0.6
MUTATION_SCALE = 8

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

# Robot boundaries (bold)
ax.plot(x + center_left, y, 'k', linewidth=ROBOT_LW, solid_capstyle='round', zorder=3)
ax.plot(x + center_right, y, 'k', linewidth=ROBOT_LW, solid_capstyle='round', zorder=3)

# Circular safety buffers (radius d_safe / 2) around each robot
t_circle = np.linspace(0, 2 * np.pi, 400)
x_circle = r_safe * np.cos(t_circle)
y_circle = r_safe * np.sin(t_circle)
ax.plot(x_circle + center_left, y_circle, color='#d32f2f', linestyle='--',
        linewidth=SAFETY_LW, alpha=0.85, zorder=2)
ax.plot(x_circle + center_right, y_circle, color='#d32f2f', linestyle='--',
        linewidth=SAFETY_LW, alpha=0.85, zorder=2)

# Annotations for 'a' and 'b' on the LEFT ellipse
ax.annotate('', xy=(center_left, 0), xytext=(center_left + a, 0),
            arrowprops=dict(arrowstyle='<->', color='k', lw=ANNOT_LW, mutation_scale=MUTATION_SCALE))
ax.text(center_left + a / 2, -0.03, r'$a$', ha='center', va='top', fontsize=FONT_SIZE, color='k')

ax.annotate('', xy=(center_left, 0), xytext=(center_left, b),
            arrowprops=dict(arrowstyle='<->', color='k', lw=ANNOT_LW, mutation_scale=MUTATION_SCALE))
ax.text(center_left - 0.03, b / 2, r'$b$', ha='right', va='center', fontsize=FONT_SIZE, color='k')

# Annotations for 'a' and 'b' on the RIGHT ellipse
ax.annotate('', xy=(center_right, 0), xytext=(center_right + a, 0),
            arrowprops=dict(arrowstyle='<->', color='k', lw=ANNOT_LW, mutation_scale=MUTATION_SCALE))
ax.text(center_right + a / 2, -0.03, r'$a$', ha='center', va='top', fontsize=FONT_SIZE, color='k')

ax.annotate('', xy=(center_right, 0), xytext=(center_right, b),
            arrowprops=dict(arrowstyle='<->', color='k', lw=ANNOT_LW, mutation_scale=MUTATION_SCALE))
ax.text(center_right - 0.03, b / 2, r'$b$', ha='right', va='center', fontsize=FONT_SIZE, color='k')

# D_safe: horizontal center-to-center span; dimension line sits above the buffers
dsafe_y = r_safe + 0.15
EXT_COLOR = '0.45'
for cx in (center_left, center_right):
    ax.plot([cx, cx], [r_safe, dsafe_y], color=EXT_COLOR, linewidth=EXT_LW, zorder=1)
ax.annotate('', xy=(center_left, dsafe_y), xytext=(center_right, dsafe_y),
            arrowprops=dict(arrowstyle='|-|,widthA=0.35,widthB=0.35', color='k',
                            lw=ANNOT_LW, mutation_scale=MUTATION_SCALE))
ax.text(0, dsafe_y + 0.03, r'$D_{\mathrm{safe}}$', ha='center', va='bottom',
        fontsize=FONT_SIZE, color='k')

margin = 0.2
ax.set_aspect('equal')
ax.set_xlim(center_left - a - margin, center_right + a + margin)
ax.set_ylim(-r_safe - margin, dsafe_y + margin)
ax.axis('off')

plt.tight_layout(pad=0.05)
# Named for the paper figure it becomes (Fig. 1 left). PAPER_FIGS lets the figure be
# regenerated straight into the draft instead of being renamed and copied by hand.
import os
PAPER_FIGS = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "results", "figures"))
os.makedirs(PAPER_FIGS, exist_ok=True)
for _dir in (PAPER_FIGS,):
    if os.path.isdir(_dir):
        plt.savefig(os.path.join(_dir, 'robot_model.png'), dpi=300,
                    bbox_inches='tight', pad_inches=0.02)
        plt.savefig(os.path.join(_dir, 'robot_model.pdf'),
                    bbox_inches='tight', pad_inches=0.02)
        print(f"  wrote robot_model.png / .pdf -> {_dir}")
