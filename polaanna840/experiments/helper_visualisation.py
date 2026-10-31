import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


# Helper: create a smooth irregular shape

def irregular_shape(cx, cy, radius, seed=1, irregularity=0.12, n=200):
    rng = np.random.default_rng(seed)

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    r = radius * (
        1
        + irregularity * np.sin(3 * theta + rng.uniform(0, 2*np.pi))
        + 0.06 * np.sin(5 * theta + rng.uniform(0, 2*np.pi))
    )

    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    vertices = np.column_stack([x, y])
    vertices = np.vstack([vertices, vertices[0]])

    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]

    return Path(vertices, codes)


# Create figure

fig, ax = plt.subplots(figsize=(9, 5.5))


# Nucleus N_i

nucleus_path = irregular_shape(
    cx=0.45,
    cy=0.52,
    radius=0.27,
    seed=4
)

nucleus = PathPatch(
    nucleus_path,
    alpha=0.35,
    linewidth=2
)

ax.add_patch(nucleus)


# Vague region V

x = np.linspace(0.52, 0.95, 200)

upper = 0.92 - 0.08 * np.sin(np.linspace(0, np.pi, 200))
lower = 0.10 + 0.06 * np.sin(np.linspace(0, np.pi, 200))

vague_x = np.concatenate([x, x[::-1]])
vague_y = np.concatenate([upper, lower[::-1]])

vague_vertices = np.column_stack([vague_x, vague_y])
vague_vertices = np.vstack([vague_vertices, vague_vertices[0]])

vague_codes = (
    [Path.MOVETO]
    + [Path.LINETO] * (len(vague_vertices) - 2)
    + [Path.CLOSEPOLY]
)

vague_path = Path(vague_vertices, vague_codes)

vague = PathPatch(
    vague_path,
    alpha=0.22,
    linewidth=2
)

ax.add_patch(vague)


# Highlight N_i ∩ V

intersection = PathPatch(
    vague_path,
    alpha=0.55,
    linewidth=0
)

intersection.set_clip_path(nucleus)

ax.add_patch(intersection)


# Labels inside shapes

ax.text(
    0.33,
    0.52,
    r"$N_i$",
    fontsize=26,
    ha="center",
    va="center"
)

ax.text(
    0.78,
    0.82,
    r"$V$",
    fontsize=26,
    ha="center",
    va="center"
)


# Label whole nucleus

ax.annotate(
    "nucleus area\n" + r"$|N_i|$",
    xy=(0.29, 0.30),
    xytext=(0.04, 0.18),
    fontsize=23,
    ha="right",
    va="center",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.5
    )
)


# Label intersection

ax.annotate(
    "overlap\n" + r"$|N_i \cap V|$",
    xy=(0.58, 0.52),
    xytext=(1.02, 0.49),
    fontsize=23,
    ha="left",
    va="center",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.5
    )
)


# Formula / interpretation

ax.text(
    0.50,
    -0.10,
    r"$r_i = \frac{|N_i \cap V|}{|N_i|}$",
    fontsize=28,
    ha="center",
    va="center"
)



# Formatting

ax.set_xlim(-0.45, 1.65)
ax.set_ylim(-0.40, 1.02)

ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()

plt.savefig(
    "ambiguity_metric.pdf",
    bbox_inches="tight",
    pad_inches=0.08
)

plt.show()