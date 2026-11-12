"""Step 1: build the frog geometry, turn it into a SoftRobot, and inspect what PyDiSMech made."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import dismech
from frog_model import FrogParams, build_frog_geometry, to_dismech_geometry

HERE = os.path.dirname(os.path.abspath(__file__))

p = FrogParams()
KAPPA_REST = 128.0   # [repo] unactuated limb curvature, 10% EGaIn
fg = build_frog_geometry(p, rest_curvature=KAPPA_REST)
geo = to_dismech_geometry(fg)

# Placeholder material/geometry so a SoftRobot can be constructed; real values come in step 2.
geom = dismech.GeomParams(rod_r0=0.001, shell_h=0)
material = dismech.Material(density=1000, youngs_rod=3e6, youngs_shell=0,
                            poisson_rod=0.5, poisson_shell=0)
sim = dismech.SimParams(static_sim=False, two_d_sim=True, use_mid_edge=False,
                        use_line_search=False, log_data=True, log_step=1, show_floor=False,
                        dt=1e-3, max_iter=25, total_time=1.0, plot_step=1,
                        tol=1e-4, ftol=1e-4, dtol=1e-2)
robot = dismech.SoftRobot(geom, material, geo, sim, dismech.Environment())

# Keep the motion in the x-y plane: fix every node's z DOF (and, in 2D, all twist DOFs).
robot = robot.fix_nodes(np.arange(robot.n_nodes), axis=2)

# ---------------- report ----------------
print(f"nodes: {robot.n_nodes}  (frame {len(fg.frame_nodes)}, limbs {sum(len(v) - 1 for v in fg.limb_nodes.values())})")
print(f"edges: {len(robot.edges)}")
print(f"edge lengths: min {robot.ref_len.min()*1e3:.2f} mm, max {robot.ref_len.max()*1e3:.2f} mm")
print(f"DOFs: {robot.n_dof} total = 3*{robot.n_nodes} node coords + {robot.n_dof - 3*robot.n_nodes} edge twist angles")
print(f"free DOFs after 2D constraint: {len(robot.state.free_dof)}  (= 2 * {robot.n_nodes} in-plane coords)")
print(f"stretch springs: {robot.stretch_springs.N}   bend springs: {robot.bend_springs.N}")

bs_nodes = robot.bend_springs.nodes_ind
for name, c in fg.corner_nodes.items():
    at_corner = np.where(bs_nodes[:, 1] == c)[0]
    desc = []
    for i in at_corner:
        n0, _, n2 = bs_nodes[i]
        part = lambda n: 'limb' if n >= len(fg.frame_nodes) else 'frame'
        desc.append(f"#{i}: {part(n0)}-{part(n2)}")
    print(f"corner {name:<11} (node {c:>2}): {len(at_corner)} bend springs -> " + ", ".join(desc))

# Measured rest curvature of the limbs: PyDiSMech's bend strain is the *integrated*
# curvature 2*tan(phi/2); divide by the Voronoi length to get curvature in 1/m.
# For a planar rod, in-plane bending is strain column 1.
be = dismech.BendEnergy(robot.bend_springs, robot.state)
kappa = be.get_strain(robot.state)[:, 1] / robot.bend_springs.voronoi_len
limb_springs = np.where(bs_nodes[:, 1] >= len(fg.frame_nodes))[0]
print(f"limb rest curvature from DER: {np.abs(kappa[limb_springs]).mean():.1f} 1/m (target {KAPPA_REST})")

# ---------------- picture ----------------
fig, ax = plt.subplots(figsize=(6, 6))
for e in fg.edges[fg.frame_edges]:
    ax.plot(*fg.nodes[e, :2].T * 1e3, color='0.4', lw=3)
colors = dict(front_left='tab:red', front_right='tab:orange', rear_left='tab:blue', rear_right='tab:cyan')
for name, chain in fg.limb_nodes.items():
    ax.plot(*fg.nodes[chain, :2].T * 1e3, '-o', ms=3, color=colors[name], label=name)
ax.plot(*fg.nodes[fg.frame_nodes, :2].T * 1e3, '.', color='k', ms=3)
for name, c in fg.corner_nodes.items():
    ax.plot(*fg.nodes[c, :2] * 1e3, 's', color='k', ms=7)
ax.annotate('', xy=(0, 25), xytext=(0, -5), arrowprops=dict(arrowstyle='->', lw=2))
ax.text(3, 10, 'swim direction (+y)')
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.set_title(f'Frog robot, {robot.n_nodes} nodes (squares = frame/limb junctions)')
ax.legend(loc='center', fontsize=8)
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', 'step1_geometry.png')
fig.savefig(out, dpi=130, bbox_inches='tight')
print("saved", out)
