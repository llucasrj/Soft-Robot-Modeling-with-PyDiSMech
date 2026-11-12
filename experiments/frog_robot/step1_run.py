"""
Step 1 run: fire all four limbs once, with NO water (no external forces).

What we expect from physics:
  - Newton's method converges every step.
  - Limbs open (curvature drops) while actuated, then curl back.
  - The frame stays square.
  - The center of mass does NOT move: internal forces cannot change total momentum,
    so without a fluid to push against, the robot cannot swim.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import dismech
from frog_model import FrogParams, build_frog_geometry, to_dismech_geometry

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- build robot (placeholder material) ----------------
p = FrogParams()
KAPPA_REST = 128.0   # [repo] unactuated limb curvature, 10% EGaIn
fg = build_frog_geometry(p, rest_curvature=KAPPA_REST)
geom = dismech.GeomParams(rod_r0=0.001, shell_h=0)
material = dismech.Material(density=1000, youngs_rod=3e6, youngs_shell=0,
                            poisson_rod=0.5, poisson_shell=0)
dt, total_time = 1e-3, 0.4
sim = dismech.SimParams(static_sim=False, two_d_sim=True, use_mid_edge=False,
                        use_line_search=False, log_data=True, log_step=1, show_floor=False,
                        dt=dt, max_iter=25, total_time=total_time, plot_step=1,
                        tol=1e-4, ftol=1e-4, dtol=1e-2)
robot = dismech.SoftRobot(geom, material, to_dismech_geometry(fg), sim, dismech.Environment())
robot = robot.fix_nodes(np.arange(robot.n_nodes), axis=2)

# Placeholder: make the frame 1000x stiffer than the limbs (real values in step 2).
bs, ss = robot.bend_springs, robot.stretch_springs
n_frame = len(fg.frame_nodes)
frame_bend = np.all(bs.nodes_ind < n_frame, axis=1)        # all 3 nodes on the frame
frame_stretch = np.all(ss.nodes_ind < n_frame, axis=1)
bs.EI[frame_bend] *= 1000
ss.EA[frame_stretch] *= 1000

stepper = dismech.ImplicitEulerTimeStepper(robot)   # builds the energies, sets nat_strain = rest shape
bend = stepper.elastic_energies[dismech.BEND]

# ---------------- actuation ----------------
# Limb bend springs = those centered on a limb node (corner springs stay passive).
# In-plane bending lives in strain column 1. Each limb's sign is kept; we only scale magnitude.
limb_node_set = np.concatenate([ch[1:] for ch in fg.limb_nodes.values()])
limb_springs = np.where(np.isin(bs.nodes_ind[:, 1], limb_node_set))[0]
nat_rest = bs.nat_strain[limb_springs, 1].copy()
kappa_rest, kappa_act = KAPPA_REST, 45.0      # [repo] 10% EGaIn: ~128 -> ~45 1/m
t_on, t_act, t_ramp = 0.05, 0.070, 0.010                 # fire at 50 ms for 70 ms [paper], 10 ms ramps

def kappa_command(t):
    """Piecewise-linear stand-in for the paper's logistic actuation curve (refined in a later step)."""
    up = np.clip((t - t_on) / t_ramp, 0, 1)
    down = np.clip((t - t_on - t_act) / t_ramp, 0, 1)
    frac = up - down                                      # 0 at rest, 1 fully actuated
    return kappa_rest + frac * (kappa_act - kappa_rest)

def mass_weighted_com(q):
    m = robot.mass_matrix[:3 * robot.n_nodes:3]
    xy = q[:3 * robot.n_nodes].reshape(-1, 3)[:, :2]
    return (m[:, None] * xy).sum(0) / m.sum()

# ---------------- time loop ----------------
n_steps = int(round(total_time / dt))
ts, coms, kap, side, iters_ok, snaps = [0.0], [mass_weighted_com(robot.state.q)], [], [], True, {}
corner = [fg.corner_nodes[k] for k in ('rear_left', 'rear_right', 'front_right', 'front_left')]

def measure(r):
    k = bend.get_strain(r.state)[limb_springs, 1] / bs.voronoi_len[limb_springs]
    kap.append(np.abs(k).mean())
    c = r.state.q[:3 * r.n_nodes].reshape(-1, 3)[corner, :2]
    side.append([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)])

xy_traj = [robot.q0[:3 * robot.n_nodes].reshape(-1, 3)[:, :2].copy()]   # node (x, y) every step
measure(robot)
for i in range(n_steps):
    t_next = (i + 1) * dt
    bs.nat_strain[limb_springs, 1] = nat_rest * kappa_command(t_next) / kappa_rest
    try:
        robot, f_norm = stepper.step(robot, debug=False)
    except ValueError as e:
        iters_ok = False
        print(f"t={t_next:.3f}: {e}")
        break
    ts.append(t_next)
    xy_traj.append(robot.state.q[:3 * robot.n_nodes].reshape(-1, 3)[:, :2].copy())
    coms.append(mass_weighted_com(robot.state.q))
    measure(robot)
    if any(abs(t_next - s) < dt / 2 for s in (0.0, 0.06, 0.12, 0.2, 0.4)):
        snaps[round(t_next, 3)] = robot.state.q[:3 * robot.n_nodes].reshape(-1, 3)[:, :2].copy()
snaps = {0.0: robot.q0[:3 * robot.n_nodes].reshape(-1, 3)[:, :2], **snaps}

ts, coms, kap, side = map(np.array, (ts, coms, kap, side))
com_drift = np.linalg.norm(coms - coms[0], axis=1).max()
print(f"steps completed: {len(ts) - 1}/{n_steps}   all converged: {iters_ok}")
print(f"limb curvature: rest {kap[0]:.1f}, min during kick {kap.min():.1f}, end {kap[-1]:.1f} 1/m")
print(f"frame side lengths: {side.min()*1e3:.3f} .. {side.max()*1e3:.3f} mm (built as {p.frame_side*1e3:.1f})")
print(f"max center-of-mass drift: {com_drift*1e9:.3g} nm   <- should be ~0 (no water, no propulsion)")

# ---------------- figure ----------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for (t, xy), c in zip(sorted(snaps.items()), plt.cm.viridis(np.linspace(0, 0.9, len(snaps)))):
    for e in fg.edges:
        axs[0].plot(*xy[e].T * 1e3, color=c, lw=1.2)
    axs[0].plot([], [], color=c, label=f't = {t*1e3:.0f} ms')
axs[0].set_aspect('equal'); axs[0].legend(fontsize=8, loc='center')
axs[0].set_title('Shape snapshots (no water)'); axs[0].set_xlabel('x [mm]'); axs[0].set_ylabel('y [mm]')

axs[1].plot(ts * 1e3, kap, label='DER limb curvature')
axs[1].plot(ts * 1e3, [kappa_command(t) for t in ts], '--', label='commanded natural curvature')
axs[1].set_xlabel('t [ms]'); axs[1].set_ylabel('curvature [1/m]'); axs[1].legend(); axs[1].set_title('Limb actuation')

axs[2].plot(ts * 1e3, (coms - coms[0]) * 1e9)
axs[2].set_xlabel('t [ms]'); axs[2].set_ylabel('COM displacement [nm]'); axs[2].legend(['x', 'y'])
axs[2].set_title('Center of mass (should stay at 0)')
fig.tight_layout()
out = os.path.join(HERE, 'figures', 'step1_run.png')
fig.savefig(out, dpi=120, bbox_inches='tight')
print("saved", out)

# ---------------- trajectory + viewer ----------------
from frog_viewer import save_trajectory, make_viewer
traj = save_trajectory(os.path.join(HERE, 'outputs', 'step1_run.npz'), fg, ts, np.array(xy_traj),
                       kappa_cmd=np.array([kappa_command(t) for t in ts]), kappa=kap)
make_viewer(traj, os.path.join(HERE, 'outputs', 'step1_run'),
            title='Step 1: single limb kick, no water (placeholder material)')
