"""
Frog-inspired swimming soft robot (Huang et al., Adv. Intell. Syst. 2022) in PyDiSMech.

The robot is a planar network of discrete elastic rods:
  - a stiff square frame (the foam body), as one closed loop of rod edges
  - four soft curved limbs (the SMA actuators), one hanging off each corner

Physics (paper, Experimental Section):
  - stretching + bending elasticity, no twist (2D)                        -> PyDiSMech springs
  - SMA actuation = time-varying natural curvature of the limbs (Eq. 9)   -> FrogActuator
  - limb stiffens while hot (E0 -> Emax)                                  -> FrogActuator
  - quadratic anisotropic drag, separate limb/body coefficients (Eq. 1-2) -> 'anisotropicDrag' force
  - added mass folded into the mass matrix, M_i = (rho_r + Ca rho_f) V_i  (Eq. 4)
  - implicit Euler + Newton-Raphson (Eq. 6-8)                             -> ImplicitEulerTimeStepper

Coordinates: x = lateral (right is +x), y = forward (swimming direction), z = 0. SI units.

Parameter sources are tagged:
  [paper]    stated in the paper
  [repo]     derived from github.com/softmachineslab/frog data (paper_data/frog_reference.json)
  [inferred] deduced from the paper's results
  [guess]    not available; placeholder until the authors' option.txt is known
"""
import dataclasses
import json
import os

import numpy as np

import dismech

if not hasattr(getattr(dismech, 'external_forces', None), 'compute_anisotropic_drag'):
    raise ImportError(
        f"This model needs this repository's PyDiSMech (src/dismech), but Python loaded {dismech.__file__}.\n"
        f"Use the project environment .venv313 (in VS Code: 'Python: Select Interpreter' -> .venv313), e.g.\n"
        f"    .venv313\\Scripts\\python.exe experiments\\frog_robot\\validate.py")

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_FILE = os.path.join(HERE, 'paper_data', 'frog_reference.json')


@dataclasses.dataclass
class FrogParams:
    # --- Gait / design (the variables of the paper's sweeps) ---
    egain: int = 10                  # EGaIn volume % in the limb interface layer: 0..50
    freq: float = 0.4                # actuation frequency (Hz)
    phase: float = 0.5               # rear-limb delay as a fraction of the period (0..1)

    # --- Frame (foam body) ---
    frame_side: float = 0.090        # [paper] L_o = 90 mm, used as the frame centerline side [guess]
    frame_edges_per_side: int = 22   # 90 mm / 22 = 4.09 mm ~ paper's dl = 4 mm; gives 128 nodes total
    frame_height: float = 0.035      # [paper] H_f = 35 mm: out-of-plane height = drag width of the body
    frame_wall: float = 0.0275       # [paper] (L_o - L_i) / 2 = 27.5 mm wall width
    body_mass: float = 0.241         # [guess] neutrally buoyant: rho_water * (L_o^2 - L_i^2) * H_f
    frame_youngs: float = 50e6       # [guess] effectively rigid compared with the limbs

    # --- Limbs (SMA actuators) ---
    limb_length: float = 0.040       # [inferred] 128 total nodes = 88 frame + 4 x 10 limb nodes at dl ~ 4 mm
    limb_edges: int = 10
    limb_height: float = 0.024       # [paper] H_a = 24 mm: out-of-plane height = drag width of the limb
    limb_thickness: float = 0.0025   # [guess] in-plane thickness of the actuator strip
    limb_base_density: float = 1100  # [guess] silicone
    youngs_cold: float = 3.0e6       # [paper] E_0
    youngs_hot: float = 8.04e6       # [paper] E_max
    stiffen_when_hot: bool = True    # [guess form] E follows the actuation level (see FrogActuator)
    # All four limbs leave their corner heading backward (-y) and curl outward. [inferred]
    # Their sim swims forward even at phase 0 (front and rear fire together), so front and rear
    # limbs can't be mirror images of each other; opening must sweep every limb backward.
    limb_heading_deg: float = -90.0

    # --- Fluid ---
    rho_fluid: float = 1000.0        # water
    c_limb_normal: float = 1.0       # [paper] C_ln
    c_limb_tangent: float = 0.01     # [paper] C_lt
    c_body_normal: float = 0.8       # [paper] C_bn (their fitted value)
    c_body_tangent: float = 0.008    # [paper] C_bt = C_bn / 100
    c_added_mass: float = 1.0        # [paper] C_a

    # --- Numerics ---
    dt: float = 1e-3                 # [repo] their outputs are sampled every 1 ms
    tol: float = 1e-6
    max_iter: int = 30
    sparse: bool = True              # sparse assembly + SciPy solver (much faster than dense here)

    @property
    def limb_density(self):
        # [guess] interface layer ~1/3 of the limb; EGaIn (6250 kg/m^3) replaces silicone (~1070)
        return self.limb_base_density + self.egain / 100 * (6250 - 1070) / 3


LIMBS = {  # name -> corner (sign of x, sign of y)
    'front_left': (-1, +1), 'front_right': (+1, +1),
    'rear_left': (-1, -1), 'rear_right': (+1, -1),
}
FRONT = ('front_left', 'front_right')


# ----------------------------------------------------------------------------------------------
# Reference data from the paper's repository
# ----------------------------------------------------------------------------------------------

_REF = None


def reference():
    global _REF
    if _REF is None:
        with open(REFERENCE_FILE) as fh:
            _REF = json.load(fh)
    return _REF


def curvature_profile(egain, freq):
    """Measured mean limb curvature over one actuation cycle: (t [s], kappa [1/m])."""
    prof = reference()['profiles'][str(int(egain))][f'{freq:.1f}']
    return np.array(prof['t']), np.array(prof['kappa'])


def their_speed(t, y, freq):
    """The paper's speed metric (find_max_speed.m): mean dy/dt over cycles 5-8, mm/s. t in s, y in m."""
    iS, iF = round(5 / freq * 1000) - 1, round(8 / freq * 1000) - 1
    return float((y[iF] - y[iS]) / (t[iF] - t[iS]) * 1000)


# ----------------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------------

@dataclasses.dataclass
class FrogGeometry:
    nodes: np.ndarray        # (n_nodes, 3)
    edges: np.ndarray        # (n_edges, 2)
    frame_nodes: np.ndarray  # node indices of the square frame (a closed loop, in order)
    corner_nodes: dict       # limb name -> corner node (shared by frame and limb)
    limb_nodes: dict         # limb name -> node indices along the limb, corner first
    frame_edges: np.ndarray
    limb_edges: dict


def build_frog_geometry(p: FrogParams, rest_curvature: float) -> FrogGeometry:
    half = p.frame_side / 2
    n = p.frame_edges_per_side

    # Frame: walk the square counter-clockwise from the rear-left corner; 4n nodes, 4n edges.
    corners_ccw = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    frame_xy = np.array([a + s * (b - a)
                         for a, b in zip(corners_ccw, np.roll(corners_ccw, -1, axis=0))
                         for s in np.arange(n) / n])
    n_frame = len(frame_xy)
    nodes = [np.column_stack([frame_xy, np.zeros(n_frame)])]
    edges = [np.column_stack([np.arange(n_frame), (np.arange(n_frame) + 1) % n_frame])]

    # Limbs: circular arcs of curvature kappa, leaving the corner along `heading`, curling outward.
    kappa, ds = rest_curvature, p.limb_length / p.limb_edges
    heading0 = np.deg2rad(p.limb_heading_deg)
    corner_nodes, limb_nodes, limb_edges = {}, {}, {}
    node_count = edge_count = n_frame
    for name, (sx, sy) in LIMBS.items():
        c = int(np.argmin(np.linalg.norm(frame_xy - [sx * half, sy * half], axis=1)))
        # Turn toward the outside of the body: the turn direction depends on the side (sx).
        turn = -sx * np.sign(np.sin(heading0))
        theta = heading0 + turn * kappa * ds * np.arange(1, p.limb_edges + 1)
        x = frame_xy[c, 0] + (np.sin(theta) - np.sin(heading0)) / (turn * kappa)
        y = frame_xy[c, 1] - (np.cos(theta) - np.cos(heading0)) / (turn * kappa)
        new = np.arange(node_count, node_count + p.limb_edges)
        chain = np.concatenate([[c], new])
        nodes.append(np.column_stack([x, y, np.zeros_like(x)]))
        edges.append(np.column_stack([chain[:-1], chain[1:]]))   # edges point corner -> tip
        corner_nodes[name], limb_nodes[name] = c, chain
        limb_edges[name] = np.arange(edge_count, edge_count + p.limb_edges)
        node_count += p.limb_edges
        edge_count += p.limb_edges

    return FrogGeometry(np.vstack(nodes), np.vstack(edges).astype(np.int64), np.arange(n_frame),
                        corner_nodes, limb_nodes, np.arange(n_frame), limb_edges)


def to_dismech_geometry(fg: FrogGeometry) -> dismech.Geometry:
    """
    Hand the arrays to PyDiSMech, then fix up the corner joints.

    PyDiSMech creates a bend spring for every pair of edges meeting at a node, so each corner
    (2 frame edges + 1 limb edge) gets 3: frame-frame plus two frame-limb springs. The paper uses
    two bending energies per joint, so we drop the frame-limb spring with the sharpest rest angle.
    (For limbs that leave the corner parallel to a frame side, that spring sits at exactly 180
    degrees, where the bend formula 2 tan(phi/2) is singular, so it must go regardless.)
    """
    geo = dismech.Geometry(fg.nodes, fg.edges, np.empty((0, 3), dtype=np.int64), plot_from_txt=False)
    springs = geo.bend_twist_springs.astype(int)          # rows: [n0, e0, n1, e1, n2]
    n_frame = len(fg.frame_nodes)
    drop = []
    for c in fg.corner_nodes.values():
        at_corner = np.where((springs[:, 2] == c) & ((springs[:, 0] >= n_frame) | (springs[:, 4] >= n_frame)))[0]
        p0, p1, p2 = (fg.nodes[springs[at_corner, k]] for k in (0, 2, 4))
        te = (p1 - p0) / np.linalg.norm(p1 - p0, axis=1, keepdims=True)
        tf = (p2 - p1) / np.linalg.norm(p2 - p1, axis=1, keepdims=True)
        turning_angle = np.arccos(np.clip(np.sum(te * tf, axis=1), -1, 1))
        drop.append(at_corner[np.argmax(turning_angle)])
    geo.remove_bend_twist_springs(drop)
    return geo


# ----------------------------------------------------------------------------------------------
# Actuation
# ----------------------------------------------------------------------------------------------

class FrogActuator:
    """
    Drives the limbs from the measured curvature profile kappa(t) of one actuation cycle.

    Front limbs fire at t = 0, T, 2T, ...; rear limbs at phase*T later. For each limb bend spring
    the natural strain is the discrete curvature of an arc with curvature kappa(t):
        nat(t) = sign * 2 tan(kappa(t) * l / 2)        (l = spring's Voronoi length)
    scaled so that nat(0) is exactly the rest shape. If stiffen_when_hot, the limb Young's modulus
    goes from E0 to Emax in proportion to how far the limb has opened, s = (k0 - kappa)/(k0 - kmin).
    """

    def __init__(self, sim: 'FrogSim'):
        p = sim.params
        self.sim = sim
        self.period = 1.0 / p.freq
        self.t_prof, self.k_prof = curvature_profile(p.egain, p.freq)
        self.k0, self.kmin = self.k_prof[0], self.k_prof.min()

        bs = sim.robot.bend_springs
        self.nat_rest = bs.nat_strain[:, 1].copy()     # in-plane bending lives in strain column 1
        self.l = bs.voronoi_len
        self.ei_per_E = bs.EI[:, 1].copy() / p.youngs_cold
        self.ea_per_E = sim.robot.stretch_springs.EA.copy() / p.youngs_cold
        self.front = np.isin(sim.limb_of_bend_spring, FRONT)

    def kappa_at(self, t, delay):
        if t < delay:
            return self.k0
        return np.interp((t - delay) % self.period, self.t_prof, self.k_prof)

    def apply(self, t):
        p, sim = self.sim.params, self.sim
        k_front = self.kappa_at(t, 0.0)
        k_rear = self.kappa_at(t, p.phase * self.period)
        bs, ss = sim.robot.bend_springs, sim.robot.stretch_springs

        idx = sim.actuated_bend
        kappa = np.where(self.front[idx], k_front, k_rear)
        g = lambda k: 2 * np.tan(k * self.l[idx] / 2)
        bs.nat_strain[idx, 1] = self.nat_rest[idx] * g(kappa) / g(self.k0)

        if p.stiffen_when_hot:
            def youngs(k):
                s = np.clip((self.k0 - k) / (self.k0 - self.kmin), 0, 1)
                return p.youngs_cold + (p.youngs_hot - p.youngs_cold) * s
            E_front, E_rear = youngs(k_front), youngs(k_rear)
            b, s = sim.limb_bend, sim.limb_stretch
            bs.EI[b] = (self.ei_per_E[b] * np.where(self.front[b], E_front, E_rear))[:, None]
            ss.EA[s] = self.ea_per_E[s] * np.where(sim.stretch_is_front[s], E_front, E_rear)
        return k_front, k_rear


# ----------------------------------------------------------------------------------------------
# Full simulation object
# ----------------------------------------------------------------------------------------------

class FrogSim:
    def __init__(self, params: FrogParams = None):
        p = self.params = params or FrogParams()
        t_prof, k_prof = curvature_profile(p.egain, p.freq)
        self.fg = fg = build_frog_geometry(p, rest_curvature=k_prof[0])
        n_frame = len(fg.frame_nodes)
        n_nodes = len(fg.nodes)

        # Which limb each node belongs to ('' for frame nodes, including corners)
        limb_of_node = np.array([''] * n_nodes, dtype=object)
        for name, chain in fg.limb_nodes.items():
            limb_of_node[chain[1:]] = name
        is_limb_node = limb_of_node != ''

        # ---- Section properties ----
        A_limb = p.limb_height * p.limb_thickness
        I_limb = p.limb_height * p.limb_thickness ** 3 / 12
        A_frame = p.frame_height * p.frame_wall
        I_frame = p.frame_height * p.frame_wall ** 3 / 12

        geom = dismech.GeomParams(rod_r0=p.limb_thickness / 2, shell_h=0, axs=A_limb,
                                  ixs1=I_limb, ixs2=I_limb, jxs=2 * I_limb)
        material = dismech.Material(density=p.limb_density, youngs_rod=p.youngs_cold, youngs_shell=0,
                                    poisson_rod=0.5, poisson_shell=0)
        sim = dismech.SimParams(static_sim=False, two_d_sim=True, use_mid_edge=False, use_line_search=False,
                                log_data=False, log_step=1, show_floor=False, dt=p.dt, max_iter=p.max_iter,
                                total_time=1.0, plot_step=1, tol=p.tol, ftol=1e-10, dtol=1e-6,
                                solver='scipy' if p.sparse else 'np', sparse=p.sparse)

        # ---- Drag (Eq. 1-2): per-node coefficients, width, length, tangent ----
        dl_frame = p.frame_side / p.frame_edges_per_side
        dl_limb = p.limb_length / p.limb_edges
        ct = np.where(is_limb_node, p.c_limb_tangent, p.c_body_tangent)
        cn = np.where(is_limb_node, p.c_limb_normal, p.c_body_normal)
        width = np.where(is_limb_node, p.limb_height, p.frame_height)
        length = np.where(is_limb_node, dl_limb, dl_frame)
        tangent_nodes = np.zeros((n_nodes, 2), dtype=np.int64)
        loop = fg.frame_nodes
        tangent_nodes[loop] = np.column_stack([np.roll(loop, 1), np.roll(loop, -1)])
        for chain in fg.limb_nodes.values():
            prev, nxt = chain[:-2], chain[2:]
            tangent_nodes[chain[1:-1]] = np.column_stack([prev, nxt])
            tangent_nodes[chain[-1]] = [chain[-2], chain[-1]]
            length[chain[-1]] = dl_limb / 2                  # the tip owns half an edge

        env = dismech.Environment()
        env.add_force('anisotropicDrag', rho=p.rho_fluid, ct=ct, cn=cn, width=width, length=length,
                      tangent_nodes=tangent_nodes)

        robot = dismech.SoftRobot(geom, material, to_dismech_geometry(fg), sim, env)
        robot = robot.fix_nodes(np.arange(n_nodes), axis=2)     # planar motion
        self.robot = robot

        # ---- Classify springs ----
        bs, ss = robot.bend_springs, robot.stretch_springs
        center = bs.nodes_ind[:, 1]
        touches_limb = is_limb_node[bs.nodes_ind].any(axis=1)

        def limb_of(nodes_row):
            names = [limb_of_node[n] for n in nodes_row if is_limb_node[n]]
            return names[0] if names else ''
        self.limb_of_bend_spring = np.array([limb_of(row) for row in bs.nodes_ind], dtype=object)
        self.actuated_bend = np.where(is_limb_node[center])[0]          # springs centered on a limb node
        self.limb_bend = np.where(touches_limb)[0]                      # + limb-root springs at corners
        frame_bend = np.where(~touches_limb)[0]
        limb_edge_mask = is_limb_node[ss.nodes_ind].any(axis=1)
        self.limb_stretch = np.where(limb_edge_mask)[0]
        frame_stretch = np.where(~limb_edge_mask)[0]
        self.stretch_is_front = np.array([limb_of(row) in FRONT for row in ss.nodes_ind])

        # ---- Stiffness per region (bend K = EI / voronoi_len, stretch K = EA / ref_len) ----
        bs.EI[self.limb_bend] = p.youngs_cold * I_limb
        bs.EI[frame_bend] = p.frame_youngs * I_frame
        ss.EA[self.limb_stretch] = p.youngs_cold * A_limb
        ss.EA[frame_stretch] = p.frame_youngs * A_frame

        # ---- Mass with added mass (Eq. 4): m_i = (rho_r + Ca rho_f) V_i ----
        V_body = p.frame_side ** 2 * p.frame_height - (p.frame_side - 2 * p.frame_wall) ** 2 * p.frame_height
        rho_body = p.body_mass / V_body
        V_node = np.zeros(n_nodes)
        rho_node = np.zeros(n_nodes)
        V_node[loop] = V_body / n_frame
        rho_node[loop] = rho_body
        for chain in fg.limb_nodes.values():
            V_node[chain[1:]] = A_limb * dl_limb
            V_node[chain[-1]] = A_limb * dl_limb / 2
            rho_node[chain[1:]] = p.limb_density
            V_node[chain[0]] += A_limb * dl_limb / 2          # corner carries half of the first limb edge
        self.node_mass = rho_node * V_node                     # structural mass (used for the COM)
        m_eff = (rho_node + p.c_added_mass * p.rho_fluid) * V_node
        robot.mass_matrix[:3 * n_nodes] = np.repeat(m_eff, 3)

        self.stepper = dismech.ImplicitEulerTimeStepper(robot)   # sets nat_strain from the rest shape
        self.bend_energy = self.stepper.elastic_energies[dismech.BEND]
        self.actuator = FrogActuator(self)

    # ---- measurements ----
    def xy(self, robot=None):
        r = robot or self.robot
        return r.state.q[:3 * r.n_nodes].reshape(-1, 3)[:, :2]

    def com(self, robot=None):
        return (self.node_mass[:, None] * self.xy(robot)).sum(0) / self.node_mass.sum()

    def limb_curvature(self, robot=None):
        """Mean |curvature| (1/m) of each limb, measured from the DER configuration."""
        r = robot or self.robot
        k = np.abs(self.bend_energy.get_strain(r.state)[:, 1] / r.bend_springs.voronoi_len)
        return {name: k[self.actuated_bend][self.limb_of_bend_spring[self.actuated_bend] == name].mean()
                for name in LIMBS}

    # ---- time integration ----
    def run(self, total_time, record_xy_every=10, progress=False):
        dt = self.params.dt
        n_steps = int(round(total_time / dt))
        out = dict(t=[0.0], com=[self.com()], kappa_front=[self.actuator.k0], kappa_rear=[self.actuator.k0],
                   kappa_cmd_front=[self.actuator.k0], kappa_cmd_rear=[self.actuator.k0],
                   t_xy=[0.0], xy=[self.xy().copy()], iters_hit_max=0)
        robot = self.robot
        for i in range(n_steps):
            t = (i + 1) * dt
            kf, kr = self.actuator.apply(t)
            robot, _ = self.stepper.step(robot, debug=False)
            self.robot = robot
            out['t'].append(t)
            out['com'].append(self.com(robot))
            kc = self.limb_curvature(robot)
            out['kappa_front'].append(np.mean([kc[n] for n in FRONT]))
            out['kappa_rear'].append(np.mean([kc[n] for n in LIMBS if n not in FRONT]))
            out['kappa_cmd_front'].append(kf)
            out['kappa_cmd_rear'].append(kr)
            if (i + 1) % record_xy_every == 0:
                out['t_xy'].append(t)
                out['xy'].append(self.xy(robot).copy())
            if progress and (i + 1) % 1000 == 0:
                print(f"  t = {t:.1f} s, COM y = {out['com'][-1][1]*1e3:.1f} mm", flush=True)
        for k in ('t', 'com', 'kappa_front', 'kappa_rear', 'kappa_cmd_front', 'kappa_cmd_rear', 't_xy', 'xy'):
            out[k] = np.array(out[k])
        return out
