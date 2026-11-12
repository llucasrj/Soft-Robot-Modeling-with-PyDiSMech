import numpy as np

import dismech
from dismech.external_forces import compute_anisotropic_drag


def _straight_rod(n=5, dl=0.01, rho=1000.0, ct=0.01, cn=1.0, width=0.02):
    nodes = np.column_stack([np.arange(n) * dl, np.zeros(n), np.zeros(n)])
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    geo = dismech.Geometry(nodes, edges, np.empty((0, 3), dtype=np.int64), plot_from_txt=False)
    sim = dismech.SimParams(static_sim=False, two_d_sim=True, use_mid_edge=False, use_line_search=False,
                            log_data=False, log_step=1, show_floor=False, dt=1e-3, max_iter=25,
                            total_time=1.0, plot_step=1, tol=1e-4, ftol=1e-4, dtol=1e-2)
    env = dismech.Environment()
    idx = np.arange(n)
    env.add_force('anisotropicDrag', rho=rho, ct=np.full(n, ct), cn=np.full(n, cn),
                  width=np.full(n, width), length=np.full(n, dl),
                  tangent_nodes=np.column_stack([np.maximum(idx - 1, 0), np.minimum(idx + 1, n - 1)]))
    robot = dismech.SoftRobot(dismech.GeomParams(rod_r0=1e-3, shell_h=0),
                              dismech.Material(1000, 1e6, 0, 0.5, 0), geo, sim, env)
    return robot, dict(rho=rho, ct=ct, cn=cn, width=width, dl=dl)


def test_drag_matches_formula_along_and_across_rod():
    robot, p = _straight_rod()
    q = robot.state.q.copy()
    A = 0.5 * p['rho'] * p['width'] * p['dl']
    for direction, coeff in (([1, 0, 0], p['ct']), ([0, 1, 0], p['cn'])):
        u = np.zeros_like(q)
        u[:3 * robot.n_nodes] = np.tile(np.array(direction) * 0.2, robot.n_nodes)
        F, _ = compute_anisotropic_drag(robot, q, u)
        expected = -A * coeff * 0.2 * 0.2 * np.array(direction)
        np.testing.assert_allclose(F[:3 * robot.n_nodes].reshape(-1, 3), np.tile(expected, (robot.n_nodes, 1)))


def test_drag_jacobian_matches_finite_difference():
    robot, _ = _straight_rod()
    rng = np.random.default_rng(0)
    q = robot.state.q.copy()
    u = np.zeros_like(q)
    u[:3 * robot.n_nodes] = rng.normal(scale=0.1, size=3 * robot.n_nodes)
    _, J = compute_anisotropic_drag(robot, q, u)

    # The Jacobian is d F / d q = (d F / d u) / dt with the tangent held fixed.
    eps, dt = 1e-7, robot.sim_params.dt
    J_fd = np.zeros_like(J)
    for j in range(3 * robot.n_nodes):
        du = np.zeros_like(u)
        du[j] = eps
        Fp, _ = compute_anisotropic_drag(robot, q, u + du)
        Fm, _ = compute_anisotropic_drag(robot, q, u - du)
        J_fd[:, j] = (Fp - Fm) / (2 * eps) / dt
    np.testing.assert_allclose(J, J_fd, rtol=1e-5, atol=1e-9)
