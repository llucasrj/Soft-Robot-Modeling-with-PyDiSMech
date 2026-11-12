import typing

import numpy as np
import scipy.sparse as sp

from ..soft_robot import SoftRobot


def compute_anisotropic_drag(robot: SoftRobot, q: np.ndarray, u: np.ndarray) -> typing.Tuple[np.ndarray, typing.Union[np.ndarray, sp.csr_matrix]]:
    """
    Quadratic, anisotropic fluid drag on rod nodes (Huang et al. 2022, Eq. 1-2):

        f_i = -1/2 * rho * D_i * dl_i * |v_i| * (ct_i * (v_i . t_i) t_i + cn_i * (v_i - (v_i . t_i) t_i))

    Coefficients are per node, so different parts of a robot (e.g. limbs vs body) can have
    different drag. The tangent t_i is taken from the two nodes in env.drag_tangent_nodes[i]
    and is treated as constant within a Newton step (its derivative is left out of the Jacobian).

    Environment fields (set via env.add_force('anisotropicDrag', ...)):
        rho                 fluid density
        drag_ct, drag_cn    (n_nodes,) tangential / normal drag coefficients
        drag_width          (n_nodes,) projected width D of the rod at each node
        drag_length         (n_nodes,) length dl associated with each node
        drag_tangent_nodes  (n_nodes, 2) node pair (a, b) whose difference gives the tangent
    """
    env = robot.env
    dt = robot.sim_params.dt
    n_nodes = robot.n_nodes
    n_dof = robot.n_dof

    X = q[:3 * n_nodes].reshape(n_nodes, 3)
    V = u[:3 * n_nodes].reshape(n_nodes, 3)

    a, b = env.drag_tangent_nodes[:, 0], env.drag_tangent_nodes[:, 1]
    T = X[b] - X[a]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)

    # Drag matrix C = cn * I + (ct - cn) * t t^T, scaled by 1/2 rho D dl
    A = 0.5 * env.rho * env.drag_width * env.drag_length
    C = env.drag_cn[:, None, None] * np.eye(3) + \
        (env.drag_ct - env.drag_cn)[:, None, None] * T[:, :, None] * T[:, None, :]
    C *= A[:, None, None]

    speed = np.linalg.norm(V, axis=1)
    CV = np.einsum('nij,nj->ni', C, V)
    F_nodes = -speed[:, None] * CV

    # d f / d v = -( |v| C + (C v) v^T / |v| );  d v / d q = 1 / dt  (backward Euler velocity)
    safe_speed = np.where(speed > 1e-12, speed, 1.0)
    outer = CV[:, :, None] * (V / safe_speed[:, None])[:, None, :]
    outer[speed <= 1e-12] = 0.0
    J_blocks = -(speed[:, None, None] * C + outer) / dt

    F = np.zeros(n_dof)
    F[:3 * n_nodes] = F_nodes.ravel()

    rows = (3 * np.arange(n_nodes))[:, None, None] + np.arange(3)[None, :, None]
    cols = (3 * np.arange(n_nodes))[:, None, None] + np.arange(3)[None, None, :]
    rows, cols = np.broadcast_arrays(rows, cols)
    if robot.sim_params.sparse:
        J = sp.coo_matrix((J_blocks.ravel(), (rows.ravel(), cols.ravel())), shape=(n_dof, n_dof)).tocsr()
    else:
        J = np.zeros((n_dof, n_dof))
        J[rows.ravel(), cols.ravel()] = J_blocks.ravel()
    return F, J
