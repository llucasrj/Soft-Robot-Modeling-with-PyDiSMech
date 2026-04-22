import numpy as np


def add_point_forces(robot, node_indices, force_vectors):
    fpt = np.zeros_like(robot.state.q)

    # Ensure numpy array
    force_vectors = np.asarray(force_vectors)

    # dof_indices = robot.map_node_to_dof(node_indices)  # original: shape (N,3), breaks multi-node assignment
    dof_indices = robot.map_node_to_dof(node_indices).reshape(-1) # flattening -RYAN 
    fpt[dof_indices] = force_vectors.reshape(-1)
    return fpt
