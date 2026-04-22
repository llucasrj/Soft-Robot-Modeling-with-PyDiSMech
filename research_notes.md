# Research Notes

## Bugs Found

### `add_point_forces` shape mismatch with multiple nodes
**File:** `src/dismech/external_forces/constant_point_force.py:12`

`map_node_to_dof` returns shape `(N, 3)` for N nodes (each node maps to 3 DOFs).
When indexing `fpt[dof_indices]`, numpy preserves that `(N, 3)` shape, so assigning
`force_vectors.reshape(-1)` (shape `(3N,)`) fails with a shape mismatch.

**Fix:** flatten `dof_indices` before indexing:
```python
dof_indices = robot.map_node_to_dof(node_indices).reshape(-1)
fpt[dof_indices] = force_vectors.reshape(-1)
```
This works for both single and multiple nodes.

---

### Gravity applied twice in time stepper — causes divergence
**File:** `src/dismech/time_steppers/time_stepper.py:249` and `:287`

`compute_gravity_forces` is called twice in `_compute_forces_and_jacobian`, doubling the gravity force.
This causes the beam to experience 2× the intended load, which overwhelms the Newton solver and causes divergence.

**Fix:** Remove the second gravity block (lines 287-288):
```python
# delete these two lines:
if "gravity" in robot.env.ext_force_list:
    forces -= compute_gravity_forces(robot)
```

---

### `before_step` passed wrong time value
**File:** `src/dismech/time_steppers/time_stepper.py:88`

`before_step(robot, i * dt)` is called with `i=0` on the first step, so at `t=dt` the callback
receives `t=0`. Should pass `t` (the actual current time) instead of `i * dt`.
This caused gravity ramping in static sims to start at zero and never reach full magnitude.


Increasing Youngs to 200e9 and density to 7800 made simulation run with two pointforces.

---

### Y-direction point force produces zero displacement (under investigation)
**File:** `src/dismech/external_forces/constant_point_force.py`

Applying a Y-direction force at the tip (node 50) results in exactly `0.0` Y displacement.
X displacement (arc-length shortening due to Z bending) is physically correct.
Added debug print to trace `dof_indices` and `fpt` values at runtime to determine if
the force is being stored/applied correctly.
**Hypothesis:** memory/storage issue in how `point_force_vectors` is passed from `env` to `add_point_forces`.

### 'RYAN NOTE' ###
When doing 2 point forces if u want to do two forces on the same node only do one node: 
env.add_force('pointForces',
              point_force_node_indices=[50],
              point_force_vectors=[[0, 5, 10]])

When doing 2 point forces on 2 different nodes call each one seperately: 
env.add_force('pointForces',
              point_force_node_indices=[25, 50],
              point_force_vectors=[[0, 5, 0], [0, 0, 10]])




