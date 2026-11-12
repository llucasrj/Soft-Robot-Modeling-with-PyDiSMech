"""
Open a validation run in the viewer (interactive HTML + GIF).

    python view_case.py 10 0.4 0.5          # EGaIn %, frequency Hz, phase
    python view_case.py 30 1.3 0.5 --last 3  # only the last 3 seconds

Needs outputs/validation/<case>_xy.npz from validate.py.
"""
import argparse
import json
import os

import numpy as np

from frog_model import FrogParams, build_frog_geometry, curvature_profile
from frog_viewer import save_trajectory, make_viewer
from validate import OUT, case_name

ap = argparse.ArgumentParser()
ap.add_argument('egain', type=int)
ap.add_argument('freq', type=float)
ap.add_argument('phase', type=float)
ap.add_argument('--last', type=float, default=None, help='only show the last N seconds')
a = ap.parse_args()

name = case_name(a.egain, a.freq, a.phase)
d = np.load(os.path.join(OUT, name + '_xy.npz'))
with open(os.path.join(OUT, name + '.json')) as fh:
    res = json.load(fh)
t, xy = d['t_xy'], d['xy']
if a.last:
    keep = t >= t[-1] - a.last
    t, xy = t[keep], xy[keep]

_, k = curvature_profile(a.egain, a.freq)
fg = build_frog_geometry(FrogParams(egain=a.egain, freq=a.freq, phase=a.phase), rest_curvature=k[0])
# Limb curvature series (stored every 5 ms) resampled onto the shape frames
tk = np.arange(len(res['kappa_front'])) * 0.005
series = dict(kappa_front=np.interp(t, tk, res['kappa_front']),
              kappa_cmd_front=np.interp(t, tk, res['kappa_cmd_front']))
traj = save_trajectory(os.path.join(OUT, name + '_view.npz'), fg, t, xy, **series)
make_viewer(traj, os.path.join(OUT, name),
            title=f"{a.egain}% EGaIn, {a.freq} Hz, phase {a.phase}: {res['speed']:.1f} mm/s")
