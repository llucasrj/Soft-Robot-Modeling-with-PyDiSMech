"""
Validation against the paper's Fig. 4: our PyDiSMech frog vs. their DER sim vs. experiment.

    python validate.py                 # run every case set (parallel), then plot
    python validate.py fig4a fig4d     # only some sets
    python validate.py --plot          # just re-plot cached results
    python validate.py --workers 8     # limit parallel processes

Each case is simulated to t = 8/f and scored with the paper's metric (mean dy/dt over cycles 5-8).
Results are cached in outputs/validation/<case>.json; delete a file to re-run that case.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')        # one core per simulation process
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs', 'validation')

# Case sets mirroring the paper's Fig. 4 panels: (egain, freq, phase)
CASE_SETS = {
    'fig4a': [(10, f, 0.5) for f in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)],     # speed vs frequency, 10%
    'fig4b': [(10, 0.4, ph) for ph in np.round(np.arange(0, 1, 0.1), 1)],              # phase sweep, 0.4 Hz
    'fig4c': [(10, 0.8, ph) for ph in np.round(np.arange(0, 1, 0.1), 1)],              # phase sweep, 0.8 Hz
    'fig4d': [(40, f, 0.5) for f in (1.5, 1.6, 1.7, 1.8, 1.9)],                        # extrapolation, 40%
    'fig4e': [(30, 1.3, 0.5), (30, 1.4, 0.5)],                                         # fastest design
    'fig4f': [(10, 0.4, 0.5), (10, 0.5, 0.5)],                                         # most efficient design
}
TITLES = {'fig4a': 'Fig 4a: speed vs frequency (10% EGaIn)', 'fig4b': 'Fig 4b: phase sweep, 10%, 0.4 Hz',
          'fig4c': 'Fig 4c: phase sweep, 10%, 0.8 Hz', 'fig4d': 'Fig 4d: extrapolation, 40% EGaIn',
          'fig4e': 'Fig 4e: fastest (30% EGaIn)', 'fig4f': 'Fig 4f: most efficient (10% EGaIn)'}


def case_name(egain, freq, phase):
    return f'eg{egain:02d}_f{freq:.1f}_ph{phase:.1f}'


def run_case(case):
    """Simulate one case and cache the result. Runs in a worker process."""
    egain, freq, phase = case
    path = os.path.join(OUT, case_name(*case) + '.json')
    if os.path.exists(path):
        return path
    from frog_model import FrogParams, FrogSim, their_speed
    t0 = time.time()
    sim = FrogSim(FrogParams(egain=egain, freq=freq, phase=phase))
    out = sim.run(8 / freq + 0.01)
    t, y = out['t'], out['com'][:, 1]
    result = dict(egain=egain, freq=freq, phase=phase, speed=their_speed(t, y, freq),
                  wall_time=time.time() - t0, x_drift_mm=float(np.abs(out['com'][:, 0]).max() * 1e3),
                  t=np.round(t[9::10], 4).tolist(), y=np.round(y[9::10] * 1e3, 4).tolist(),
                  kappa_front=np.round(out['kappa_front'][::5], 2).tolist(),
                  kappa_cmd_front=np.round(out['kappa_cmd_front'][::5], 2).tolist())
    with open(path, 'w') as fh:
        json.dump(result, fh)
    # Keep the full shape history of one representative run for the viewer
    np.savez_compressed(path.replace('.json', '_xy.npz'), t_xy=out['t_xy'], xy=out['xy'])
    print(f"  done {case_name(*case)}: {result['speed']:.1f} mm/s  ({result['wall_time'] / 60:.1f} min)", flush=True)
    return path


def load_results():
    res = {}
    if os.path.isdir(OUT):
        for f in os.listdir(OUT):
            if f.endswith('.json'):
                with open(os.path.join(OUT, f)) as fh:
                    r = json.load(fh)
                res[(r['egain'], round(r['freq'], 1), round(r['phase'], 1))] = r
    return res


def reference_points():
    from frog_model import reference
    ref = reference()
    sim = {(int(r['egain']), round(r['freq'], 1), round(r['phase'], 1)): r for r in ref['sim']}
    exp = {}
    for r in ref['exp']:
        exp.setdefault((int(r['egain']), round(r['freq'], 1), round(r['phase'], 1)), r['speed'])
    return sim, exp


def plot(sets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ours = load_results()
    their_sim, their_exp = reference_points()
    rows = []

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, name in zip(axs.ravel(), CASE_SETS):
        cases = CASE_SETS[name]
        xkey = 2 if name in ('fig4b', 'fig4c') else 1          # phase sweeps vs frequency sweeps
        xs = [c[xkey] for c in cases]
        o = [ours[c]['speed'] if c in ours else None for c in cases]
        s = [their_sim[c]['speed'] if c in their_sim else None for c in cases]
        e = [their_exp.get(c) for c in cases]
        if name in ('fig4e', 'fig4f'):
            labels = [f'{c[1]} Hz' for c in cases]
            w = 0.25
            xi = np.arange(len(cases))
            ax.bar(xi - w, [v or 0 for v in e], w, label='experiment', color='tab:green')
            ax.bar(xi, [v or 0 for v in s], w, label='paper DER sim', color='tab:orange')
            ax.bar(xi + w, [v or 0 for v in o], w, label='PyDiSMech (ours)', color='tab:blue')
            ax.set_xticks(xi, labels)
        else:
            ax.plot(xs, e, '*', ms=11, color='tab:green', label='experiment')
            ax.plot(xs, s, 'o', mfc='none', ms=9, color='tab:orange', label='paper DER sim')
            ax.plot(xs, [np.nan if v is None else v for v in o], 's-', ms=6, color='tab:blue', label='PyDiSMech (ours)')
            ax.set_xlabel('phase offset' if xkey == 2 else 'frequency [Hz]')
        ax.set_ylabel('speed [mm/s]')
        ax.set_ylim(0, 70)
        ax.set_title(TITLES[name], fontsize=10)
        ax.legend(fontsize=8)
        if name in sets:
            for c, vo, vs, ve in zip(cases, o, s, e):
                rows.append((name, c, vo, vs, ve))
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    fig.savefig(os.path.join(HERE, 'figures', 'validation_fig4.png'), dpi=120)

    # Trajectory comparison: COM y(t) for the fig4a cases
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(CASE_SETS['fig4a'])))
    for c, col in zip(CASE_SETS['fig4a'], colors):
        if c in ours:
            ax2.plot(ours[c]['t'], ours[c]['y'], '-', color=col, label=f'{c[1]} Hz ours')
        if c in their_sim:
            ax2.plot(their_sim[c]['t'], their_sim[c]['y'], '--', color=col, lw=1)
    ax2.set_xlim(0, 20); ax2.set_xlabel('t [s]'); ax2.set_ylabel('COM y [mm]')
    ax2.set_title('Forward displacement, 10% EGaIn: ours (solid) vs paper DER sim (dashed)')
    ax2.legend(fontsize=7, ncol=2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(HERE, 'figures', 'validation_trajectories.png'), dpi=120)

    # Table
    def fmt(v):
        return '   -  ' if v is None else f'{v:6.1f}'
    print(f"\n{'set':6} {'EGaIn':>5} {'f[Hz]':>5} {'phase':>5} | {'ours':>6} {'paper':>6} {'exp':>6} | {'ours/paper':>10}")
    errs = []
    for name, (eg, f, ph), vo, vs, ve in rows:
        ratio = f'{vo / vs:10.2f}' if vo and vs else '        - '
        if vo and vs:
            errs.append(vo / vs - 1)
        print(f"{name:6} {eg:5d} {f:5.1f} {ph:5.1f} | {fmt(vo)} {fmt(vs)} {fmt(ve)} | {ratio}")
    if errs:
        print(f"\nmean |ours - paper| / paper over {len(errs)} cases: {np.mean(np.abs(errs)) * 100:.1f}%")
    print("figures: figures/validation_fig4.png, figures/validation_trajectories.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sets', nargs='*', default=list(CASE_SETS))
    ap.add_argument('--plot', action='store_true', help='only plot cached results')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    if not args.plot:
        cases = sorted({c for s in args.sets for c in CASE_SETS[s]}, key=lambda c: -8 / c[1])  # longest first
        todo = [c for c in cases if not os.path.exists(os.path.join(OUT, case_name(*c) + '.json'))]
        os.makedirs(OUT, exist_ok=True)
        print(f"{len(cases)} cases, {len(todo)} to run on {min(args.workers, max(1, len(todo)))} processes", flush=True)
        if todo:
            from multiprocessing import Pool
            with Pool(min(args.workers, len(todo))) as pool:
                for _ in pool.imap_unordered(run_case, todo):
                    pass
    plot(args.sets)


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    main()
