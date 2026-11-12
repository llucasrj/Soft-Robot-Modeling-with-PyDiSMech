"""
Recreate the data figures of Huang et al., "Design and Closed-Loop Motion Planning of an Untethered
Swimming Soft Robot Using 2D Discrete Elastic Rods Simulations", Adv. Intell. Syst. 2022, 4, 2200163
(CC-BY 4.0), from the authors' released data (github.com/softmachineslab/frog).

    python paper_figures.py            -> figures/paper_fig3b.png, paper_fig4.png, paper_fig5.png

Data sources (not committed: the authors' repo has no license, so it is fetched locally instead):
  paper_data/frog_reference.json  (built by extract_paper_data.py)  -> Fig. 3b, Fig. 4a-f
  paper_data/fig5/*.csv           (control_data/data/ in their repo, downloaded automatically) -> Fig. 5d-g
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
SIM_STYLE = dict(marker='o', mfc='none', mec='#f28e2b', ls='none', ms=7, mew=1.5, label='sim')
EXP_STYLE = dict(marker='*', color='#2ca25f', ls='none', ms=9, label='exp')


def _load_ref():
    with open(os.path.join(HERE, 'paper_data', 'frog_reference.json')) as fh:
        return json.load(fh)


def _pick(rows, egain, freq=None, phase=None, sweep=None):
    """(x, speed) rows matching the filter; x is phase if freq is fixed, else frequency."""
    out = []
    for r in rows:
        if round(r['egain']) != egain:
            continue
        if freq is not None and abs(r['freq'] - freq) > 1e-6:
            continue
        if phase is not None and abs(r['phase'] - phase) > 1e-6:
            continue
        if sweep is not None and r.get('sweep') != sweep:
            continue
        out.append((r['phase'] if freq is not None and phase is None else r['freq'], r['speed']))
    out.sort()
    return np.array(out).reshape(-1, 2)


# ---------------------------------------------------------------- Figure 3b
def fig3b(ref):
    """Curvature change (max - min of the measured mean limb curvature over one cycle) vs frequency.
    Released data holds only the mean over the 4 samples, so the paper's error bars cannot be redrawn."""
    colors = {'0': '#f28e2b', '10': '#2ca25f', '20': '#d62728', '30': '#1f77b4', '40': '#17becf', '50': '#7b3fa0'}
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for eg in sorted(ref['profiles'], key=int):
        pts = sorted((float(f), np.ptp(p['kappa'])) for f, p in ref['profiles'][eg].items())
        pts = [(f, dk) for f, dk in pts if dk >= 70]          # paper: changes < 70 1/m not included
        if pts:
            f, dk = np.array(pts).T
            ax.plot(f, dk, '-o', ms=4, color=colors.get(eg, 'k'), label=f'{eg}%')
    ax.set(xlim=(0, 3), ylim=(0, 140), xlabel='Frequency [Hz]', ylabel='Curvature Change [1/m]')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Fig. 3b (means only; no error bars in released data)', fontsize=9)
    fig.tight_layout()
    return _save(fig, 'paper_fig3b.png')


# ---------------------------------------------------------------- Figure 4
def _scatter(ax, sim, exp, xlabel, xlim, legend_loc):
    ax.plot(*sim.T, **SIM_STYLE)
    ax.plot(*exp.T, **EXP_STYLE)
    ax.set(xlim=xlim, ylim=(0, 60), xlabel=xlabel, ylabel='Speed [mm/s]')
    ax.legend(loc=legend_loc, fontsize=8)


def _bars(ax, labels, values, colors, ylim):
    for i, (lab, v, c) in enumerate(zip(labels, values, colors)):
        exp = 'exp' in lab
        ax.bar(i, v, width=0.65, color=c, hatch='///' if exp else None,
               edgecolor='k', linestyle='--' if exp else '-', linewidth=1)
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set(ylim=ylim, ylabel='Speed [mm/s]')


def fig4(ref):
    sim, exp = ref['sim'], ref['exp']
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    a, b, c, d, e, f = axs.ravel()

    _scatter(a, _pick(sim, 10, phase=0.5), _pick([r for r in exp if 'sweep' not in r], 10, phase=0.5),
             'Frequency [Hz]', (0.4, 1.2), 'lower right')
    _scatter(b, _pick(sim, 10, freq=0.4), _pick(exp, 10, freq=0.4, sweep='phase'),
             'Phase offset', (0, 0.9), 'upper right')
    _scatter(c, _pick(sim, 10, freq=0.8), _pick(exp, 10, freq=0.8, sweep='phase'),
             'Phase offset', (0, 0.9), 'upper right')
    _scatter(d, _pick(sim, 40, phase=0.5), _pick(exp, 40, phase=0.5),
             'Frequency [Hz]', (1.5, 1.9), 'upper right')

    def speed(rows, eg, fr):
        m = _pick([r for r in rows if 'sweep' not in r], eg, freq=fr, phase=0.5)
        return float(m[0, 1])

    _bars(e, ['1.3Hz, exp', '1.3Hz, sim', '1.4Hz, exp', '1.4Hz, sim'],
          [speed(exp, 30, 1.3), speed(sim, 30, 1.3), speed(exp, 30, 1.4), speed(sim, 30, 1.4)],
          ['#f28e2b', '#f28e2b', '#2ca25f', '#2ca25f'], (0, 70))
    _bars(f, ['0.4Hz, exp', '0.4Hz, sim', '0.5Hz, exp', '0.5Hz, sim'],
          [speed(exp, 10, 0.4), speed(sim, 10, 0.4), speed(exp, 10, 0.5), speed(sim, 10, 0.5)],
          ['#e31a1c', '#e31a1c', '#1f9bd7', '#1f9bd7'], (0, 40))

    for ax, lab in zip(axs.ravel(), 'abcdef'):
        ax.text(-0.16, 1.02, f'({lab})', transform=ax.transAxes, fontsize=12)
    fig.tight_layout()
    return _save(fig, 'paper_fig4.png')


# ---------------------------------------------------------------- Figure 5
# Trials and pairing exactly as in the authors' process_frog_2.m
FIG5_RUNS = [('Line 1', 'line1', 'linep1', '#e41a1c'), ('Line 2', 'line2', 'linep2', '#377eb8'),
             ('Sin 1', 'sin1', 'sinp1', '#4daf4a'), ('Sin 2', 'sin4', 'sinp4', '#ff7f00'),
             ('Ellipse 1', 'ellipse5', 'ellipsep5', '#e6d64a'), ('Ellipse 2', 'ellipse4', 'ellipsep4', '#a65628')]


FIG5_URL = 'https://raw.githubusercontent.com/softmachineslab/frog/HEAD/control_data/data/{}.csv'


def _fetch_fig5(d5):
    """Download the six path-following logs + their target paths from the authors' repo if missing."""
    from urllib.request import urlopen
    os.makedirs(d5, exist_ok=True)
    for _, run, path, _ in FIG5_RUNS:
        for name in (run, path):
            dest = os.path.join(d5, name + '.csv')
            if not os.path.exists(dest):
                print('downloading', name + '.csv')
                with urlopen(FIG5_URL.format(name), timeout=30) as r, open(dest, 'wb') as fh:
                    fh.write(r.read())


def _read_log(path):
    """cv_datalogger csv: 2 header lines, then t_ms, x, y, theta, xdot, ydot, thetadot, costs..., action."""
    rows = []
    with open(path) as fh:
        for line in list(fh)[2:]:
            parts = line.strip().split(',')
            try:
                rows.append([float(v) for v in parts[:7]])
            except ValueError:
                continue
    a = np.array(rows)
    return (a[:, 0] - a[0, 0]) / 1000, a[:, 1:3]          # time [s], xy [m]


def fig5():
    d5 = os.path.join(HERE, 'paper_data', 'fig5')
    _fetch_fig5(d5)
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5])
    path_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    gax = fig.add_subplot(gs[0, 3])

    for k, (label, run, path, color) in enumerate(FIG5_RUNS):
        t, xy = _read_log(os.path.join(d5, run + '.csv'))
        p = np.loadtxt(os.path.join(d5, path + '.csv'), delimiter=',')[:, :2]
        ax = path_axes[k // 2]
        ax.plot(p[:, 0] * 100, p[:, 1] * 100, '--', color=color, lw=1.5)
        ax.plot(xy[:, 0] * 100, xy[:, 1] * 100, '-', color=color, lw=1.5)
        dist = np.min(np.linalg.norm(xy[:, None, :] - p[None, :, :], axis=2), axis=1)
        gax.plot(t, dist * 100, color=color, lw=1.5, label=label)

    for ax, lab in zip(path_axes, 'def'):
        ax.set_aspect('equal', adjustable='datalim')
        ax.set(xlabel='x [cm]', ylabel='y [cm]')
        ax.set_title(f'({lab})', loc='left')
    gax.set(xlabel='Time [s]', ylabel='Distance from path [cm]', xlim=(0, None), ylim=(0, None))
    gax.set_title('(g)', loc='left')
    gax.legend(fontsize=8)
    fig.text(0.01, 0.01, 'dashed = desired path, solid = robot (camera-tracked)', fontsize=8)
    fig.tight_layout()
    return _save(fig, 'paper_fig5.png')


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)
    return path


if __name__ == '__main__':
    if os.path.exists(os.path.join(HERE, 'paper_data', 'frog_reference.json')):
        ref = _load_ref()
        fig3b(ref)
        fig4(ref)
    else:
        print('skipping Fig. 3b/4: paper_data/frog_reference.json missing. Build it with:\n'
              '    git -c core.longpaths=true clone https://github.com/softmachineslab/frog.git\n'
              '    python extract_paper_data.py <path-to-frog-repo>')
    fig5()
