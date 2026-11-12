"""
Display tools for frog robot simulations.

save_trajectory(...) -> writes an .npz with everything needed to replay a run
make_viewer(traj, out_base) -> writes
    out_base.html : interactive (play/pause, time slider, hover), open in any browser
    out_base.gif  : quick preview that VS Code can show directly

Run directly to re-render a saved run:
    python frog_viewer.py outputs/step1_run.npz
"""
import os
import sys

import numpy as np

LIMB_COLORS = {'front_left': '#d62728', 'front_right': '#ff7f0e',
               'rear_left': '#1f77b4', 'rear_right': '#17becf'}


def save_trajectory(path, fg, ts, xy, **series):
    """xy: (T, n_nodes, 2) node positions. series: optional extra 1-D time series (same length as ts)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = dict(ts=np.asarray(ts), xy=np.asarray(xy), frame_nodes=fg.frame_nodes,
                limb_names=np.array(list(fg.limb_nodes)))
    for name, chain in fg.limb_nodes.items():
        data[f'limb__{name}'] = chain
    for k, v in series.items():
        data[f'series__{k}'] = np.asarray(v)
    np.savez_compressed(path, **data)
    return load_trajectory(path)


def load_trajectory(path):
    d = np.load(path)
    return dict(ts=d['ts'], xy=d['xy'], frame_nodes=d['frame_nodes'],
                limbs={n: d[f'limb__{n}'] for n in d['limb_names']},
                series={k[8:]: d[k] for k in d.files if k.startswith('series__')})


def _frame_indices(n_total, max_frames):
    return np.unique(np.linspace(0, n_total - 1, min(max_frames, n_total)).round().astype(int))


def _com(xy):
    return xy.mean(axis=1)   # geometric centroid of nodes (good enough for display)


def _com_speed(ts, com):
    """COM velocity in mm/s: returns (speed magnitude, forward (+y) velocity, mean forward speed)."""
    if len(ts) < 2:
        z = np.zeros(len(ts))
        return z, z, 0.0
    v = np.gradient(com, ts, axis=0) * 1e3          # (T, 2) mm/s
    mean_fwd = (com[-1, 1] - com[0, 1]) / (ts[-1] - ts[0]) * 1e3
    return np.linalg.norm(v, axis=1), v[:, 1], mean_fwd


def _axes_range(xy, pad=0.01):
    lo, hi = xy.reshape(-1, 2).min(0) - pad, xy.reshape(-1, 2).max(0) + pad
    c, half = (lo + hi) / 2, (hi - lo).max() / 2
    return (c - half) * 1e3, (c + half) * 1e3


def make_html(traj, path, title='', max_frames=200):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ts, xy = traj['ts'], traj['xy']
    loop = np.append(traj['frame_nodes'], traj['frame_nodes'][0])
    com = _com(xy)
    idx = _frame_indices(len(ts), max_frames)
    (x0, y0), (x1, y1) = _axes_range(xy)
    has_plot = bool(traj['series'])
    speed, v_fwd, mean_fwd = _com_speed(ts, com)
    speed_title = f'COM speed (mean forward {mean_fwd:.1f} mm/s)'

    if has_plot:   # robot on the left spanning both rows; curvature top-right, speed bottom-right
        fig = make_subplots(rows=2, cols=2, column_widths=[0.55, 0.45], vertical_spacing=0.14,
                            specs=[[{'rowspan': 2}, {}], [None, {}]],
                            subplot_titles=('Robot (top view)', 'Limb curvature', speed_title))
        speed_cell = dict(row=2, col=2)
    else:
        fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                            subplot_titles=('Robot (top view)', speed_title))
        speed_cell = dict(row=1, col=2)

    def robot_traces(k):
        p = xy[k] * 1e3
        tr = [go.Scatter(x=p[loop, 0], y=p[loop, 1], mode='lines+markers', name='frame',
                         line=dict(color='#555', width=6), marker=dict(size=3, color='black'))]
        for name, chain in traj['limbs'].items():
            tr.append(go.Scatter(x=p[chain, 0], y=p[chain, 1], mode='lines+markers', name=name,
                                 line=dict(color=LIMB_COLORS.get(name, 'k'), width=4), marker=dict(size=5)))
        tr.append(go.Scatter(x=com[:k + 1, 0] * 1e3, y=com[:k + 1, 1] * 1e3, mode='lines', name='COM path',
                             line=dict(color='green', dash='dot')))
        tr.append(go.Scatter(x=[com[k, 0] * 1e3], y=[com[k, 1] * 1e3], mode='markers', name='COM',
                             marker=dict(symbol='x', size=12, color='green')))
        return tr

    n_robot = len(robot_traces(0))
    for tr in robot_traces(idx[0]):
        fig.add_trace(tr, row=1, col=1)

    if has_plot:
        for name, s in traj['series'].items():
            fig.add_trace(go.Scatter(x=ts * 1e3, y=s, mode='lines', name=name,
                                     line=dict(dash='dash' if 'cmd' in name else 'solid')), row=1, col=2)
        lo = min(s.min() for s in traj['series'].values())
        hi = max(s.max() for s in traj['series'].values())
        fig.add_trace(go.Scatter(x=[ts[idx[0]] * 1e3] * 2, y=[lo, hi], mode='lines', name='now',
                                 line=dict(color='black', width=1), showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text='t [ms]', row=1, col=2)
        fig.update_yaxes(title_text='curvature [1/m]', row=1, col=2)
    now_trace = len(fig.data) - 1

    fig.add_trace(go.Scatter(x=ts * 1e3, y=speed, mode='lines', name='|v| COM',
                             line=dict(color='green')), **speed_cell)
    fig.add_trace(go.Scatter(x=ts * 1e3, y=v_fwd, mode='lines', name='forward v (+y)',
                             line=dict(color='purple', dash='dash')), **speed_cell)
    s_lo, s_hi = min(v_fwd.min(), 0.0), speed.max()
    fig.add_trace(go.Scatter(x=[ts[idx[0]] * 1e3] * 2, y=[s_lo, s_hi], mode='lines', name='now',
                             line=dict(color='black', width=1), showlegend=False), **speed_cell)
    fig.update_xaxes(title_text='t [ms]', **speed_cell)
    fig.update_yaxes(title_text='speed [mm/s]', **speed_cell)
    speed_now_trace = len(fig.data) - 1

    frames = []
    for k in idx:
        data = robot_traces(k)
        trace_ids = list(range(n_robot))
        if has_plot:
            data.append(go.Scatter(x=[ts[k] * 1e3] * 2, y=[lo, hi]))
            trace_ids.append(now_trace)
        data.append(go.Scatter(x=[ts[k] * 1e3] * 2, y=[s_lo, s_hi]))
        trace_ids.append(speed_now_trace)
        frames.append(go.Frame(data=data, traces=trace_ids, name=f'{ts[k]*1e3:.0f}'))
    fig.frames = frames

    fig.update_xaxes(range=[x0, x1], title_text='x [mm]', row=1, col=1)
    fig.update_yaxes(range=[y0, y1], title_text='y [mm] (forward)', scaleanchor='x', scaleratio=1, row=1, col=1)
    play = dict(frame=dict(duration=40, redraw=False), transition=dict(duration=0), fromcurrent=True, mode='immediate')
    fig.update_layout(
        title=title, height=620, template='plotly_white',
        updatemenus=[dict(type='buttons', x=0.0, y=-0.12, xanchor='left', direction='left', buttons=[
            dict(label='▶ Play', method='animate', args=[None, play]),
            dict(label='❚❚ Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])],
        sliders=[dict(x=0.12, y=-0.08, len=0.88, currentvalue=dict(prefix='t = ', suffix=' ms'),
                      steps=[dict(label=f.name, method='animate',
                                  args=[[f.name], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                             for f in frames])])
    fig.write_html(path, include_plotlyjs=True, auto_play=False)   # self-contained, works offline
    return path


def make_gif(traj, path, title='', max_frames=120, fps=25):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ts, xy = traj['ts'], traj['xy']
    loop = np.append(traj['frame_nodes'], traj['frame_nodes'][0])
    com = _com(xy)
    idx = _frame_indices(len(ts), max_frames)
    (x0, y0), (x1, y1) = _axes_range(xy)

    speed, v_fwd, mean_fwd = _com_speed(ts, com)

    fig, (ax, sax) = plt.subplots(2, 1, figsize=(5.5, 7.5), gridspec_kw=dict(height_ratios=[3, 1]))
    sax.plot(ts * 1e3, speed, color='green', lw=1.5, label='|v| COM')
    sax.plot(ts * 1e3, v_fwd, color='purple', lw=1, ls='--', label='forward v (+y)')
    sax.set_xlabel('t [ms]'); sax.set_ylabel('speed [mm/s]')
    sax.set_title(f'mean forward {mean_fwd:.1f} mm/s', fontsize=9)
    sax.legend(fontsize=7, loc='upper right')
    now_line = sax.axvline(ts[idx[0]] * 1e3, color='black', lw=1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm] (forward)')
    frame_line, = ax.plot([], [], color='#555', lw=5)
    limb_lines = {n: ax.plot([], [], '-o', ms=3, lw=2.5, color=LIMB_COLORS.get(n, 'k'))[0] for n in traj['limbs']}
    com_path, = ax.plot([], [], ':', color='green')
    label = ax.set_title(f'{title}\nt')   # two-line placeholder so tight_layout leaves room
    fig.tight_layout()

    def draw(k):
        p = xy[k] * 1e3
        frame_line.set_data(p[loop, 0], p[loop, 1])
        for n, chain in traj['limbs'].items():
            limb_lines[n].set_data(p[chain, 0], p[chain, 1])
        com_path.set_data(com[:k + 1, 0] * 1e3, com[:k + 1, 1] * 1e3)
        label.set_text(f'{title}\nt = {ts[k]*1e3:.0f} ms   |v| = {speed[k]:.1f} mm/s')
        now_line.set_xdata([ts[k] * 1e3] * 2)
        return [frame_line, com_path, now_line, *limb_lines.values()]

    FuncAnimation(fig, draw, frames=idx, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def make_viewer(traj, out_base, title=''):
    html = make_html(traj, out_base + '.html', title)
    gif = make_gif(traj, out_base + '.gif', title.split(':')[0])
    print('viewer:', html)
    print('preview:', gif)
    return html, gif


def _find_saved_runs():
    """Saved trajectories (newest first) under this folder's outputs/."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    runs = []
    for f in glob.glob(os.path.join(here, 'outputs', '**', '*.npz'), recursive=True):
        with np.load(f) as d:
            if 'limb_names' in d.files:
                runs.append(f)
    return sorted(runs, key=os.path.getmtime, reverse=True)


def _resolve(path):
    """Accept a path relative to the current folder or to this script's folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (path, os.path.join(here, path)):
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f'{path} not found (also looked in {here})')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        src = _resolve(sys.argv[1])
    else:
        runs = _find_saved_runs()
        if not runs:
            sys.exit('No saved runs yet. Run step1_run.py, or validate.py then view_case.py, first.\n'
                     'Usage: python frog_viewer.py [outputs/<run>.npz]')
        src = runs[0]
        print('No file given; showing the most recent run. Others available:')
        for r in runs[1:]:
            print('   ', os.path.relpath(r))
    make_viewer(load_trajectory(src), os.path.splitext(src)[0], os.path.basename(src))
