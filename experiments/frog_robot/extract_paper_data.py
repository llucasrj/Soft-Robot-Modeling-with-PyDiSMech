"""
One-time extraction of the reference data from github.com/softmachineslab/frog
into paper_data/frog_reference.json (small, version-controllable).

    git clone https://github.com/softmachineslab/frog.git   (on Windows: git -c core.longpaths=true clone ...)
    python extract_paper_data.py <path-to-frog-repo>

Contents:
  profiles[egain][freq] = {t, kappa}      measured mean limb curvature over one cycle (1/m), Profile/<k>/<n>.mat
  sim[...]  = {egain, freq, phase, speed, t, y}   their DER sim: speed (mm/s, cycles 5-8) + COM y(t) every 10 ms
  exp[...]  = {egain, freq, phase, speed}          experimental speed (mm/s) from the .xlsx tables
"""
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def their_speed(t, y, freq):
    """Exactly find_max_speed.m: mean forward speed between t = 5/f and t = 8/f, in mm/s."""
    iS, iF = round(5 / freq * 1000) - 1, round(8 / freq * 1000) - 1
    return float((y[iF] - y[iS]) / (t[iF] - t[iS]) * 1000)


def read_xlsx(path):
    """Minimal .xlsx reader (no pandas/openpyxl): {sheet name: [[col0, col1], ...]} numeric rows."""
    z = zipfile.ZipFile(path)
    wb = ET.fromstring(z.read('xl/workbook.xml'))
    names = [s.get('name') for s in wb.find(NS + 'sheets')]
    out = {}
    for i, name in enumerate(names, 1):
        rows = []
        for r in ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml')).iter(NS + 'row'):
            vals = [float(c.find(NS + 'v').text) for c in r.findall(NS + 'c')
                    if c.find(NS + 'v') is not None and c.get('t') != 's']
            if len(vals) >= 2:
                rows.append(vals[:2])
        out[name] = rows
    return out


def main(repo):
    root = os.path.join(repo, 'frog_der_sim_exp_data')
    data = {'source': 'https://github.com/softmachineslab/frog', 'profiles': {}, 'sim': [], 'exp': []}

    for d in sorted(glob.glob(os.path.join(root, 'Profile', '[0-9]'))):
        egain = int(os.path.basename(d)) * 10
        for f in glob.glob(os.path.join(d, '*.mat')):
            m = sio.loadmat(f)
            freq = int(os.path.basename(f)[:-4]) / 10
            data['profiles'].setdefault(str(egain), {})[f'{freq:.1f}'] = {
                't': np.round(m['t'].ravel(), 6).tolist(), 'kappa': np.round(m['mean'].ravel(), 4).tolist()}

    seen = set()
    for f in glob.glob(os.path.join(root, 'sim', '**', '*.txt'), recursive=True):
        ph, fr, eg = map(float, re.search(r'phaseDelay_(\d+\.\d+)_frequency_(\d+\.\d+)_EGaIn_(\d+\.\d+)', f).groups())
        if (eg, fr, ph) in seen:
            continue
        seen.add((eg, fr, ph))
        t, x, y = np.loadtxt(f).T
        data['sim'].append(dict(egain=eg, freq=fr, phase=ph, speed=round(their_speed(t, y, fr), 3),
                                t=np.round(t[9::10], 4).tolist(), y=np.round(y[9::10] * 1e3, 4).tolist()))

    # Experimental speeds. speed.xlsx sheets: '0','1','3','4','Sheet5' -> 0,10,30,40,50 % EGaIn, phase 0.5
    sheet_egain = {'0': 0, '1': 10, '3': 30, '4': 40, 'Sheet5': 50}
    for name, rows in read_xlsx(os.path.join(root, 'exp', 'speed.xlsx')).items():
        for fr, v in rows:
            data['exp'].append(dict(egain=sheet_egain[name], freq=round(fr, 2), phase=0.5, speed=round(v, 3)))
    # Phase sweeps: first column is phase * 10
    for fn, eg, fr in [('exp/phase/speed_10_0dot4hz.xlsx', 10, 0.4), ('exp/speed_10_0dot8hz.xlsx', 10, 0.8),
                       ('exp/phase/speed_30_1dot2hz.xlsx', 30, 1.2)]:
        for rows in read_xlsx(os.path.join(root, fn)).values():
            for ph10, v in rows:
                data['exp'].append(dict(egain=eg, freq=fr, phase=round(ph10 / 10, 2), speed=round(v, 3), sweep='phase'))

    os.makedirs(os.path.join(HERE, 'paper_data'), exist_ok=True)
    out = os.path.join(HERE, 'paper_data', 'frog_reference.json')
    with open(out, 'w') as fh:
        json.dump(data, fh)
    n_prof = sum(len(v) for v in data['profiles'].values())
    print(f"wrote {out}: {n_prof} curvature profiles, {len(data['sim'])} sim runs, {len(data['exp'])} exp points, "
          f"{os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == '__main__':
    main(sys.argv[1])
