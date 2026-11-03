#!/usr/bin/env python3
"""Offline check of src/pin_access.py on a routed placement: score its pins and
compare with the extract_grid.py labels (signal-PG markers).
    python3 pinaccess/check_module.py <routed.def> <cells.lef> <grid prefix> <model.pt>"""
import gzip, re, sys
import numpy as np, torch
sys.path.insert(0, '.')
from src.pin_access import PinAccessModel
d, lef, pre, ck = sys.argv[1:5]
pa = PinAccessModel(ck, d, lef)
t = open(d).read()
comp = re.search(r'\nCOMPONENTS \d+ ;(.*?)END COMPONENTS', t, re.S).group(1)
names, types, xs, ys, ors = [], [], [], [], []
for c in re.finditer(r'- (\S+) (\S+) .*?\+ (?:PLACED|FIXED) \( (-?\d+) (-?\d+) \) (\S+)', comp, re.S):
    if c.group(2) in pa.ptab:
        names.append(c.group(1).replace('\\', '')); types.append(c.group(2)); xs.append(int(c.group(3))); ys.append(int(c.group(4))); ors.append(c.group(5))
n = pa.index_cells(types)
xl = torch.tensor(xs, dtype=torch.float64, device='cuda'); yb = torch.tensor(ys, dtype=torch.float64, device='cuda')
r = torch.round((yb - pa.y0) / pa.rowh).long()
isN = (pa.rail_vdd[r] == torch.tensor([1 if pa.ptab[ty]['bottom'] == 'VDD' else 0 for ty in types], device='cuda'))
act = np.array([o in ('N', 'FN') for o in ors])
print('orientation (row parity rule vs DEF, N/FN vs FS/S): agree %.4f' % (isN.cpu().numpy() == act).mean())
code = {'N': 0, 'FS': 1, 'FN': 2, 'S': 3}
oc = torch.tensor([code[o] for o in ors], device='cuda')
p, sel = pa.score(xl, yb, orient=oc)
pavg, _ = pa.score(xl, yb)
p = p.detach().cpu().numpy(); sel = sel.cpu().numpy()
lab = {}
rows = gzip.open(pre + '.csv.gz', 'rt').read().splitlines()[1:]
ex = gzip.open(pre + '.extra.csv.gz', 'rt').read().splitlines()[1:]
for line, e in zip(rows, ex):
    v = line.split(','); lab[(v[1], v[3])] = int(e.split(',')[0])
key = [(names[i], pn) for i, pn in pa.pin_names]
yy = np.array([lab.get(k, -1) for k in key])
m = yy >= 0
print('pins scored %d, selected (PDN in window) %d; in dataset %d; dataset pins selected %.4f' % (n, sel.sum(), m.sum(), sel[m].mean()))
def ap(pp, t):
    o = np.argsort(-pp, kind='mergesort'); ts = t[o]; tp = np.cumsum(ts)
    return float((tp[ts == 1] / (np.nonzero(ts == 1)[0] + 1)).sum() / max(ts.sum(), 1))
print('signal-PG positives %d, AP with the DEF orientation %.3f (predicted sum %.1f); flip-averaged AP %.3f (sum %.1f)' % (yy[m].sum(), ap(p[m], yy[m].astype(float)), p[m].sum(), ap(pavg.detach().cpu().numpy()[m], yy[m].astype(float)), pavg.detach().cpu().numpy()[m].sum()))
