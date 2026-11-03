#!/usr/bin/env python3
"""Expected signal-PG violations (model) at the GP / legalized / routed stages of a run.
    python3 pinaccess/stage_check.py <model.pt> <cells.lef> <in.def> <stage.def>...
in.def gives the PDN; each stage DEF gives the COMPONENTS. Prints, per stage,
the orientation mix, the expected count with the DEF orientations (where the
DEF has real ones), with the flip averaged, and with the best flip per cell."""
import re, sys
import numpy as np, torch
sys.path.insert(0, '.')
from src.pin_access import PinAccessModel
ck, lef, in_def = sys.argv[1:4]
pa = PinAccessModel(ck, in_def, lef)
code = {'N': 0, 'FS': 1, 'FN': 2, 'S': 3}
for f in sys.argv[4:]:
    t = open(f).read()
    comp = re.search(r'\nCOMPONENTS \d+ ;(.*?)END COMPONENTS', t, re.S).group(1)
    names, types, xs, ys, ors = [], [], [], [], []
    for c in re.finditer(r'- (\S+) (\S+) .*?\+ (?:PLACED|FIXED) \( (-?\d+) (-?\d+) \) (\S+)', comp, re.S):
        if c.group(2) in pa.ptab:
            names.append(c.group(1)); types.append(c.group(2)); xs.append(int(c.group(3))); ys.append(int(c.group(4))); ors.append(c.group(5))
    pa.index_cells(types)
    xl = torch.tensor(xs, dtype=torch.float64, device='cuda'); yb = torch.tensor(ys, dtype=torch.float64, device='cuda')
    mix = {o: ors.count(o) for o in code}
    with torch.no_grad():
        pdef, _ = pa.score(xl, yb, orient=torch.tensor([code[o] for o in ors], device='cuda'))
        pavg, _ = pa.score(xl, yb)
        # best flip per cell: row orientation from parity, x-mirror chosen per cell
        P = pa.pins
        r = torch.round((yb[P['cid']] - pa.y0) / pa.rowh).long().clamp(0, pa.nrows - 1)
        o1 = torch.where(pa.rail_vdd[r] == P['bot'], 0, 1)
        pa_, _ = pa._score_o(xl, yb, o1); pb_, _ = pa._score_o(xl, yb, o1 + 2)
        nc = len(types)
        ca = torch.zeros(nc, device='cuda', dtype=pa_.dtype).index_add_(0, P['cid'], pa_)
        cb = torch.zeros(nc, device='cuda', dtype=pb_.dtype).index_add_(0, P['cid'], pb_)
        best = torch.minimum(ca, cb).sum()
    real = 'yes' if mix['FN'] + mix['S'] + mix['FS'] > 0 else 'no (all N)'
    print('%-60s orient %s  E[viol]: DEF orient %.1f (real orients: %s) | flip-avg %.1f | best flip %.1f'
          % (f.split('/pnr/out/')[-1], mix, pdef.sum().item(), real, pavg.sum().item(), best.item()))
