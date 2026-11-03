#!/usr/bin/env python3
"""Choose the x-mirror of each standard cell near the PDN before routing.

    python3 pinaccess/flip.py <model.pt> <cells.lef> <pdn.def> <placement.def> <out.flip> [--min_gain 0.05]

For every cell of placement.def with a signal pin near the PDN, the summed
signal-PG risk of its pins (src/pin_access.py) is computed for the row's
orientation (N / FS, kept) and its x-mirror (FN / S); the lower one is
written to out.flip ("instance orientation") when it differs from the
current orientation and lowers the risk by at least --min_gain.

--shift_thr T (with --shift_sites S): a cell whose risk stays above T in its
better orientation (both mirrors fail, e.g. XNOR2 under an M3 stripe) is also
scored at x shifts of 1..S sites either way, both mirrors; the option with the
lowest risk + --shift_cost per site is written as "instance SHIFT dx orientation"
(dx in sites, same row) when it lowers the risk by at least --min_gain.
pnr/pa_flip.tcl (PA_SHIFT=1) places it there and lets refinePlace legalize
the other cells around it.
"""
import argparse, re, sys
import torch
sys.path.insert(0, '.')
from src.pin_access import PinAccessModel

ap = argparse.ArgumentParser()
ap.add_argument('model', help='pin-access model checkpoint (ignored with --rule)'); ap.add_argument('lef'); ap.add_argument('pdn_def'); ap.add_argument('place_def'); ap.add_argument('out')
ap.add_argument('--min_gain', type=float, default=0.05)
ap.add_argument('--rule', default='', choices=['', 'block', 'track', 'track23'],
                help='rule-based baseline instead of the model: block = pins with a PG shape (M2-M4, any track) '
                     'within --rule_w half-sites of the pin centre; track / track23 = pins whose every M2 access track is '
                     'covered there (track23: on M2 or M3) (src/pin_access.py _rule_score)')
ap.add_argument('--rule_w', type=int, default=1)
ap.add_argument('--shift_thr', type=float, default=0.0, help='also shift cells whose best-mirror risk exceeds this (0 = off)')
ap.add_argument('--shift_sites', type=int, default=4)
ap.add_argument('--shift_cost', type=float, default=0.01, help='risk-equivalent cost per site of shift')
a = ap.parse_args()
# with --rule the model is not loaded (its path, e.g. "none", is ignored)
pa = PinAccessModel(None if a.rule else a.model, a.pdn_def, a.lef)
pa.rule, pa.rule_w = a.rule or None, a.rule_w
t = open(a.place_def).read()
comp = re.search(r'\nCOMPONENTS \d+ ;(.*?)END COMPONENTS', t, re.S).group(1)
names, types, xs, ys, ors = [], [], [], [], []
for c in re.finditer(r'- (\S+) (\S+) .*?\+ PLACED \( (-?\d+) (-?\d+) \) (\S+)', comp, re.S):
    if c.group(2) in pa.ptab and c.group(5) in ('N', 'FS', 'FN', 'S'):
        names.append(c.group(1)); types.append(c.group(2)); xs.append(int(c.group(3))); ys.append(int(c.group(4))); ors.append(c.group(5))
n = len(names)
pa.index_cells(types)
dev = pa.device
xl = torch.tensor(xs, dtype=torch.float64, device=dev); yb = torch.tensor(ys, dtype=torch.float64, device=dev)
code = {'N': 0, 'FS': 1, 'FN': 2, 'S': 3}
cur = torch.tensor([code[o] for o in ors], device=dev)
base = cur % 2                       # row orientation: N (0) / FS (1)
P = pa.pins
with torch.no_grad():
    risk = []
    for opt in (base, base + 2):     # as is in the row, x-mirrored
        p, _ = pa._score_o(xl, yb, opt[P['cid']])
        risk.append(torch.zeros(n, dtype=p.dtype, device=dev).index_add(0, P['cid'], p))
    r0, r1 = risk
best = torch.where(r1 + a.min_gain < r0, base + 2, base)
rcur = torch.where(cur >= 2, r1, r0)
rbest = torch.minimum(r0, r1)
# keep the current orientation unless the other one is clearly better
keep = rcur <= rbest + a.min_gain
best = torch.where(keep, cur, best)
change = (best != cur).nonzero().squeeze(1).tolist()
inv = {v: k for k, v in code.items()}
shifts = []
rfinal = torch.where(best >= 2, r1, r0)
if a.shift_thr > 0:
    site = int(re.search(r'^ROW \S+ \S+ .* STEP (\d+) \d+', t, re.M).group(1))
    cand = (rbest > a.shift_thr).nonzero().squeeze(1)
    if cand.numel():
        bestv = rbest[cand].clone(); bestdx = torch.zeros_like(cand); besto = best[cand].clone()
        with torch.no_grad():
            for dx in [d for k in range(1, a.shift_sites + 1) for d in (-k, k)]:
                for opt in (base, base + 2):
                    p, _ = pa._score_o(xl + dx * site, yb, opt[P['cid']])
                    rr = torch.zeros(n, dtype=p.dtype, device=dev).index_add(0, P['cid'], p)[cand] + a.shift_cost * abs(dx)
                    better = rr < bestv
                    bestv = torch.where(better, rr, bestv)
                    bestdx = torch.where(better, torch.full_like(bestdx, dx), bestdx)
                    besto = torch.where(better, opt[cand], besto)
        ok = (bestdx != 0) & (bestv + a.min_gain < rbest[cand])
        for j in ok.nonzero().squeeze(1).tolist():
            i = int(cand[j])
            shifts.append((i, int(bestdx[j]), int(besto[j])))
            rfinal[i] = bestv[j] - a.shift_cost * abs(int(bestdx[j]))
shifted = {i for i, _, _ in shifts}
with open(a.out, 'w') as f:
    for i in change:
        if i not in shifted:
            f.write('%s %s\n' % (names[i], inv[int(best[i])]))
    for i, dx, o in shifts:
        f.write('%s SHIFT %d %s\n' % (names[i], dx, inv[o]))
if a.shift_thr > 0:
    print('pa shift: %d cells over %.2f in both mirrors, %d shifted (|dx| <= %d sites)' % (
        int((rbest > a.shift_thr).sum()), a.shift_thr, len(shifts), a.shift_sites))
print(('pa flip: %d cells, %d flipped; ' + ('rule %s/%d blocked pins' % (a.rule, a.rule_w) if a.rule else 'expected signal-PG violations') + ' %.1f -> %.1f')
      % (n, len(change), float(rcur.sum()), float(rfinal.sum())))
