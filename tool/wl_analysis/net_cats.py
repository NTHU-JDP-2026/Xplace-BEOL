"""Routed wire length of two routed DEFs by net category: did they route the same netlist?

    python3 tool/wl_analysis/net_cats.py <a.def> <b.def> [--scale 4]

Categories (per DEF): clock (USE CLOCK), signal nets with the same pins in both, signal nets in
both with different pins, nets in this DEF only. Optimization after placement (e.g. the
optDesign -postCTS inside Innovus' ccopt_design) rebuilds high-fanout and DRV-violating nets
with buffers, differently for each placement: the last two categories show how much of the
difference comes from there (NS3K jpeg, full flow: ~30% of the routed length). The nets with
different pins whose length differs most are listed.
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from defio import read_def  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('defs', nargs=2)
ap.add_argument('--scale', type=float, default=4.0, help='LEF scale of the DEFs (NS3K Innovus: 4x)')
a = ap.parse_args()
A, B = [read_def(f) for f in a.defs]


def cats(X, Y):
    dbu, _, _, nets = X
    other = Y[3]
    c = collections.defaultdict(lambda: [0, 0.0])
    for n, (conns, wl, clk) in nets.items():
        if clk:
            k = 'clock'
        elif n in other and other[n][0] == conns and not other[n][2]:
            k = 'same pins'
        elif n in other:
            k = 'diff pins'
        else:
            k = 'only here'
        c[k][0] += 1
        c[k][1] += wl / dbu / a.scale
    return c


ca, cb = cats(A, B), cats(B, A)
for k in ['same pins', 'diff pins', 'only here', 'clock']:
    print('%-10s a: %6d nets %8.0f um | b: %6d nets %8.0f um | b-a %+7.0f' % (
        k, ca[k][0], ca[k][1], cb[k][0], cb[k][1], cb[k][1] - ca[k][1]))
ta, tb = sum(v[1] for v in ca.values()), sum(v[1] for v in cb.values())
print('total      a: %8.0f   b: %8.0f   b-a %+.0f' % (ta, tb, tb - ta))
sa, sb = 1.0 / A[0] / a.scale, 1.0 / B[0] / a.scale
diff = sorted(((n, A[3][n][1] * sa, B[3][n][1] * sb, len(A[3][n][0]), len(B[3][n][0]))
               for n in A[3] if n in B[3] and A[3][n][0] != B[3][n][0] and not A[3][n][2]),
              key=lambda r: -abs(r[2] - r[1]))
for r in diff[:8]:
    print('  diff %-30s a %.0f um (%d pins)  b %.0f um (%d pins)' % r)
