"""Per-net decomposition of the routed wire length of two routed DEFs of the same netlist.

    python3 tool/wl_analysis/wl_decomp.py <cells.lef> <a.def> <b.def> [--scale 4] [--big]

Run inside the Xplace container (FLUTE from cpp_to_py). For every signal net with the same pins
in both DEFs (not USE CLOCK):
    HPWL    pin-based half perimeter (pin = mean of its LEF rect centers, orientation applied)
    RSMT    FLUTE rectilinear Steiner tree of the same pins (nets over 100 pins: rectilinear
            MST, >= RSMT, FLUTE crashes on thousands of pins)
    routed  the net's routed segments, all layers, vias not counted
in um at 1x (DEF units / dbu / scale), by net degree. routed / RSMT is the detour, RSMT / HPWL
the topology. --big: the nets of >= 31 pins one by one, largest RSMT difference first.
A placement-only DEF (no routing) gives routed = 0, the HPWL / RSMT columns still hold.
"""
import argparse
import collections
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, root)
sys.path.insert(0, here)
from cpp_to_py import flute_cpp  # noqa: E402
from defio import read_cell_pins, read_def, pin_xy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('lef')
ap.add_argument('defs', nargs=2)
ap.add_argument('--scale', type=float, default=4.0, help='LEF scale of the DEFs (NS3K Innovus: 4x)')
ap.add_argument('--big', action='store_true', help='list the nets of >= 31 pins one by one')
a = ap.parse_args()
flute_cpp.read_lut(os.path.join(root, 'thirdparty/flute/POWV9.dat'), os.path.join(root, 'thirdparty/flute/POST9.dat'))


def rmst(xs, ys):
    x = np.array(xs, dtype=np.int64)
    y = np.array(ys, dtype=np.int64)
    d = np.abs(x - x[0]) + np.abs(y - y[0])
    used = np.zeros(len(x), bool)
    used[0] = True
    tot = 0
    for _ in range(len(x) - 1):
        d[used] = np.iinfo(np.int64).max
        j = int(d.argmin())
        tot += int(d[j])
        used[j] = True
        d = np.minimum(d, np.abs(x - x[j]) + np.abs(y - y[j]))
    return tot


cells = read_cell_pins(a.lef)
D = [read_def(f) for f in a.defs]
common = [n for n, v in D[0][3].items() if not v[2] and n in D[1][3] and not D[1][3][n][2] and D[1][3][n][0] == v[0]]
print('nets: %d / %d in both with the same pins' % (len(common), len(D[0][3])))
bins = [(2, 2), (3, 3), (4, 5), (6, 10), (11, 30), (31, 100), (101, 10 ** 9)]
acc = [collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0]) for _ in D]
tot = [[0.0, 0.0, 0.0] for _ in D]
per_net = []
for n in common:
    conns = D[0][3][n][0]
    deg = len(conns)
    if deg < 2:
        continue
    b = next(b for b in bins if b[0] <= deg <= b[1])
    row = []
    for k, (dbu, comps, iopins, nets) in enumerate(D):
        xs, ys = [], []
        for inst, pin in conns:
            q = iopins.get(pin) if inst == 'PIN' else (pin_xy(cells, dbu, comps[inst], pin) if inst in comps else None)
            if q:
                xs.append(int(round(q[0])))
                ys.append(int(round(q[1])))
        s = 1.0 / dbu / a.scale
        h = (max(xs) - min(xs) + max(ys) - min(ys)) * s
        r = (flute_cpp.flute_rsmt_wl(xs, ys) if len(xs) <= 100 else rmst(xs, ys)) * s
        w = nets[n][1] * s
        acc[k][b][0] += 1
        acc[k][b][1] += h
        acc[k][b][2] += r
        acc[k][b][3] += w
        tot[k][0] += h
        tot[k][1] += r
        tot[k][2] += w
        row.append((h, r, w))
    per_net.append((n, deg, row))

names = [os.path.basename(f) for f in a.defs]
print('%-9s %6s | %-33s | %-33s | b - a (um): HPWL RSMT routed' % (
    'degree', 'nets', names[0] + ' HPWL/RSMT/routed', names[1] + ' HPWL/RSMT/routed'))
for b in bins:
    a0, a1 = acc[0][b], acc[1][b]
    if not a0[0]:
        continue
    lab = '%d' % b[0] if b[0] == b[1] else ('%d-%d' % b if b[1] < 10 ** 9 else '>=%d' % b[0])
    print('%-9s %6d | %8.0f %8.0f %8.0f %5.2f | %8.0f %8.0f %8.0f %5.2f | %+7.0f %+7.0f %+7.0f' % (
        lab, a0[0], a0[1], a0[2], a0[3], a0[3] / a0[2], a1[1], a1[2], a1[3], a1[3] / a1[2],
        a1[1] - a0[1], a1[2] - a0[2], a1[3] - a0[3]))
for k in range(2):
    h, r, w = tot[k]
    print('%s total: HPWL %.0f  RSMT %.0f  routed %.0f   RSMT/HPWL %.3f  routed/RSMT %s' % (
        names[k], h, r, w, r / h, '%.3f' % (w / r) if w else '-'))
(h0, r0, w0), (h1, r1, w1) = tot
print('b vs a: HPWL %+.1f%%  RSMT %+.1f%%' % (100 * (h1 / h0 - 1), 100 * (r1 / r0 - 1)) + (
    '  routed %+.1f%%  | routed difference: placement (RSMT) %+.0f um, detour %+.0f um' % (
        100 * (w1 / w0 - 1), (r1 - r0) * w0 / r0, w1 - w0 - (r1 - r0) * w0 / r0) if w0 else ''))

if a.big:
    print('nets of >= 31 pins: name degree | HPWL a b | RSMT a b | routed a b')
    for n, deg, row in sorted(per_net, key=lambda r: -(r[2][1][1] - r[2][0][1])):
        if deg >= 31:
            print('  %-40s %5d | %6.0f %6.0f | %6.0f %6.0f | %6.0f %6.0f' % (
                n[:40], deg, row[0][0], row[1][0], row[0][1], row[1][1], row[0][2], row[1][2]))
